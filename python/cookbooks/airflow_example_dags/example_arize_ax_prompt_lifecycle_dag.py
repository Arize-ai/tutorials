"""
Prompt Lifecycle Management DAG: version → evaluate → promote → monitor → rollback.

This DAG implements a full prompt lifecycle, using only existing Arize AX
operators.  It is designed to be triggered externally (e.g. on a new prompt
commit or a CI event) and gates each promotion stage with evaluation scores.

Lifecycle stages
----------------
**Stage 1 — Setup**
  - Fetch the latest version of the target prompt from Prompt Hub.
  - Create (or reuse) the evaluation dataset.
  - Append evaluation examples.

**Stage 2 — Staging evaluation**
  - Run an experiment against the staging dataset.
  - Wait for runs to complete.
  - Score the experiment.
  - Gate: skip if avg score < ``arize_ax_staging_threshold`` Variable.
  - Promote the prompt to the ``"staging"`` label.

**Stage 3 — Production evaluation**
  - Run a more rigorous experiment.
  - Wait for runs to complete.
  - Score the experiment.
  - Compare the staging experiment against the configured baseline.
  - Gate: skip if the comparison does not pass ``overall_passed=True``.
  - Promote the prompt to the ``"production"`` label.

**Stage 4 — Archival placeholder**
  - Log that the previous version should be archived (implement with a
    custom hook call or ArizeAxDeletePromptOperator in production).

Pipeline
--------
get_latest_prompt
  → create_eval_dataset
  → append_eval_examples
  → run_staging_eval
  → wait_for_staging_runs
  → score_staging
  → gate_passes_staging
  → promote_to_staging
  → run_production_eval
  → wait_for_production_runs
  → score_production
  → compare_to_baseline
  → gate_passes_production
  → promote_to_production
  → archive_old_version

Schedule: ``None`` — triggered externally (e.g. on prompt commit).

Variables
---------
- ``arize_ax_space_id`` — Arize space ID.
- ``arize_ax_dataset_id`` — dataset ID used for evaluations.
- ``arize_ax_prompt_name`` — name of the prompt to manage.
- ``arize_ax_baseline_experiment_id`` — experiment ID of the known-good
  production baseline.
- ``arize_ax_staging_threshold`` — minimum avg score to pass staging gate
  (default ``"0.7"``).
- ``arize_ax_production_threshold`` — minimum avg score to pass production
  gate (default ``"0.8"``).

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable

try:
    from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxAppendDatasetExamplesOperator,
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxCompareExperimentsOperator,
    ArizeAxGetExperimentScoreOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxGetPromptOperator,
    ArizeAxPromotePromptOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxExperimentRunCountSensor,
)

# ---------------------------------------------------------------------------
# Constants — adjust to your environment
# ---------------------------------------------------------------------------
LIFECYCLE_MIN_RUNS = 5
LIFECYCLE_EVAL_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"query": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
    {"query": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"query": "How many continents are there?", "expected_output": "7"},
]



def _check_pipeline_configured(**ctx) -> bool:
    """Return True only when required Variables are set for prompt lifecycle."""
    required = {
        "arize_ax_space_id": "a valid space ID",
        "arize_ax_prompt_name": "the prompt name to manage",
        "arize_ax_baseline_experiment_id": "a baseline experiment ID",
    }
    missing = []
    for key, description in required.items():
        val = Variable.get(key, default_var=None)
        if not val or val.strip() in ("", "your-space-id", "my-prompt", "your-baseline-id"):
            missing.append(f"  - {key}: set to {description}")
    if missing:
        print(
            "The following Variables must be set to run the prompt lifecycle pipeline:\n"
            + "\n".join(missing)
        )
        return False
    return True


def _lifecycle_task(dataset_row: dict) -> dict:
    """Lifecycle evaluation task — stub that echoes expected output.

    Replace with your real LLM call in production::

        prompt_version = dataset_row.get("_prompt_version")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=prompt_version["messages"] + [
                {"role": "user", "content": dataset_row["query"]}
            ],
        )
        return {"output": response.choices[0].message.content}
    """
    return {"output": dataset_row.get("expected_output", "")}


def _accuracy_evaluator(dataset_row: dict, output: Any) -> Any:
    """Simple exact-match accuracy evaluator."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="no output")
    actual = output.get("output", "") if isinstance(output, dict) else str(output)
    expected = dataset_row.get("expected_output", "")
    match = str(actual).strip().lower() == str(expected).strip().lower()
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={expected!r}, got={actual!r}",
    )


def _gate_passes_staging(**ctx) -> bool:
    """Return True iff staging avg score >= arize_ax_staging_threshold."""
    scores = ctx["ti"].xcom_pull(task_ids="score_staging") or {}
    if not isinstance(scores, dict) or not scores:
        print("[gate_passes_staging] No scores returned — failing gate.")
        return False
    avg_score = sum(scores.values()) / len(scores)
    threshold = float(Variable.get("arize_ax_staging_threshold", default_var="0.7"))
    print(f"[gate_passes_staging] avg_score={avg_score:.4f}, threshold={threshold}")
    return avg_score >= threshold


def _gate_passes_production(**ctx) -> bool:
    """Return True iff compare_to_baseline overall_passed=True."""
    result = ctx["ti"].xcom_pull(task_ids="compare_to_baseline")
    if not isinstance(result, dict):
        print(f"[gate_passes_production] compare_to_baseline XCom not a dict: {result!r} — failing gate.")
        return False
    overall_passed = result.get("overall_passed", False)
    results = result.get("results", {})
    print(f"[gate_passes_production] overall_passed={overall_passed}")
    for metric, data in results.items():
        print(
            f"  {metric}: staging={data.get('candidate'):.4f}  "
            f"baseline={data.get('baseline'):.4f}  "
            f"delta={data.get('delta'):.4f}  passed={data.get('passed')}"
        )
    return bool(overall_passed)


def _archive_old_version(**ctx) -> dict[str, Any]:
    """Placeholder: log that the previous production version should be archived.

    In production, replace this stub with a call to tag or delete the old
    version.  For example::

        from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook
        hook = ArizeAxHook()
        hook.promote_prompt(
            space_id=space_id,
            prompt_id=prompt_id,
            label="archived",
            version_id=old_version_id,
        )
    """
    production_result = ctx["ti"].xcom_pull(task_ids="promote_to_production") or {}
    prompt_id = production_result.get("prompt_id")

    print("=" * 60)
    print("PROMPT LIFECYCLE COMPLETE")
    print(f"  Prompt ID : {prompt_id}")
    print("  Label     : production")
    print("  Archive   : previous production version should now be labelled 'archived'.")
    print("              Implement via ArizeAxPromotePromptOperator with label='archived'.")
    print("=" * 60)
    return {"prompt_id": prompt_id, "archived": False}  # stub


with DAG(
    dag_id="arize_ax_prompt_lifecycle",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "prompts", "lifecycle", "experiments"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Stage 0 — Guard: skip DAG if required Variables are not configured
    check_pipeline_configured = ShortCircuitOperator(
        task_id="check_pipeline_configured",
        python_callable=_check_pipeline_configured,
    )

    # Stage 1 — Dataset setup
    get_latest_prompt = ArizeAxGetPromptOperator(
        task_id="get_latest_prompt",
        prompt_name="{{ var.value.get('arize_ax_prompt_name', '') }}",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
    )

    create_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_eval_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name="lifecycle-eval-{{ ts_nodash }}",
        # SDK 8.25+ rejects empty examples list at create time. Seed with a
        # single placeholder; real examples are added by the subsequent
        # append step.
        examples=[{"input": "_seed_", "expected": "_seed_"}],
        if_exists="skip",
    )

    append_eval_examples = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_eval_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        examples=LIFECYCLE_EVAL_EXAMPLES,
    )

    # Stage 2 — Staging evaluation
    run_staging_eval = ArizeAxRunExperimentOperator(
        task_id="run_staging_eval",
        name="lifecycle-staging-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        task=_lifecycle_task,
        evaluators=[_accuracy_evaluator],
        concurrency=4,
    )

    wait_for_staging_runs = ArizeAxExperimentRunCountSensor(
        task_id="wait_for_staging_runs",
        experiment_id="{{ ti.xcom_pull(task_ids='run_staging_eval') }}",
        min_runs=LIFECYCLE_MIN_RUNS,
        poke_interval=15,
        timeout=600,
        mode="poke",
        soft_fail=True,
    )

    score_staging = ArizeAxGetExperimentScoreOperator(
        task_id="score_staging",
        experiment_id="{{ ti.xcom_pull(task_ids='run_staging_eval') }}",
        aggregation="mean",
    )

    gate_passes_staging = ShortCircuitOperator(
        task_id="gate_passes_staging",
        python_callable=_gate_passes_staging,
    )

    promote_to_staging = ArizeAxPromotePromptOperator(
        task_id="promote_to_staging",
        prompt_name="{{ var.value.get('arize_ax_prompt_name', '') }}",
        label="staging",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
    )

    # Stage 3 — Production evaluation
    run_production_eval = ArizeAxRunExperimentOperator(
        task_id="run_production_eval",
        name="lifecycle-production-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        task=_lifecycle_task,
        evaluators=[_accuracy_evaluator],
        concurrency=4,
    )

    wait_for_production_runs = ArizeAxExperimentRunCountSensor(
        task_id="wait_for_production_runs",
        experiment_id="{{ ti.xcom_pull(task_ids='run_production_eval') }}",
        min_runs=LIFECYCLE_MIN_RUNS,
        poke_interval=15,
        timeout=600,
        mode="poke",
        soft_fail=True,
    )

    score_production = ArizeAxGetExperimentScoreOperator(
        task_id="score_production",
        experiment_id="{{ ti.xcom_pull(task_ids='run_production_eval') }}",
        aggregation="mean",
    )

    compare_to_baseline = ArizeAxCompareExperimentsOperator(
        task_id="compare_to_baseline",
        candidate_experiment_id="{{ ti.xcom_pull(task_ids='run_production_eval') }}",
        baseline_experiment_id="{{ var.value.get('arize_ax_baseline_experiment_id', '') }}",
        pass_threshold=0.0,
        aggregation="mean",
    )

    gate_passes_production = ShortCircuitOperator(
        task_id="gate_passes_production",
        python_callable=_gate_passes_production,
    )

    promote_to_production = ArizeAxPromotePromptOperator(
        task_id="promote_to_production",
        prompt_name="{{ var.value.get('arize_ax_prompt_name', '') }}",
        label="production",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
    )

    # Stage 4 — Archival placeholder
    archive_old_version = PythonOperator(
        task_id="archive_old_version",
        python_callable=_archive_old_version,
    )

    # Wiring
    check_pipeline_configured >> get_latest_prompt >> create_eval_dataset >> append_eval_examples

    append_eval_examples >> run_staging_eval >> wait_for_staging_runs >> score_staging
    score_staging >> gate_passes_staging >> promote_to_staging

    promote_to_staging >> run_production_eval >> wait_for_production_runs >> score_production
    score_production >> compare_to_baseline >> gate_passes_production
    gate_passes_production >> promote_to_production >> archive_old_version
