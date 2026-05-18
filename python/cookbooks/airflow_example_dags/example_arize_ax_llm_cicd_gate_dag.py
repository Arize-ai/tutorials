"""
LLM CI/CD Gate DAG: automated evaluation-gated promotion for LLM changes.

This DAG implements a CI/CD gate pattern for LLM-powered applications.  It
runs a candidate experiment against a held-out evaluation dataset, scores both
the candidate and an established baseline experiment, then raises AirflowException
if the candidate does not meet the improvement threshold.

Pipeline stages
---------------
0. **check_baseline_configured** — short-circuit if
   ``arize_ax_baseline_experiment_id`` Variable is not set.
1. **create_candidate_dataset** — create (or skip if exists) the evaluation
   dataset for this run.
2. **append_candidate_examples** — load evaluation examples into the dataset.
3. **run_candidate_experiment** — run the candidate LLM task against the
   dataset with Python evaluators; returns a scalar experiment ID on XCom.
4. **wait_for_candidate** — block until the experiment has at least
   ``CICD_MIN_RUNS`` completed runs (ArizeAxExperimentRunCountSensor).
5. **score_candidate** — aggregate evaluation scores for the candidate
   (ArizeAxGetExperimentScoreOperator).
6. **score_baseline** — aggregate evaluation scores for the baseline
   experiment (same operator, different experiment ID).
7. **compare_experiments** — compute per-metric deltas and an overall pass/
   fail verdict (ArizeAxCompareExperimentsOperator).  Raises AirflowException
   when ``fail_on_regression=True`` and the candidate did not improve.
8. **check_prompt_configured** — short-circuit the promotion path if no
   ``arize_ax_prompt_name`` Variable is set (gate-only mode).
9. **promote_prompt** — tag the configured prompt version with the
   ``arize_ax_prompt_label`` label (ArizeAxPromotePromptOperator).  Only
   runs when the gate passed *and* a prompt name is configured.
10. **promote_notification** — placeholder for a real deployment hook; logs
    the promotion outcome (and which label was applied) to the Airflow task
    log.

Variables
---------
- ``arize_ax_space_id`` — Arize space ID. Optional; falls back to connection
  ``default_space`` when absent.
- ``arize_ax_baseline_experiment_id`` — experiment ID of the production
  baseline to compare against. **Required.** The DAG skips when not set.
- ``arize_ax_prompt_name`` — name of the prompt to promote on gate pass.
  Optional. When unset the DAG runs in gate-only mode (no promotion).
- ``arize_ax_prompt_label`` — label to apply to the promoted prompt version
  (default ``"production"``).

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Variable ``arize_ax_baseline_experiment_id`` set to a real experiment ID.
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
    ArizeAxPromotePromptOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxExperimentRunCountSensor,
)

# ---------------------------------------------------------------------------
# Constants — override via Airflow Variables in production
# ---------------------------------------------------------------------------
CICD_DATASET_NAME = "cicd-eval-dataset"
CICD_MIN_RUNS = 5
CICD_PASS_THRESHOLD = 0.0  # candidate score must be >= baseline + threshold

# Evaluation examples shared across all CI/CD gate runs.
CICD_EVAL_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"query": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
    {"query": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"query": "How many continents are there?", "expected_output": "7"},
]


def _check_baseline_configured(**ctx) -> bool:
    """Return True only if arize_ax_baseline_experiment_id Variable is set."""
    baseline_id = Variable.get("arize_ax_baseline_experiment_id", default_var=None)
    if not baseline_id or baseline_id.strip() in ("", "your-baseline-experiment-id"):
        print(
            "Set the arize_ax_baseline_experiment_id Variable to a real experiment ID "
            "to run this CI/CD gate pipeline."
        )
        return False
    return True


def _check_prompt_configured(**ctx) -> bool:
    """Return True only if arize_ax_prompt_name Variable is set.

    When unset, the DAG runs in gate-only mode: the regression gate still
    fires (compare_experiments raises on regression) but no prompt is
    promoted.  The promote_prompt and promote_notification tasks are skipped.
    """
    prompt_name = Variable.get("arize_ax_prompt_name", default_var=None)
    if not prompt_name or prompt_name.strip() in ("", "your-prompt-name"):
        print(
            "arize_ax_prompt_name Variable is not set — running in gate-only mode. "
            "Set it to a real prompt name to enable automatic promotion on pass."
        )
        return False
    return True


def _candidate_task(dataset_row: dict) -> dict:
    """Candidate LLM task.

    Replace this with a real LLM call in production (e.g. ``openai.chat``).
    This stub echoes the expected output to demonstrate the pipeline shape.
    """
    return {"output": dataset_row.get("expected_output", "")}


def _accuracy_evaluator(dataset_row: dict, output: Any):
    """Simple accuracy evaluator: 1.0 if output matches expected, 0.0 otherwise."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")
    actual = output.get("output", "") if isinstance(output, dict) else str(output)
    expected = dataset_row.get("expected_output", "")
    match = str(actual).strip().lower() == str(expected).strip().lower()
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={expected!r}, got={actual!r}",
    )


def _promote_notification(**ctx) -> dict[str, Any]:
    """Placeholder for a real deployment hook.

    In production, replace this with a call to your deployment API,
    a Slack notification, or a webhook to trigger your release pipeline.
    """
    candidate_id = ctx["ti"].xcom_pull(task_ids="run_candidate_experiment")
    scores = ctx["ti"].xcom_pull(task_ids="score_candidate") or {}
    promoted_version_id = ctx["ti"].xcom_pull(task_ids="promote_prompt")
    prompt_name = Variable.get("arize_ax_prompt_name", default_var=None)
    prompt_label = Variable.get("arize_ax_prompt_label", default_var="production")
    print("=" * 60)
    print("CANDIDATE PROMOTED")
    print(f"  Experiment ID : {candidate_id}")
    print(f"  Scores        : {scores}")
    print(f"  Prompt name   : {prompt_name}")
    print(f"  Prompt label  : {prompt_label}")
    print(f"  Prompt verId  : {promoted_version_id}")
    print("  Next step     : trigger deployment pipeline (not implemented in this stub).")
    print("=" * 60)
    return {
        "promoted": True,
        "experiment_id": candidate_id,
        "scores": scores,
        "prompt_name": prompt_name,
        "prompt_label": prompt_label,
        "prompt_version_id": promoted_version_id,
    }


with DAG(
    dag_id="arize_ax_llm_cicd_gate",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "cicd", "llm", "experiments"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Stage 0 — Guard: skip DAG if baseline experiment ID is not configured
    check_baseline_configured = ShortCircuitOperator(
        task_id="check_baseline_configured",
        python_callable=_check_baseline_configured,
    )

    # Stage 1 — Dataset setup
    create_candidate_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_candidate_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name=CICD_DATASET_NAME,
        # SDK 8.25+ rejects empty examples list at create time. Seed with a
        # single placeholder; real examples are loaded by the subsequent
        # append step.
        examples=[{"input": "_seed_", "expected": "_seed_"}],
        if_exists="skip",
    )

    append_candidate_examples = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_candidate_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_candidate_dataset') }}",
        examples=CICD_EVAL_EXAMPLES,
    )

    # Stage 2 — Run candidate experiment
    run_candidate_experiment = ArizeAxRunExperimentOperator(
        task_id="run_candidate_experiment",
        name="cicd-candidate-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_candidate_dataset') }}",
        task=_candidate_task,
        evaluators=[_accuracy_evaluator],
        concurrency=4,
    )

    # Stage 3 — Wait for runs to complete
    wait_for_candidate = ArizeAxExperimentRunCountSensor(
        task_id="wait_for_candidate",
        experiment_id="{{ ti.xcom_pull(task_ids='run_candidate_experiment') }}",
        min_runs=CICD_MIN_RUNS,
        poke_interval=15,
        timeout=600,
        mode="poke",
        soft_fail=True,
    )

    # Stage 4 — Score candidate and baseline
    score_candidate = ArizeAxGetExperimentScoreOperator(
        task_id="score_candidate",
        experiment_id="{{ ti.xcom_pull(task_ids='run_candidate_experiment') }}",
        aggregation="mean",
    )

    score_baseline = ArizeAxGetExperimentScoreOperator(
        task_id="score_baseline",
        experiment_id="{{ var.value.get('arize_ax_baseline_experiment_id', '') }}",
        aggregation="mean",
    )

    # Stage 5 — Compare and gate (raises AirflowException on regression)
    compare_experiments = ArizeAxCompareExperimentsOperator(
        task_id="compare_experiments",
        candidate_experiment_id="{{ ti.xcom_pull(task_ids='run_candidate_experiment') }}",
        baseline_experiment_id="{{ var.value.get('arize_ax_baseline_experiment_id', '') }}",
        pass_threshold=CICD_PASS_THRESHOLD,
        aggregation="mean",
        fail_on_regression=True,
    )

    # Stage 6a — Short-circuit promotion path if no prompt is configured
    check_prompt_configured = ShortCircuitOperator(
        task_id="check_prompt_configured",
        python_callable=_check_prompt_configured,
    )

    # Stage 6b — Promote the prompt version with the configured label
    promote_prompt = ArizeAxPromotePromptOperator(
        task_id="promote_prompt",
        prompt_name="{{ var.value.get('arize_ax_prompt_name', '') }}",
        label="{{ var.value.get('arize_ax_prompt_label', 'production') }}",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
    )

    # Stage 7 — Promotion notification (only reached on gate-pass + promote)
    promote_notification = PythonOperator(
        task_id="promote_notification",
        python_callable=_promote_notification,
    )

    # Wiring
    check_baseline_configured >> create_candidate_dataset >> append_candidate_examples >> run_candidate_experiment
    run_candidate_experiment >> wait_for_candidate
    wait_for_candidate >> [score_candidate, score_baseline]
    [score_candidate, score_baseline] >> compare_experiments
    compare_experiments >> check_prompt_configured >> promote_prompt >> promote_notification
