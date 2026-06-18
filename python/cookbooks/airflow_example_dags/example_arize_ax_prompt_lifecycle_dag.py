"""
Prompt Lifecycle Management DAG (server-side, self-contained):
version → evaluate → promote → re-evaluate → promote → archive.

The DAG implements a two-stage gated promotion ladder for an LLM prompt.
A *staging* experiment must clear an absolute accuracy threshold; if it
does, the prompt is labelled ``"staging"``. A *production* experiment
(with an improved system prompt) then runs and is compared against the
staging experiment; if it doesn't regress, the prompt is labelled
``"production"`` and a final archival notification fires.

It is fully self-contained — no pre-existing prompt, dataset, baseline
experiment, or thresholds Variable required. Everything is provisioned
and torn down on every run.

Lifecycle stages
----------------
0. **check_prereqs** — short-circuit if ``ARIZE_AI_INTEGRATION_ID`` isn't
   resolvable.
1. **create_eval_dataset** — provision the evaluation dataset.
2. **build_accuracy_judge_config / create_accuracy_evaluator /
   create_accuracy_eval_task** — the LLM-as-judge that scores each row.
3. **create_demo_prompt** — the prompt whose lifecycle we manage.
4. **Staging gate**:
   - ``build_staging_run_config / create_staging_run_exp_task`` register a
     ``run_experiment`` task driven by a *baseline* system prompt.
   - ``run_staging_experiment.{trigger,wait,get_result}`` fire the
     experiment with the accuracy judge chained server-side.
   - ``score_staging`` aggregates the average accuracy.
   - ``gate_passes_staging`` short-circuits if avg < ``STAGING_THRESHOLD``.
   - ``promote_to_staging`` tags the prompt with the ``"staging"`` label.
5. **Production gate** (runs *serially* after staging — avoids the Arize
   concurrent-trigger stall):
   - ``build_production_run_config / create_production_run_exp_task``
     register a ``run_experiment`` task with an *improved* system prompt.
   - ``run_production_experiment.{trigger,wait,get_result}`` fire it with
     the same chained accuracy judge.
   - ``score_production`` aggregates the average.
   - ``compare_production_to_staging`` checks whether production beats
     staging on the chained eval metric.
   - ``gate_passes_production`` short-circuits unless
     ``overall_passed=True``.
   - ``promote_to_production`` tags the prompt with the ``"production"``
     label.
6. **archive_notification** — final placeholder that logs the lifecycle
   outcome (Slack/webhook integration point).
7. **cleanup_*** — delete the per-run-ephemeral evaluator, eval task, the
   two run-experiment tasks, the dataset, and the demo prompt.

Schedule: ``None`` — triggered externally (e.g. on a prompt commit).

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_project_id`` — project ID used to scope the accuracy eval task.
- ``arize_ai_integration_id`` *or* env var ``ARIZE_AI_INTEGRATION_ID`` —
  OpenAI integration in Arize with gpt-4.1 access.

Constants:
- ``STAGING_THRESHOLD`` — minimum staging accuracy to pass the staging
  gate (default 0.6).
- ``PRODUCTION_PASS_THRESHOLD`` — minimum delta production-over-staging
  to pass the production gate (default 0.0, ties pass).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxCompareExperimentsOperator,
    ArizeAxGetExperimentScoreOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxDeletePromptOperator,
    ArizeAxPromotePromptOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxCreateTaskOperator,
    ArizeAxDeleteTaskOperator,
)
from airflow.providers.arize_ax.utils.task_groups import (
    arize_ax_chained_experiment_eval,
)
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

STAGING_THRESHOLD = 0.6
PRODUCTION_PASS_THRESHOLD = 0.0  # production must be >= staging + threshold

LIFECYCLE_EVAL_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"query": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
    {"query": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"query": "How many continents are there?", "expected_output": "7"},
]

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_RUN_SUFFIX = (
    "{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') "
    "| replace('.', '') | replace('_', '') }}"
)

ACCURACY_TEMPLATE = (
    "[Question]: {query}\n"
    "[Expected]: {expected_output}\n"
    "[Output]: {output}\n\n"
    "Reply with ONLY 'correct' or 'incorrect'."
)

STAGING_SYSTEM_PROMPT = (
    "You are an assistant. Answer the question briefly."
)
PRODUCTION_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer the question with only "
    "the direct factual answer — no extra explanation or punctuation."
)


def _resolve_integration_id() -> str:
    return (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )


def _check_prereqs(**_ctx) -> bool:
    if not _resolve_integration_id():
        print(
            "ARIZE_AI_INTEGRATION_ID env var or arize_ai_integration_id "
            "Variable must be set."
        )
        return False
    return True


def _build_accuracy_judge_config(**_ctx) -> dict[str, Any]:
    return {
        "name": "lifecycle_accuracy_judge",
        "template": ACCURACY_TEMPLATE,
        "include_explanations": True,
        "use_function_calling_if_available": False,
        "classification_choices": {"correct": 1.0, "incorrect": 0.0},
        "llm_config": {
            "ai_integration_id": _resolve_integration_id(),
            "model_name": "gpt-4o-mini",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        },
    }


def _build_run_config(model: str, system_prompt: str):
    def _callable(**_ctx) -> dict[str, Any]:
        return {
            "experiment_type": "llm_generation",
            "ai_integration_id": _resolve_integration_id(),
            "model_name": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                # run_experiment messages use mustache substitution.
                {"role": "user", "content": "{{query}}"},
            ],
            "input_variable_format": "mustache",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        }

    _callable.__name__ = f"_build_run_config_{model.replace('-', '_').replace('.', '_')}"
    return _callable


def _gate_passes_staging(**ctx) -> bool:
    """Return True iff staging avg score >= STAGING_THRESHOLD."""
    scores = ctx["ti"].xcom_pull(task_ids="score_staging") or {}
    if not isinstance(scores, dict) or not scores:
        print("[gate_passes_staging] No scores returned — failing gate.")
        return False
    avg_score = sum(scores.values()) / len(scores)
    print(f"[gate_passes_staging] avg={avg_score:.4f} threshold={STAGING_THRESHOLD}")
    return avg_score >= STAGING_THRESHOLD


def _gate_passes_production(**ctx) -> bool:
    """Return True iff compare_production_to_staging overall_passed=True."""
    result = ctx["ti"].xcom_pull(task_ids="compare_production_to_staging") or {}
    if not isinstance(result, dict):
        print(
            f"[gate_passes_production] compare XCom not a dict: {result!r} — "
            "failing gate."
        )
        return False
    overall_passed = bool(result.get("overall_passed", False))
    print(f"[gate_passes_production] overall_passed={overall_passed}")
    for metric, data in (result.get("results") or {}).items():
        print(
            f"  {metric}: production={data.get('candidate'):.4f}  "
            f"staging={data.get('baseline'):.4f}  "
            f"delta={data.get('delta'):.4f}  passed={data.get('passed')}"
        )
    return overall_passed


def _archive_notification(**ctx) -> dict[str, Any]:
    """Final placeholder: log the lifecycle outcome."""
    promote_prod = ctx["ti"].xcom_pull(task_ids="promote_to_production") or {}
    prompt_id = (
        promote_prod.get("prompt_id") if isinstance(promote_prod, dict) else None
    )
    print("=" * 60)
    print("PROMPT LIFECYCLE COMPLETE")
    print(f"  Prompt ID         : {prompt_id}")
    print("  Promoted to label : production")
    print(
        "  Archive step      : the prior 'production' version would be "
        "tagged 'archived' in production deployments."
    )
    print("=" * 60)
    return {"prompt_id": prompt_id, "archived": False}


_STAGING_EXP_ID = (
    "{{ ti.xcom_pull(task_ids='run_staging_experiment.trigger', "
    "key='result')['experiment_id'] }}"
)
_PRODUCTION_EXP_ID = (
    "{{ ti.xcom_pull(task_ids='run_production_experiment.trigger', "
    "key='result')['experiment_id'] }}"
)


with DAG(
    dag_id="arize_ax_prompt_lifecycle",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "prompts", "lifecycle", "experiments", "eval-hub", "self-contained"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    # ----- Phase 1: ephemeral dataset --------------------------------------
    create_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_eval_dataset",
        space_id=_SPACE_JINJA,
        name=f"lifecycle-eval-dataset-{_RUN_SUFFIX}",
        examples=LIFECYCLE_EVAL_EXAMPLES,
        if_exists="skip",
    )

    # ----- Phase 2: accuracy judge ------------------------------------------
    build_accuracy_judge_config = PythonOperator(
        task_id="build_accuracy_judge_config",
        python_callable=_build_accuracy_judge_config,
    )
    create_accuracy_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_accuracy_evaluator",
        space_id=_SPACE_JINJA,
        name=f"lifecycle_accuracy_judge_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial lifecycle accuracy judge",
        template_config_task_id="build_accuracy_judge_config",
        description="LLM-as-judge for prompt lifecycle gates.",
    )
    create_accuracy_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_accuracy_eval_task",
        name=f"lifecycle-accuracy-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_accuracy_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    # ----- Phase 3: demo prompt (target of promotion labels) ----------------
    create_demo_prompt = ArizeAxCreatePromptOperator(
        task_id="create_demo_prompt",
        space_id=_SPACE_JINJA,
        name=f"lifecycle-demo-prompt-{_RUN_SUFFIX}",
        messages=[
            {"role": "system", "content": PRODUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": "{query}"},
        ],
        model="gpt-4.1",
        commit_message="initial lifecycle prompt version",
        input_variable_format="f_string",
    )

    # ----- Phase 4: staging experiment (baseline-quality prompt) ------------
    build_staging_run_config = PythonOperator(
        task_id="build_staging_run_config",
        python_callable=_build_run_config("gpt-4.1", STAGING_SYSTEM_PROMPT),
    )
    create_staging_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_staging_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"lifecycle-staging-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_staging_run_config') }}",
        if_exists="skip",
    )
    run_staging_experiment = arize_ax_chained_experiment_eval(
        group_id="run_staging_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_staging_run_exp_task') }}",
        experiment_name=f"lifecycle-staging-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )
    score_staging = ArizeAxGetExperimentScoreOperator(
        task_id="score_staging",
        experiment_id=_STAGING_EXP_ID,
        aggregation="mean",
    )
    gate_passes_staging = ShortCircuitOperator(
        task_id="gate_passes_staging",
        python_callable=_gate_passes_staging,
        # Without this, the ShortCircuit cascades skip to ALL downstream
        # tasks (including the trigger_rule="all_done" cleanups) when the
        # gate doesn't pass — leaving ephemeral resources behind.
        ignore_downstream_trigger_rules=False,
    )
    promote_to_staging = ArizeAxPromotePromptOperator(
        task_id="promote_to_staging",
        prompt_name=f"lifecycle-demo-prompt-{_RUN_SUFFIX}",
        label="staging",
        space_id=_SPACE_JINJA,
    )

    # ----- Phase 5: production experiment (improved prompt) -----------------
    # Serially follows staging — avoids the Arize concurrent-trigger stall
    # we observed under parallel fan-out.
    build_production_run_config = PythonOperator(
        task_id="build_production_run_config",
        python_callable=_build_run_config("gpt-4.1", PRODUCTION_SYSTEM_PROMPT),
    )
    create_production_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_production_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"lifecycle-production-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_production_run_config') }}",
        if_exists="skip",
    )
    run_production_experiment = arize_ax_chained_experiment_eval(
        group_id="run_production_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_production_run_exp_task') }}",
        experiment_name=f"lifecycle-production-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )
    score_production = ArizeAxGetExperimentScoreOperator(
        task_id="score_production",
        experiment_id=_PRODUCTION_EXP_ID,
        aggregation="mean",
    )
    compare_production_to_staging = ArizeAxCompareExperimentsOperator(
        task_id="compare_production_to_staging",
        candidate_experiment_id=_PRODUCTION_EXP_ID,
        baseline_experiment_id=_STAGING_EXP_ID,
        pass_threshold=PRODUCTION_PASS_THRESHOLD,
        aggregation="mean",
    )
    gate_passes_production = ShortCircuitOperator(
        task_id="gate_passes_production",
        python_callable=_gate_passes_production,
        ignore_downstream_trigger_rules=False,
    )
    promote_to_production = ArizeAxPromotePromptOperator(
        task_id="promote_to_production",
        prompt_name=f"lifecycle-demo-prompt-{_RUN_SUFFIX}",
        label="production",
        space_id=_SPACE_JINJA,
    )

    archive_notification = PythonOperator(
        task_id="archive_notification",
        python_callable=_archive_notification,
    )

    # ----- Phase 6: cleanup -------------------------------------------------
    cleanup_accuracy_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_accuracy_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_staging_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_staging_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_staging_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_production_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_production_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_production_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_accuracy_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_accuracy_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_accuracy_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_demo_prompt = ArizeAxDeletePromptOperator(
        task_id="cleanup_demo_prompt",
        prompt_id="{{ ti.xcom_pull(task_ids='create_demo_prompt') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # Wiring
    check_prereqs >> [
        create_eval_dataset,
        build_accuracy_judge_config,
        build_staging_run_config,
        build_production_run_config,
        create_demo_prompt,
    ]
    build_accuracy_judge_config >> create_accuracy_evaluator >> create_accuracy_eval_task
    [create_eval_dataset, build_staging_run_config] >> create_staging_run_exp_task
    [create_eval_dataset, build_production_run_config] >> create_production_run_exp_task

    [create_staging_run_exp_task, create_accuracy_eval_task] >> run_staging_experiment
    run_staging_experiment >> score_staging >> gate_passes_staging
    [gate_passes_staging, create_demo_prompt] >> promote_to_staging

    # Production stage waits for the staging gate (and chain) to terminate
    # before triggering — avoids concurrent run_experiment triggers.
    [
        promote_to_staging,
        create_production_run_exp_task,
        create_accuracy_eval_task,
    ] >> run_production_experiment
    run_production_experiment >> score_production
    [score_staging, score_production] >> compare_production_to_staging
    compare_production_to_staging >> gate_passes_production
    [gate_passes_production, create_demo_prompt] >> promote_to_production
    promote_to_production >> archive_notification

    archive_notification >> [
        cleanup_accuracy_eval_task,
        cleanup_staging_run_exp_task,
        cleanup_production_run_exp_task,
        cleanup_accuracy_evaluator,
        cleanup_dataset,
        cleanup_demo_prompt,
    ]
