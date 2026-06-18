"""
Drift Detection & Auto-Rollback DAG (server-side, self-contained).

This DAG demonstrates the production watchdog pattern: a scheduled run
provisions a *baseline* experiment (a strong known-good config), then runs
a *current* experiment with a deliberately-degraded system prompt to make
drift actually fire. ``ArizeAxDetectEvalDriftOperator`` (with
``fail_on_drift=True``) raises ``AirflowException`` when the per-metric
delta exceeds the threshold, which triggers the rollback branch
(``trigger_rule="all_failed"``): the stable prompt version is re-promoted
to the ``"production"`` label, and a notification fires. Everything is
provisioned and torn down on every run — no external Variables required.

Pipeline stages
---------------
0. **check_prereqs** — short-circuit if ``ARIZE_AI_INTEGRATION_ID`` isn't
   resolvable.
1. **create_eval_dataset** — provision the evaluation dataset.
2. **build_accuracy_judge_config / create_accuracy_evaluator** — build the
   accuracy LLM-as-judge config and create an evaluator (per-run-unique
   name to avoid the soft-delete tombstone trap).
3. **create_accuracy_eval_task** — wrap the evaluator in a
   ``template_evaluation`` task.
4. **create_demo_prompt** — create the demo prompt that the rollback
   targets. Its single version is the "known stable" version we re-promote
   when drift fires.
5. **build_baseline_run_config / create_baseline_run_exp_task** — register
   a ``run_experiment`` task with a *good* system prompt.
6. **run_baseline_experiment.{trigger,wait,get_result}** — fire baseline
   with the accuracy judge chained server-side.
7. **build_current_run_config / create_current_run_exp_task** — register a
   ``run_experiment`` task with a deliberately-degraded "Always reply
   exactly 'I don't know.'" system prompt that drives accuracy to 0.
8. **run_current_experiment.{trigger,wait,get_result}** — fire current
   serially after baseline (avoids the Arize concurrent-trigger stall).
9. **detect_drift** — compare current vs baseline. With
   ``fail_on_drift=True`` and a 0.1 threshold this *raises*
   ``AirflowException`` because current_avg ≈ 0 vs baseline_avg ≈ 1.
10. **rollback_prompt** — fires only on ``ALL_FAILED`` of detect_drift.
    Re-promotes the stable prompt version to ``"production"``.
11. **notify_rollback** — logs the rollback outcome (``trigger_rule=ALL_DONE``).
12. **cleanup_*** — delete the per-run-ephemeral evaluator, eval task, the
    two run-experiment tasks, the dataset, and the demo prompt.

Schedule: ``@daily``

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_project_id`` — project ID used to scope the accuracy eval task.
- ``arize_ai_integration_id`` *or* env var ``ARIZE_AI_INTEGRATION_ID`` —
  OpenAI integration in Arize with gpt-4.1 access.

Constants:
- ``DRIFT_THRESHOLD`` — absolute score drop that triggers rollback
  (default 0.1).
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
    ArizeAxDetectEvalDriftOperator,
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
from airflow.task.trigger_rule import TriggerRule

DRIFT_THRESHOLD = 0.1

DRIFT_EVAL_EXAMPLES = [
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

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer the question with only "
    "the direct factual answer — no extra explanation or punctuation."
)
# Deliberately adversarial so current_avg drops to ~0 and drift > threshold.
CURRENT_SYSTEM_PROMPT = (
    "You are a confused assistant. Always reply with exactly the string "
    "\"I don't know.\" — never anything else."
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
        "name": "drift_accuracy_judge",
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


def _notify_rollback(**ctx) -> dict[str, Any]:
    """Log rollback confirmation; placeholder for Slack/email integration."""
    rollback = ctx["ti"].xcom_pull(task_ids="rollback_prompt") or {}
    prompt_id = rollback.get("prompt_id") if isinstance(rollback, dict) else None
    label = rollback.get("label") if isinstance(rollback, dict) else None
    print("=" * 60)
    print("PROMPT ROLLBACK COMPLETE")
    print(f"  Prompt ID : {prompt_id}")
    print(f"  Label     : {label}")
    print("=" * 60)
    return {"notified": True, "prompt_id": prompt_id, "label": label}


with DAG(
    dag_id="arize_ax_drift_detection",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "drift", "experiments", "prompts", "rollback", "eval-hub", "self-contained"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    # ----- Phase 1: dataset -------------------------------------------------
    create_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_eval_dataset",
        space_id=_SPACE_JINJA,
        name=f"drift-eval-dataset-{_RUN_SUFFIX}",
        examples=DRIFT_EVAL_EXAMPLES,
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
        name=f"drift_accuracy_judge_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial drift accuracy judge",
        template_config_task_id="build_accuracy_judge_config",
        description="LLM-as-judge for drift detection.",
    )
    create_accuracy_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_accuracy_eval_task",
        name=f"drift-accuracy-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_accuracy_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    # ----- Phase 3: demo prompt (the rollback target) -----------------------
    create_demo_prompt = ArizeAxCreatePromptOperator(
        task_id="create_demo_prompt",
        space_id=_SPACE_JINJA,
        name=f"drift-demo-prompt-{_RUN_SUFFIX}",
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": "{query}"},
        ],
        model="gpt-4.1",
        commit_message="known-stable prompt version (rollback target)",
        input_variable_format="f_string",
    )

    # ----- Phase 4: baseline experiment (good config) -----------------------
    build_baseline_run_config = PythonOperator(
        task_id="build_baseline_run_config",
        python_callable=_build_run_config("gpt-4.1", BASELINE_SYSTEM_PROMPT),
    )
    create_baseline_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_baseline_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"drift-baseline-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_baseline_run_config') }}",
        if_exists="skip",
    )
    run_baseline_experiment = arize_ax_chained_experiment_eval(
        group_id="run_baseline_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        experiment_name=f"drift-baseline-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    # ----- Phase 5: current (deliberately degraded) experiment --------------
    build_current_run_config = PythonOperator(
        task_id="build_current_run_config",
        python_callable=_build_run_config("gpt-4.1", CURRENT_SYSTEM_PROMPT),
    )
    create_current_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_current_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"drift-current-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_current_run_config') }}",
        if_exists="skip",
    )
    run_current_experiment = arize_ax_chained_experiment_eval(
        group_id="run_current_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_current_run_exp_task') }}",
        experiment_name=f"drift-current-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    # ----- Phase 6: drift detection (expected to fire) ----------------------
    _BASELINE_EXP_ID = (
        "{{ ti.xcom_pull(task_ids='run_baseline_experiment.trigger', "
        "key='result')['experiment_id'] }}"
    )
    _CURRENT_EXP_ID = (
        "{{ ti.xcom_pull(task_ids='run_current_experiment.trigger', "
        "key='result')['experiment_id'] }}"
    )
    detect_drift = ArizeAxDetectEvalDriftOperator(
        task_id="detect_drift",
        current_experiment_id=_CURRENT_EXP_ID,
        baseline_experiment_id=_BASELINE_EXP_ID,
        drift_threshold=DRIFT_THRESHOLD,
        aggregation="mean",
        fail_on_drift=True,
    )

    # ----- Phase 7: rollback (only fires when drift detected) ---------------
    rollback_prompt = ArizeAxPromotePromptOperator(
        task_id="rollback_prompt",
        prompt_name=f"drift-demo-prompt-{_RUN_SUFFIX}",
        label="production",
        space_id=_SPACE_JINJA,
        trigger_rule=TriggerRule.ALL_FAILED,
    )
    notify_rollback = PythonOperator(
        task_id="notify_rollback",
        python_callable=_notify_rollback,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ----- Phase 8: cleanup -------------------------------------------------
    cleanup_accuracy_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_accuracy_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_baseline_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_baseline_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_current_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_current_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_current_run_exp_task') }}",
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

    # Wiring — baseline runs first, then current (serially) to avoid the
    # Arize concurrent-trigger stall we observed under parallel fan-out.
    check_prereqs >> [
        create_eval_dataset,
        build_accuracy_judge_config,
        build_baseline_run_config,
        build_current_run_config,
        create_demo_prompt,
    ]
    build_accuracy_judge_config >> create_accuracy_evaluator >> create_accuracy_eval_task
    [create_eval_dataset, build_baseline_run_config] >> create_baseline_run_exp_task
    [create_eval_dataset, build_current_run_config] >> create_current_run_exp_task
    [create_baseline_run_exp_task, create_accuracy_eval_task] >> run_baseline_experiment
    [run_baseline_experiment, create_current_run_exp_task, create_accuracy_eval_task] >> run_current_experiment
    run_current_experiment >> detect_drift >> rollback_prompt >> notify_rollback
    notify_rollback >> [
        cleanup_accuracy_eval_task,
        cleanup_baseline_run_exp_task,
        cleanup_current_run_exp_task,
        cleanup_accuracy_evaluator,
        cleanup_dataset,
        cleanup_demo_prompt,
    ]
