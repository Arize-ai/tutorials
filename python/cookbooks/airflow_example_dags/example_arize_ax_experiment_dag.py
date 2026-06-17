"""
Example DAG: Arize AX experiment workflow (server-side).

Creates a dataset, registers a server-side ``run_experiment`` task that
calls gpt-5.5 with a mustache-templated prompt, chains a single
``template_evaluation`` accuracy judge that scores each row server-side,
and lists the resulting experiments.

Requires:
- Airflow connection ``arize_ax_default`` with API key.
- Airflow Variable ``arize_ax_space_id``.
- Airflow Variable ``arize_ai_integration_id`` (or env var
  ``ARIZE_AI_INTEGRATION_ID``) — OpenAI integration with gpt-5.5 access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxListExperimentsOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxCreateTaskOperator,
    ArizeAxDeleteTaskOperator,
)
from airflow.providers.arize_ax.utils.task_groups import (
    arize_ax_chained_experiment_eval,
)
from airflow.providers.standard.operators.python import PythonOperator

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_RUN_SUFFIX = (
    "{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') "
    "| replace('.', '') | replace('_', '') }}"
)

EXACT_MATCH_TEMPLATE = (
    "[Question]: {query}\n"
    "[Expected]: {expected_output}\n"
    "[Output]: {output}\n\n"
    "Reply with ONLY 'correct' or 'incorrect'."
)


def _resolve_integration_id() -> str:
    return (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )


def _build_judge_config(**_ctx) -> dict[str, Any]:
    return {
        "name": "exact_match_judge",
        "template": EXACT_MATCH_TEMPLATE,
        "include_explanations": True,
        "use_function_calling_if_available": False,
        "classification_choices": {"correct": 1.0, "incorrect": 0.0},
        "llm_config": {
            "ai_integration_id": _resolve_integration_id(),
            "model_name": "gpt-5.4-mini",
            "invocation_parameters": {},
            "provider_parameters": {},
        },
    }


def _build_run_config(**_ctx) -> dict[str, Any]:
    return {
        "experiment_type": "llm_generation",
        "ai_integration_id": _resolve_integration_id(),
        "model_name": "gpt-5.5",
        "messages": [
            {"role": "system",
             "content": "Answer with only the direct factual answer."},
            {"role": "user", "content": "{{query}}"},
        ],
        "input_variable_format": "mustache",
        "invocation_parameters": {},
        "provider_parameters": {},
    }


with DAG(
    dag_id="example_arize_ax_experiment",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "experiment", "eval-hub"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id=_SPACE_JINJA,
        name="airflow-experiment-dataset",
        examples=[
            {"query": "What is 2+2?", "expected_output": "4"},
            {"query": "Capital of France?", "expected_output": "Paris"},
            {"query": "Largest planet?", "expected_output": "Jupiter"},
        ],
        if_exists="skip",
    )

    build_judge_config = PythonOperator(
        task_id="build_judge_config",
        python_callable=_build_judge_config,
    )

    create_judge_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_judge_evaluator",
        space_id=_SPACE_JINJA,
        name=f"exact_match_judge_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial exact_match judge",
        template_config_task_id="build_judge_config",
        description="LLM-as-judge accuracy evaluator.",
    )

    create_judge_task = ArizeAxCreateTaskOperator(
        task_id="create_judge_task",
        name=f"exact-match-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_judge_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    build_run_config = PythonOperator(
        task_id="build_run_config",
        python_callable=_build_run_config,
    )

    create_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"airflow-run-exp-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_run_config') }}",
        if_exists="skip",
    )

    run_experiment = arize_ax_chained_experiment_eval(
        group_id="run_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_run_exp_task') }}",
        experiment_name=f"airflow-nightly-eval-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_judge_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    list_experiments = ArizeAxListExperimentsOperator(
        task_id="list_experiments",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=5,
        trigger_rule="all_done",
    )

    cleanup_judge_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_judge_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_judge_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_judge_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_judge_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_judge_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    create_dataset >> [build_judge_config, build_run_config]
    build_judge_config >> create_judge_evaluator >> create_judge_task
    [create_dataset, build_run_config] >> create_run_exp_task
    [create_run_exp_task, create_judge_task] >> run_experiment
    run_experiment >> list_experiments
    list_experiments >> [cleanup_judge_task, cleanup_run_exp_task, cleanup_judge_evaluator]
