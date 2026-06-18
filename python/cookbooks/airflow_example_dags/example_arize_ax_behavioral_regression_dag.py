"""
Behavioral Regression Testing DAG (self-contained, server-side).

Creates a small demo dataset, runs **two server-side experiment tasks** (Arize
makes the LLM calls — no Python task callable runs on the Airflow worker),
then compares the two experiment runs with
``ArizeAxBehavioralRegressionOperator`` to detect output-length, refusal-rate,
or eval-score-distribution drift between baseline and candidate. When the
candidate diverges beyond ``significance_threshold`` the operator raises
``AirflowException`` (``fail_on_regression=True``) to block downstream
promotion in a real pipeline.

Baseline and candidate share the same ``RunConfiguration`` so the demo ends
green. To watch the gate fire, edit ``_build_candidate_run_config`` to use a
different model, system prompt, or invocation parameters.

Required configuration
----------------------
- Airflow connection ``arize_ax_default`` with a valid API key. ``space_id``
  is read from the connection's ``extra.default_space`` if present, else from
  the ``arize_ax_space_id`` Variable.
- ``ARIZE_AI_INTEGRATION_ID`` Airflow Variable (or worker env var) — the
  Arize AI Integration ID identifying the LLM provider/credentials used by
  the server-side experiment tasks. If absent, ``check_ai_integration``
  short-circuits cleanly and the rest of the DAG is skipped.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxBehavioralRegressionOperator,
    ArizeAxDeleteExperimentOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxGetTaskRunOperator,
    ArizeAxTriggerTaskRunOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import ArizeAxTaskRunSensor
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor
from airflow.task.trigger_rule import TriggerRule

_SPACE_ID = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_DEMO_EXAMPLES = [
    {"input": "What is 2+2?", "expected_output": "4"},
    {"input": "Capital of France?", "expected_output": "Paris"},
    {"input": "First president of the United States?", "expected_output": "George Washington"},
    {"input": "Chemical symbol for water?", "expected_output": "H2O"},
    {"input": "How many continents are there?", "expected_output": "7"},
    {"input": "Largest planet in the solar system?", "expected_output": "Jupiter"},
    {"input": "Speed of light in km/s (approx.)?", "expected_output": "300000"},
    {"input": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
]


def _get_ai_integration_id() -> str:
    """Worker env ``ARIZE_AI_INTEGRATION_ID`` wins; else Airflow Variable."""
    v = os.environ.get("ARIZE_AI_INTEGRATION_ID", "").strip()
    if v:
        return v
    return Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()


def _check_ai_integration(**_ctx) -> bool:
    if not _get_ai_integration_id():
        print(
            "ARIZE_AI_INTEGRATION_ID is not set in worker env or Airflow Variables — "
            "skipping behavioral-regression demo. Set it to your Arize AI Integration "
            "ID to enable."
        )
        return False
    return True


def _build_baseline_run_config(**_ctx) -> dict[str, Any]:
    """Build the server-side run config for the baseline experiment.

    Returns a plain dict matching the ``LlmGenerationRunConfig`` schema; the
    hook wraps it in the ``RunConfiguration`` SDK type before forwarding.
    Arize makes the LLM calls — no Python task runs on the Airflow worker.
    """
    return {
        "experiment_type": "llm_generation",
        "ai_integration_id": _get_ai_integration_id(),
        "model_name": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Answer in as few words as possible: {{input}}"},
        ],
        "input_variable_format": "mustache",
        "invocation_parameters": {"temperature": 0},
        "provider_parameters": {},
    }


def _build_candidate_run_config(**_ctx) -> dict[str, Any]:
    """Build the server-side run config for the candidate experiment.

    Identical to the baseline by default so the demo DAG ends green. To watch
    ``ArizeAxBehavioralRegressionOperator`` fire, change the model name, the
    prompt, or invocation parameters here.
    """
    return _build_baseline_run_config()


def _extract_experiment_id(task_id: str) -> str:
    """Pull ``experiment_id`` from an ``ArizeAxGetTaskRunOperator`` XCom payload."""
    return "{{ ti.xcom_pull(task_ids='" + task_id + "')['experiment_id'] }}"


with DAG(
    dag_id="arize_ax_behavioral_regression",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "regression", "behavioral", "llmops"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
    params={"significance_threshold": 0.05},
) as dag:

    check_ai_integration = ShortCircuitOperator(
        task_id="check_ai_integration",
        python_callable=_check_ai_integration,
    )

    create_demo_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_demo_dataset",
        name="behavioral-regression-demo-{{ ts_nodash }}",
        space_id=_SPACE_ID,
        examples=_DEMO_EXAMPLES,
        if_exists="skip",
    )

    build_baseline_run_config = PythonOperator(
        task_id="build_baseline_run_config",
        python_callable=_build_baseline_run_config,
    )

    create_baseline_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_baseline_task",
        name="behavioral-regression-baseline-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_baseline_run_config') }}",
        space_id=_SPACE_ID,
        if_exists="skip",
    )

    trigger_baseline_run = ArizeAxTriggerTaskRunOperator(
        task_id="trigger_baseline_run",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_task') }}",
        space_id=_SPACE_ID,
        experiment_name="baseline-{{ ts_nodash }}",
    )

    wait_for_baseline_run = ArizeAxTaskRunSensor(
        task_id="wait_for_baseline_run",
        run_id="{{ ti.xcom_pull(task_ids='trigger_baseline_run') }}",
        poke_interval=15,
        timeout=900,
        mode="reschedule",
    )

    get_baseline_run_result = ArizeAxGetTaskRunOperator(
        task_id="get_baseline_run_result",
        run_id="{{ ti.xcom_pull(task_ids='trigger_baseline_run') }}",
    )

    build_candidate_run_config = PythonOperator(
        task_id="build_candidate_run_config",
        python_callable=_build_candidate_run_config,
    )

    create_candidate_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_candidate_task",
        name="behavioral-regression-candidate-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_candidate_run_config') }}",
        space_id=_SPACE_ID,
        if_exists="skip",
    )

    trigger_candidate_run = ArizeAxTriggerTaskRunOperator(
        task_id="trigger_candidate_run",
        task_id_param="{{ ti.xcom_pull(task_ids='create_candidate_task') }}",
        space_id=_SPACE_ID,
        experiment_name="candidate-{{ ts_nodash }}",
    )

    wait_for_candidate_run = ArizeAxTaskRunSensor(
        task_id="wait_for_candidate_run",
        run_id="{{ ti.xcom_pull(task_ids='trigger_candidate_run') }}",
        poke_interval=15,
        timeout=900,
        mode="reschedule",
    )

    get_candidate_run_result = ArizeAxGetTaskRunOperator(
        task_id="get_candidate_run_result",
        run_id="{{ ti.xcom_pull(task_ids='trigger_candidate_run') }}",
    )

    behavioral_regression_check = ArizeAxBehavioralRegressionOperator(
        task_id="behavioral_regression_check",
        baseline_experiment_id=_extract_experiment_id("get_baseline_run_result"),
        candidate_experiment_id=_extract_experiment_id("get_candidate_run_result"),
        significance_threshold="{{ params.significance_threshold }}",
        fail_on_regression=True,
    )

    # Pause so users can inspect both experiments side-by-side in the Arize
    # UI before cleanup. ``mode="reschedule"`` releases the worker slot for
    # the duration of the wait. Runs regardless of whether the regression
    # check passed or failed.
    inspect_pause = TimeDeltaSensor(
        task_id="inspect_pause",
        delta=timedelta(minutes=5),
        mode="reschedule",
        trigger_rule="all_done",
    )

    cleanup_baseline_experiment = ArizeAxDeleteExperimentOperator(
        task_id="cleanup_baseline_experiment",
        experiment_id=_extract_experiment_id("get_baseline_run_result"),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    cleanup_candidate_experiment = ArizeAxDeleteExperimentOperator(
        task_id="cleanup_candidate_experiment",
        experiment_id=_extract_experiment_id("get_candidate_run_result"),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    cleanup_demo_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_demo_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    check_ai_integration >> create_demo_dataset
    create_demo_dataset >> build_baseline_run_config >> create_baseline_task
    create_baseline_task >> trigger_baseline_run >> wait_for_baseline_run
    wait_for_baseline_run >> get_baseline_run_result
    create_demo_dataset >> build_candidate_run_config >> create_candidate_task
    create_candidate_task >> trigger_candidate_run >> wait_for_candidate_run
    wait_for_candidate_run >> get_candidate_run_result
    [get_baseline_run_result, get_candidate_run_result] >> behavioral_regression_check
    behavioral_regression_check >> inspect_pause
    inspect_pause >> [
        cleanup_baseline_experiment,
        cleanup_candidate_experiment,
        cleanup_demo_dataset,
    ]
