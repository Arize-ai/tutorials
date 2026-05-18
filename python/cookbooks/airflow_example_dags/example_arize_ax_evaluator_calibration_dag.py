"""
Evaluator Calibration DAG

Runs an evaluator experiment against a dataset that has human annotations, then
measures how well the LLM evaluator's scores correlate with the ground truth.
When calibration quality is poor, ArizeAxEvaluatorCalibrationOperator raises
AirflowException (``fail_on_poor_calibration=True``) making the task fail (red)
and logs a detailed report.  A downstream ``notify_on_calibration_failure``
task fires only on failure (``trigger_rule=ONE_FAILED``) and is the place to
plug in Slack / PagerDuty / email alerts.

The bundled ``_simple_task`` and ``_simple_evaluator`` callables are *demo
stubs* — they just echo ``expected_output`` and do an exact-match score.
**For production**: replace them with a real LLM call (``openai.chat``,
``anthropic.messages.create``, ``litellm.completion``, etc.) and the
real evaluator you want to recalibrate.  The DAG shape stays the same.

Schedule defaults to ``@monthly``.  To change cadence, edit the
``schedule=`` argument below — DAG-level schedules can't be Variable-driven
in Airflow because they're parsed at module load time.

Requires:
  - Airflow Variable ``arize_ax_space_id``
  - Airflow Variable ``arize_ax_eval_dataset_id``
  - Airflow Variable ``arize_ax_metric_name``
  - Airflow Variable ``arize_ax_annotation_name``
  - Airflow connection ``arize_ax_default`` with valid API key

Operators exercised:
  ArizeAxRunExperimentOperator, ArizeAxExperimentRunCountSensor,
  ArizeAxEvaluatorCalibrationOperator
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable

try:
    from airflow.providers.standard.operators.python import (
        PythonOperator,
        ShortCircuitOperator,
    )
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

try:
    from airflow.utils.trigger_rule import TriggerRule
except ImportError:  # Airflow 3.x sdk path
    from airflow.sdk.definitions._internal.trigger_rule import TriggerRule

from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxEvaluatorCalibrationOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import ArizeAxExperimentRunCountSensor


def _check_dataset_configured(**ctx) -> bool:
    """Return True only if arize_ax_eval_dataset_id Variable is set to a real value."""
    dataset_id = Variable.get("arize_ax_eval_dataset_id", default_var=None)
    if not dataset_id or dataset_id.strip() in ("", "your-dataset-id"):
        print(
            "Set the arize_ax_eval_dataset_id Variable to a real dataset ID to run this pipeline."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# DEMO STUBS — replace for production use
# ---------------------------------------------------------------------------
# In production, ``_simple_task`` should call your real LLM (OpenAI, Anthropic,
# Bedrock, Vertex, LiteLLM, etc.) with the dataset row as input, and
# ``_simple_evaluator`` should compute the metric you actually want to
# recalibrate against the human annotations (LLM-as-judge, ROUGE, exact match,
# fuzzy match, hallucination score, etc.).  The Airflow + Arize operator
# wiring around them stays exactly the same.
# ---------------------------------------------------------------------------
def _simple_task(dataset_row: dict) -> dict:
    """[DEMO STUB] Echoes ``expected_output`` — replace with a real LLM call."""
    return {"output": dataset_row.get("expected_output", dataset_row.get("output", "no-answer"))}


def _simple_evaluator(dataset_row: dict, output: dict) -> Any:
    """[DEMO STUB] Exact-match scorer — replace with the real evaluator under test."""
    from arize.experiments import EvaluationResult

    actual = (output or {}).get("output", "")
    expected = dataset_row.get("expected_output", "")
    match = str(actual).strip().lower() == str(expected).strip().lower()
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={expected!r}, got={actual!r}",
    )


def _notify_on_calibration_failure(**ctx) -> None:
    """Demo alert hook — fires only when calibrate_evaluator failed.

    Replace this stub with a real alerting integration (Slack webhook,
    PagerDuty trigger, email via EmailOperator, etc.).  This task uses
    ``trigger_rule=ONE_FAILED`` so it only runs when calibration regressed.

    Alternative: for DAG-wide failure alerts, set ``on_failure_callback`` at
    the DAG level instead — both approaches are valid.
    """
    experiment_id = ctx["ti"].xcom_pull(task_ids="run_calibration_experiment")
    metric = Variable.get("arize_ax_metric_name", default_var="<unset>")
    annotation = Variable.get("arize_ax_annotation_name", default_var="<unset>")
    print("=" * 60)
    print("EVALUATOR CALIBRATION REGRESSED")
    print(f"  Experiment ID  : {experiment_id}")
    print(f"  Metric         : {metric}")
    print(f"  Annotation     : {annotation}")
    print("  Next step      : trigger alerting (not implemented in this stub).")
    print("=" * 60)



with DAG(
    dag_id="arize_ax_evaluator_calibration",
    start_date=datetime(2025, 1, 1),
    schedule="@monthly",
    tags=["arize_ax", "evaluator", "calibration", "llmops"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
    params={
        "min_samples": 10,
    },
) as dag:

    check_dataset_configured = ShortCircuitOperator(
        task_id="check_dataset_configured",
        python_callable=_check_dataset_configured,
    )

    run_calibration_experiment = ArizeAxRunExperimentOperator(
        task_id="run_calibration_experiment",
        name="calibration-run-{{ ts_nodash }}",
        dataset_id="{{ var.value.get('arize_ax_eval_dataset_id', '') }}",
        task=_simple_task,
        evaluators=[_simple_evaluator],
        concurrency=4,
    )

    wait_for_runs = ArizeAxExperimentRunCountSensor(
        task_id="wait_for_runs",
        experiment_id="{{ ti.xcom_pull(task_ids='run_calibration_experiment') }}",
        min_runs=1,
        poke_interval=15,
        timeout=600,
        mode="poke",
    )

    calibrate_evaluator = ArizeAxEvaluatorCalibrationOperator(
        task_id="calibrate_evaluator",
        experiment_id="{{ ti.xcom_pull(task_ids='run_calibration_experiment') }}",
        annotation_name="{{ var.value.get('arize_ax_annotation_name', 'human_quality') }}",
        metric_name="{{ var.value.get('arize_ax_metric_name', 'correctness') }}",
        min_samples="{{ params.min_samples }}",
        fail_on_poor_calibration=True,
    )

    notify_on_calibration_failure = PythonOperator(
        task_id="notify_on_calibration_failure",
        python_callable=_notify_on_calibration_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Wiring
    check_dataset_configured >> run_calibration_experiment >> wait_for_runs
    wait_for_runs >> calibrate_evaluator >> notify_on_calibration_failure
