"""
Drift Detection & Auto-Rollback DAG: scheduled evaluation watchdog with prompt
rollback when drift is detected.

This DAG runs daily to guard against degradation in LLM evaluation scores.  It
creates a fresh "current" experiment, waits for it to complete, then compares
it against a known-good baseline experiment.  When drift is detected the
ArizeAxDetectEvalDriftOperator raises AirflowException (``fail_on_drift=True``)
causing the task to fail (red), making the drift event visible and preventing
downstream promotion.  A separate ``rollback_prompt`` branch is triggered by
catching the failure with an Airflow trigger rule, or can be wired as a
separate monitoring pipeline.

Pipeline stages
---------------
1. **run_current_eval** — run a daily evaluation experiment against a shared
   eval dataset (ArizeAxRunExperimentOperator).
2. **wait_for_current** — block until the experiment has enough completed runs
   (ArizeAxExperimentRunCountSensor).
3. **detect_drift** — compare the current experiment against the configured
   baseline and raise AirflowException if drift is found
   (ArizeAxDetectEvalDriftOperator with ``fail_on_drift=True``).
4. **rollback_prompt** — re-promote the known-stable prompt version to the
   ``"production"`` label (ArizeAxPromotePromptOperator).  Runs only when
   detect_drift fails, via ``trigger_rule="all_failed"``.
5. **notify_rollback** — log rollback confirmation; placeholder for
   Slack/email integration.

Schedule: ``@daily``

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_eval_dataset_id`` — dataset ID to run evaluations against.
- ``arize_ax_baseline_experiment_id`` — experiment ID of the known-good
  baseline to compare against.
- ``arize_ax_prompt_name`` — name of the prompt to roll back.
- ``arize_ax_stable_prompt_version_id`` — version ID of the stable prompt
  version to promote back to ``"production"``.

Constants (edit in this file or override via Variables):
- ``DRIFT_MIN_RUNS`` — minimum runs before scoring (default 5).
- ``DRIFT_THRESHOLD`` — absolute score drop that triggers rollback (default 0.1).

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Variables ``arize_ax_space_id``, ``arize_ax_eval_dataset_id``,
    ``arize_ax_baseline_experiment_id``, ``arize_ax_prompt_name``,
    ``arize_ax_stable_prompt_version_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable

try:
    from airflow.task.trigger_rule import TriggerRule
except ImportError:
    from airflow.utils.trigger_rule import TriggerRule

try:
    from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxDetectEvalDriftOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxPromotePromptOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxExperimentRunCountSensor,
)

# ---------------------------------------------------------------------------
# Constants — adjust to your environment
# ---------------------------------------------------------------------------
DRIFT_MIN_RUNS = 5
DRIFT_THRESHOLD = 0.1  # flag metrics that dropped more than 10 points



def _check_pipeline_configured(**ctx) -> bool:
    """Return True only when all required Variables are set for drift detection."""
    required = {
        "arize_ax_eval_dataset_id": "an evaluation dataset ID",
        "arize_ax_baseline_experiment_id": "a baseline experiment ID",
        "arize_ax_prompt_name": "the prompt name to roll back",
        "arize_ax_stable_prompt_version_id": "the stable prompt version ID",
    }
    missing = []
    for key, description in required.items():
        val = Variable.get(key, default_var=None)
        if not val or val.strip() in ("", f"your-{key.split('_', 2)[-1].replace('_', '-')}"):
            missing.append(f"  - {key}: set to {description}")
    if missing:
        print(
            "The following Variables must be set to run the drift detection pipeline:\n"
            + "\n".join(missing)
        )
        return False
    return True


def _eval_task(dataset_row: dict) -> dict:
    """Evaluation task: echo the expected output as a stub.

    Replace with a real LLM call in production, e.g.::

        response = openai.chat.completions.create(...)
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


def _notify_rollback(**ctx) -> dict[str, Any]:
    """Log rollback confirmation.

    In production, add a Slack/email notification here, e.g.::

        import slack_sdk
        client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        client.chat_postMessage(
            channel="#ops-alerts",
            text=f"Prompt rolled back: {prompt_name} -> stable version",
        )
    """
    rollback_result = ctx["ti"].xcom_pull(task_ids="rollback_prompt") or {}
    prompt_id = rollback_result.get("prompt_id")
    label = rollback_result.get("label")

    print("=" * 60)
    print("PROMPT ROLLBACK COMPLETE")
    print(f"  Prompt ID : {prompt_id}")
    print(f"  Label     : {label}")
    print("  Action    : notification sent (replace stub with Slack/email/webhook).")
    print("=" * 60)
    return {"notified": True, "prompt_id": prompt_id, "label": label}


with DAG(
    dag_id="arize_ax_drift_detection",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "drift", "experiments", "prompts", "rollback"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Stage 0 — Guard: skip DAG if required Variables are not configured
    check_pipeline_configured = ShortCircuitOperator(
        task_id="check_pipeline_configured",
        python_callable=_check_pipeline_configured,
    )

    # Stage 1 — Run today's evaluation experiment
    run_current_eval = ArizeAxRunExperimentOperator(
        task_id="run_current_eval",
        name="drift-check-current-{{ ds }}",
        dataset_id="{{ var.value.get('arize_ax_eval_dataset_id', '') }}",
        task=_eval_task,
        evaluators=[_accuracy_evaluator],
        concurrency=4,
    )

    # Stage 2 — Wait for runs to complete
    wait_for_current = ArizeAxExperimentRunCountSensor(
        task_id="wait_for_current",
        experiment_id="{{ ti.xcom_pull(task_ids='run_current_eval') }}",
        min_runs=DRIFT_MIN_RUNS,
        poke_interval=15,
        timeout=600,
        mode="poke",
        soft_fail=True,
    )

    # Stage 3 — Detect drift vs baseline (raises on drift)
    detect_drift = ArizeAxDetectEvalDriftOperator(
        task_id="detect_drift",
        current_experiment_id="{{ ti.xcom_pull(task_ids='run_current_eval') }}",
        baseline_experiment_id="{{ var.value.get('arize_ax_baseline_experiment_id', '') }}",
        drift_threshold=DRIFT_THRESHOLD,
        aggregation="mean",
        fail_on_drift=True,
    )

    # Stage 4 — Roll back prompt when drift was detected (task failed)
    rollback_prompt = ArizeAxPromotePromptOperator(
        task_id="rollback_prompt",
        prompt_name="{{ var.value.get('arize_ax_prompt_name', '') }}",
        label="production",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        version_id="{{ var.value.get('arize_ax_stable_prompt_version_id', None) }}",
        trigger_rule=TriggerRule.ALL_FAILED,
    )

    # Stage 5 — Notify (Slack/email placeholder)
    notify_rollback = PythonOperator(
        task_id="notify_rollback",
        python_callable=_notify_rollback,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Wiring
    check_pipeline_configured >> run_current_eval >> wait_for_current >> detect_drift
    detect_drift >> rollback_prompt >> notify_rollback
