"""
Example DAG: Arize AX Annotation Queues demo (alpha).

This DAG demonstrates a fully self-contained Human-in-the-Loop (HITL) workflow.
It creates its own annotation config, builds a queue against that config, inspects
it, then cleans up — no pre-existing Arize resources are required.

**Flow**

1. ``create_annotation_config`` — create a freeform annotation config used by the queue.
2. ``create_queue`` — create an annotation queue, pulling the config ID from XCom.
3. ``get_queue`` — fetch the queue details to confirm creation.
4. ``list_queues`` — list all annotation queues in the space.
5. ``list_records`` — list records pending human review (empty after fresh create).
6. ``cleanup_queue`` — delete the demo queue (runs regardless of upstream success).
7. ``cleanup_config`` — delete the demo annotation config (runs regardless of upstream success).

**Required**

- Airflow connection ``arize_ax_default`` with API key (password or extra ``api_key``).
  The ``default_space`` extra (or ``arize_ax_space_id`` Airflow Variable) must resolve
  to a valid space; the hook will raise a clear error if neither is set.
- Arize Python SDK >= 8.11.0 (``client.annotation_queues`` and
  ``client.annotation_configs`` sub-clients).

**Optional Airflow Variables**

- ``arize_ax_space_id``     — space global ID or name; SDK infers from API key when absent.
- ``arize_queue_name``      — name for the demo queue (default: ``"airflow-demo-queue"``).

**Required Airflow Variable**

- ``arize_annotator_email`` — comma-separated reviewer email(s) for the queue.
  The task fails fast with a clear error if this Variable is unset.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook
from airflow.providers.arize_ax.operators.annotations import (
    ArizeAxCreateAnnotationConfigOperator,
    ArizeAxDeleteAnnotationConfigOperator,
    ArizeAxDeleteAnnotationQueueOperator,
    ArizeAxGetAnnotationQueueOperator,
    ArizeAxListAnnotationQueueRecordsOperator,
    ArizeAxListAnnotationQueuesOperator,
)
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor

log = logging.getLogger(__name__)

_CONFIG_NAME = "airflow-demo-annotation-config"


def _resolve_annotator_emails() -> list[str]:
    """Read reviewer emails from the ``arize_annotator_email`` Variable.

    Fails fast (AirflowException) if the Variable is unset so the DAG doesn't
    silently fall back to a hardcoded address.
    """
    raw = Variable.get("arize_annotator_email", default_var=None)
    if not raw:
        raise AirflowException(
            "Set the 'arize_annotator_email' Airflow Variable to one or more "
            "reviewer email addresses (comma-separated) before running this DAG."
        )
    emails = [e.strip() for e in str(raw).split(",") if e.strip()]
    if not emails:
        raise AirflowException(
            "'arize_annotator_email' Variable is set but resolved to no addresses."
        )
    return emails


def _create_queue(**context):
    config_id = context["ti"].xcom_pull(task_ids="create_annotation_config")
    space = Variable.get("arize_ax_space_id", default_var=None)
    queue_name = Variable.get("arize_queue_name", default_var="airflow-demo-queue")
    log.info("Using annotation config ID: %s", config_id)
    hook = ArizeAxHook()
    return hook.create_annotation_queue(
        name=queue_name,
        space=space,
        annotation_config_ids=[config_id],
        annotator_emails=_resolve_annotator_emails(),
        instructions="Review LLM responses for accuracy and tone.",
        assignment_method="all",
    )


with DAG(
    dag_id="arize_ax_annotation_queues_demo",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "annotations", "hitl"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    _space_jinja = "{{ var.value.get('arize_ax_space_id', '') or None }}"
    _queue_name_jinja = "{{ var.value.get('arize_queue_name', 'airflow-demo-queue') }}"

    create_annotation_config = ArizeAxCreateAnnotationConfigOperator(
        task_id="create_annotation_config",
        name=_CONFIG_NAME,
        space=_space_jinja,
        config_type="freeform",
    )

    create_queue = PythonOperator(
        task_id="create_queue",
        python_callable=_create_queue,
    )

    get_queue = ArizeAxGetAnnotationQueueOperator(
        task_id="get_queue",
        annotation_queue=_queue_name_jinja,
        space=_space_jinja,
    )

    list_queues = ArizeAxListAnnotationQueuesOperator(
        task_id="list_queues",
        space=_space_jinja,
        limit=25,
    )

    list_records = ArizeAxListAnnotationQueueRecordsOperator(
        task_id="list_records",
        annotation_queue=_queue_name_jinja,
        space=_space_jinja,
        limit=50,
    )

    # Pause for manual inspection in the Arize AX UI. ``mode="reschedule"``
    # releases the worker slot during the wait — unlike time.sleep which
    # would hold a slot for the full 5 minutes.
    inspect_pause = TimeDeltaSensor(
        task_id="inspect_pause",
        delta=timedelta(minutes=5),
        mode="reschedule",
        trigger_rule="all_done",
    )

    cleanup_queue = ArizeAxDeleteAnnotationQueueOperator(
        task_id="cleanup_queue",
        annotation_queue=_queue_name_jinja,
        space=_space_jinja,
        trigger_rule="all_done",
    )

    cleanup_config = ArizeAxDeleteAnnotationConfigOperator(
        task_id="cleanup_config",
        annotation_config=_CONFIG_NAME,
        space=_space_jinja,
        trigger_rule="all_done",
    )

    (
        create_annotation_config
        >> create_queue
        >> get_queue
        >> list_queues
        >> list_records
        >> inspect_pause
        >> cleanup_queue
        >> cleanup_config
    )
