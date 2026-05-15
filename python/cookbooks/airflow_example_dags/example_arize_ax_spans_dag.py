"""
Example DAG: Arize AX spans workflow.

Demonstrates listing spans with filters, exporting spans to Parquet with
targeted ``where`` / ``columns``, and updating evaluations on exported spans.

This DAG requires a real Arize project with existing spans.  Set the
``arize_ax_project_id`` Airflow Variable before triggering.  If the Variable
is absent or empty the downstream operators will raise AirflowException with
a clear message — no ShortCircuit task is required.

Requires:
- Airflow connection ``arize_ax_default`` with API key.
- Airflow variable ``arize_ax_project_id`` — set to a real project ID.
- Optional variable ``arize_ax_space_id`` (falls back to connection default_space).
- Optional variable ``arize_ax_project_name`` (defaults to resolved from project).
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxListSpansOperator,
    ArizeAxSpansExportToDataframeOperator,
    ArizeAxSpansExportToParquetOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import ArizeAxSpanCountSensor

# Resolve the project name from an Airflow Variable so the DAG is not
# tied to a hard-coded string.  Falls back to None so the hook uses its
# default resolution when the variable is not defined.
try:
    _project_name: str | None = Variable.get(
        "arize_ax_project_name", default_var=None
    )
except Exception:
    _project_name = None


with DAG(
    dag_id="example_arize_ax_spans",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "spans"],
    catchup=False,
    doc_md=__doc__,
) as dag:
    wait_for_spans = ArizeAxSpanCountSensor(
        task_id="wait_for_spans",
        project_id="{{ var.value.get('arize_ax_project_id', '') }}",
        min_count=10,
        poke_interval=60,
        timeout=600,
        mode="poke",
    )

    list_error_spans = ArizeAxListSpansOperator(
        task_id="list_error_spans",
        project_id="{{ var.value.get('arize_ax_project_id', '') }}",
        filter="status_code = 'ERROR'",
        limit=50,
    )

    export_spans_df = ArizeAxSpansExportToDataframeOperator(
        task_id="export_spans_filtered",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        project_name=_project_name,
        start_time="{{ (logical_date - macros.timedelta(days=7)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        where="span.status_code = 'ERROR'",
        columns=["context.span_id", "name", "status_code", "latency_ms"],
    )

    export_spans_parquet = ArizeAxSpansExportToParquetOperator(
        task_id="export_spans_parquet",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        project_name=_project_name,
        start_time="{{ (logical_date - macros.timedelta(days=7)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        path="/tmp/spans_export.parquet",
        where="span.status_code = 'OK'",
    )

    wait_for_spans >> [list_error_spans, export_spans_df, export_spans_parquet]
