"""
Automated Dataset Curation DAG: filter, deduplicate, and append
high-quality production spans into an Arize evaluation dataset on a
daily schedule.

Pipeline stages
---------------
1. **curate_high_quality_spans** — export spans with correctness > 0.85,
   deduplicate on the ``input`` column, and append up to 500 examples to the
   target dataset (ArizeAxCurateSpansToDatasetOperator).
2. **check_curated** — short-circuit if no new examples were appended.
3. **get_dataset_stats** — list current dataset examples to get the total
   count (ArizeAxListDatasetExamplesOperator).
4. **log_curation_summary** — log total examples, appended today, dedup rate.

Schedule: ``@daily`` — curate new high-quality spans every day.

Variables
---------
- ``arize_ax_project_id`` — Arize project ID (base64).
- ``arize_ax_dataset_id`` — ID of the target Arize dataset.

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Project ``arize_ax_project_id`` must have spans with eval scores.
  - Dataset ``arize_ax_dataset_id`` must already exist; create it first with
    ``ArizeAxCreateDatasetOperator`` if needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxListDatasetExamplesOperator,
)
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxCurateSpansToDatasetOperator,
)


def _check_curated(**ctx) -> bool:
    """Return True iff at least one example was appended."""
    result = ctx["ti"].xcom_pull(task_ids="curate_high_quality_spans")
    if not isinstance(result, dict):
        print(f"[check_curated] unexpected XCom type: {type(result).__name__!r}")
        return False
    count = result.get("appended_count", 0)
    print(f"[check_curated] appended_count={count}")
    return count > 0


def _log_curation_summary(**ctx) -> dict[str, Any]:
    """Log dataset curation summary: total, appended, dedup rate."""
    curate_result = ctx["ti"].xcom_pull(task_ids="curate_high_quality_spans") or {}
    stats_result = ctx["ti"].xcom_pull(task_ids="get_dataset_stats") or {}

    appended = curate_result.get("appended_count", 0)
    deduped = curate_result.get("deduplicated_count", 0)
    dataset_id = curate_result.get("dataset_id", "N/A")
    total_in_dataset = stats_result.get("count", "N/A")

    total_before_dedup = appended + deduped
    dedup_rate = (deduped / total_before_dedup * 100) if total_before_dedup > 0 else 0.0

    data_interval_start = ctx.get("data_interval_start", "N/A")
    data_interval_end = ctx.get("data_interval_end", "N/A")

    print("=" * 60)
    print("DATASET CURATION SUMMARY")
    print(f"  Date range    : {data_interval_start} → {data_interval_end}")
    print(f"  Dataset ID    : {dataset_id}")
    print(f"  Total in ds   : {total_in_dataset}")
    print(f"  Appended today: {appended}")
    print(f"  Dedup removed : {deduped} ({dedup_rate:.1f}%)")
    print("=" * 60)
    return {
        "appended": appended,
        "deduplicated": deduped,
        "dedup_rate_pct": round(dedup_rate, 2),
        "total_in_dataset": total_in_dataset,
    }


with DAG(
    dag_id="arize_ax_dataset_curation",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "spans", "dataset", "curation"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Stage 1 — Curate spans into dataset
    curate_high_quality_spans = ArizeAxCurateSpansToDatasetOperator(
        task_id="curate_high_quality_spans",
        project_id="{{ var.value.get('arize_ax_project_id', '') }}",
        dataset_id="{{ var.value.get('arize_ax_dataset_id', '') }}",
        where="evals['correctness'].score > 0.85",
        # Manual triggers leave data_interval_start == data_interval_end (both
        # equal logical_date), and the SDK's exporter validates start < end
        # strictly. Compute a 24h window off logical_date so manual triggers
        # work; scheduled runs still see (close to) one schedule interval.
        start_time="{{ (logical_date - macros.timedelta(hours=24)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        deduplicate_on="input",
        max_examples=500,
    )

    # Stage 2 — Gate: skip if nothing was appended
    check_curated = ShortCircuitOperator(
        task_id="check_curated",
        python_callable=_check_curated,
    )

    # Stage 3 — Get current dataset stats
    get_dataset_stats = ArizeAxListDatasetExamplesOperator(
        task_id="get_dataset_stats",
        dataset_id="{{ var.value.get('arize_ax_dataset_id', '') }}",
        limit=1,
    )

    # Stage 4 — Log summary
    log_curation_summary = PythonOperator(
        task_id="log_curation_summary",
        python_callable=_log_curation_summary,
    )

    # Wiring
    curate_high_quality_spans >> check_curated >> get_dataset_stats >> log_curation_summary
