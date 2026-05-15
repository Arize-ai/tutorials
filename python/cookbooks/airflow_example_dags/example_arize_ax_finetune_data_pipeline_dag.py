"""
Fine-tuning Data Pipeline DAG: export high-quality production spans and build
a fine-tuning dataset for LLM improvement.

This DAG implements the observe→curate→fine-tune loop:
1. Observe: production LLM calls are traced and stored in Arize.
2. Curate: spans with high evaluation scores are exported as fine-tuning data.
3. Fine-tune: the curated file is staged in Arize as a dataset, ready to
   trigger an OpenAI fine-tune job (or equivalent).

Pipeline stages
---------------
1. **export_high_quality_spans** — export spans with correctness > 0.8 as
   OpenAI JSONL format (ArizeAxExportSpansToFineTuningOperator). Raises
   AirflowException if ``arize_ax_project_id`` Variable is absent or empty.
   OpenAI JSONL format (ArizeAxExportSpansToFineTuningOperator).
2. **check_has_examples** — short-circuit if no examples were exported.
3. **validate_finetune_file** — count lines, validate JSONL structure, and
   log a summary.
4. **create_finetune_dataset** — create (or skip) an Arize dataset to hold
   the fine-tuning examples.
5. **prepare_arize_examples** — parse the JSONL file and convert to Arize
   dataset example format.
6. **append_to_arize_dataset** — append the examples to the Arize dataset.
7. **notify_ready** — log the output path and count; placeholder for
   triggering an OpenAI fine-tune job.

Schedule: ``@weekly`` — re-run every week to accumulate new high-quality spans.

Variables
---------
- ``arize_ax_project_id`` — Arize project ID (base64). **Required.**
  Set this Variable to a real project ID that has spans with eval scores.
  ArizeAxExportSpansToFineTuningOperator raises AirflowException when unset.
- ``arize_ax_space_id`` — Arize space ID. Optional; falls back to connection
  ``default_space`` when absent.

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Project ``arize_ax_project_id`` must exist and have spans with eval scores.
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
    ArizeAxAppendDatasetExamplesOperator,
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxExportSpansToFineTuningOperator,
)

SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer concisely and factually."
)


def _check_has_examples(**ctx) -> bool:
    """Return True iff the export produced at least one example."""
    result = ctx["ti"].xcom_pull(task_ids="export_high_quality_spans")
    if not isinstance(result, dict):
        print(f"[check_has_examples] unexpected XCom type: {type(result).__name__!r}")
        return False
    count = result.get("count", 0)
    print(f"[check_has_examples] exported {count} examples.")
    return count > 0


def _validate_finetune_file(**ctx) -> dict[str, Any]:
    """Validate JSONL structure and log a summary."""
    import json

    result = ctx["ti"].xcom_pull(task_ids="export_high_quality_spans")
    path = result.get("path", "") if isinstance(result, dict) else ""
    if not path:
        raise ValueError("No output path in export_high_quality_spans XCom.")

    valid_count = 0
    invalid_count = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj or "input" in obj:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        print(f"  [validate] line {lineno}: missing 'messages' key")
                except json.JSONDecodeError as exc:
                    invalid_count += 1
                    print(f"  [validate] line {lineno}: invalid JSON: {exc}")
    except FileNotFoundError:
        raise ValueError(f"Fine-tune file not found: {path!r}")

    total = valid_count + invalid_count
    print(f"[validate] total={total}  valid={valid_count}  invalid={invalid_count}")
    if invalid_count > 0:
        print(f"[validate] WARNING: {invalid_count} invalid lines detected.")
    return {"path": path, "valid": valid_count, "invalid": invalid_count}


def _prepare_arize_examples(**ctx) -> list[dict[str, Any]]:
    """Load the JSONL file and convert to Arize dataset examples format."""
    import json

    result = ctx["ti"].xcom_pull(task_ids="export_high_quality_spans")
    path = result.get("path", "") if isinstance(result, dict) else ""
    if not path:
        return []

    examples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Handle openai_jsonl format ({"messages": [...]})
            if "messages" in obj:
                messages = obj["messages"]
                user_msg = next(
                    (m.get("content", "") for m in messages if m.get("role") == "user"),
                    "",
                )
                assistant_msg = next(
                    (m.get("content", "") for m in messages if m.get("role") == "assistant"),
                    "",
                )
                examples.append({"input": user_msg, "output": assistant_msg})
            else:
                # raw jsonl — pass through
                examples.append(obj)
    print(f"[prepare_arize_examples] prepared {len(examples)} examples.")
    return examples


def _notify_ready(**ctx) -> dict[str, Any]:
    """Log the ready file and provide a placeholder for triggering fine-tune."""
    result = ctx["ti"].xcom_pull(task_ids="export_high_quality_spans") or {}
    path = result.get("path", "N/A")
    count = result.get("count", 0)
    ds_id = ctx["ti"].xcom_pull(task_ids="create_finetune_dataset")
    print("=" * 60)
    print("FINE-TUNE DATASET READY")
    print(f"  Output JSONL  : {path}")
    print(f"  Examples      : {count}")
    print(f"  Arize dataset : {ds_id}")
    print("  Next step     : trigger OpenAI fine-tune job (not implemented).")
    print("  Example CLI   : openai api fine_tuning.jobs.create -t <file_id> -m gpt-4o-mini")
    print("=" * 60)
    return {"path": path, "count": count, "dataset_id": ds_id}


with DAG(
    dag_id="arize_ax_finetune_data_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    tags=["arize_ax", "spans", "finetune", "dataset"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    # Stage 1 — Export high-quality spans
    export_high_quality_spans = ArizeAxExportSpansToFineTuningOperator(
        task_id="export_high_quality_spans",
        project_id="{{ var.value.get('arize_ax_project_id', '') }}",
        output_path="/tmp/finetune_{{ ds }}.jsonl",
        output_format="openai_jsonl",
        where="evals['correctness'].score > 0.8",
        start_time="{{ (logical_date - macros.timedelta(days=7)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        system_prompt=SYSTEM_PROMPT,
    )

    # Stage 2 — Gate: skip if no examples
    check_has_examples = ShortCircuitOperator(
        task_id="check_has_examples",
        python_callable=_check_has_examples,
    )

    # Stage 3 — Validate
    validate_finetune_file = PythonOperator(
        task_id="validate_finetune_file",
        python_callable=_validate_finetune_file,
    )

    # Stage 4 — Create Arize dataset
    create_finetune_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_finetune_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', '') or None }}",
        name="finetune-{{ ds }}",
        # SDK 8.25+ rejects empty examples list at create time. Seed with a
        # single placeholder; real examples are loaded dynamically later.
        examples=[{"input": "_seed_", "expected": "_seed_"}],
        if_exists="skip",
    )

    # Stage 5 — Prepare and append examples
    prepare_arize_examples = PythonOperator(
        task_id="prepare_arize_examples",
        python_callable=_prepare_arize_examples,
    )

    append_to_arize_dataset = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_to_arize_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_finetune_dataset') }}",
        examples="{{ ti.xcom_pull(task_ids='prepare_arize_examples') }}",
    )

    # Stage 6 — Notify
    notify_ready = PythonOperator(
        task_id="notify_ready",
        python_callable=_notify_ready,
    )

    # Wiring
    export_high_quality_spans >> check_has_examples >> validate_finetune_file
    validate_finetune_file >> create_finetune_dataset >> prepare_arize_examples
    prepare_arize_examples >> append_to_arize_dataset >> notify_ready
