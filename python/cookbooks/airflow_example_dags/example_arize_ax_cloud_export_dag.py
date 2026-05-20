"""Cloud-Export + Round-Trip demo: spans → cloud → dataset, on S3 / GCS / ABFS.

This DAG demonstrates both halves of the cloud-storage support:

* **Write-side (Phase 1)** — write-side operators accept ``path`` /
  ``output_path`` as either a local filesystem path or a remote URI
  (``s3://``, ``gs://``, ``gcs://``, ``abfs://``, ``az://``, ``azure://``).
  Remote URIs are streamed through ``fsspec`` after the SDK writes
  locally — multipart upload + parallel transfers are handled transparently
  by the protocol-specific backend (``s3fs``, ``gcsfs``, ``adlfs``).
* **Read-side (Phase 2)** — read-side operators accept the same URI shapes
  for ``examples_path`` / ``experiment_runs_path``. They download via
  ``fsspec`` to a temp file (when the SDK cannot consume a URI directly)
  or pass the URI through to pandas' native fsspec layer (for CSV /
  parquet). Both halves accept ``storage_options`` (inline dict) or
  ``cloud_conn_id`` (Airflow Connection) — the latter is idiomatic for
  production where credentials live in Airflow's encrypted Connection
  store.

Pipeline (6 stages)::

    check_prereqs           (ShortCircuit — Variables + project_id set)
        │
        ▼
    resolve_storage_options (PythonOperator — Variable → dict)
        │
        ▼
    export_spans_to_cloud   (ArizeAxSpansExportToParquetOperator, path="s3://...")
        │
        ▼
    verify_object_landed    (PythonOperator — reads parquet back via fsspec)
        │
        ▼
    stage_dataset_examples  (PythonOperator — converts spans parquet → examples JSON in cloud)
        │
        ▼
    load_dataset_from_cloud (ArizeAxCreateDatasetOperator, examples_path="s3://...")
        │
        ▼
    log_summary             (PythonOperator — prints row count + dataset ID)

Required Airflow Variables
--------------------------
- ``arize_ax_project_name`` — name of the Arize project to export spans from
  (e.g. ``"live-fin-langgraph"`` or ``"my-llm-app"``). Use a project NAME,
  not the base64 project ID — ``ArizeAxSpansExportToParquetOperator`` looks
  the export endpoint up by name.
- ``arize_ax_cloud_target_prefix`` — scheme + bucket / container as a
  plain string, no Jinja inside. Examples::

      s3://demo-bucket
      gs://demo-bucket
      abfs://demo-container

  The DAG composes the full URI inline as
  ``{prefix}/spans-{{ ds_nodash }}.parquet`` so the date partition
  actually renders. **Do not** include any Jinja syntax inside the
  Variable value — Airflow does not recursively render templates that
  live inside a Variable (it expands ``{{ var.value.* }}`` once and
  stops). Fixed strings are fine; templates stored inside a Variable
  land literally in the output. Keep configuration in the Variable;
  keep templating in the DAG.
- ``arize_ax_cloud_storage_options`` — JSON dict of credentials/endpoint
  overrides for the target backend. Examples:

  S3 (real AWS)::

      {"key": "<AWS_ACCESS_KEY_ID>", "secret": "<AWS_SECRET_ACCESS_KEY>"}

  MinIO (local, no cloud account)::

      {"key": "minioadmin", "secret": "minioadmin",
       "client_kwargs": {"endpoint_url": "http://minio:9000"}}

  GCS (service account JSON loaded by gcsfs)::

      {"token": "/path/to/service-account.json"}

  Azure (account key)::

      {"account_name": "myaccount", "account_key": "<KEY>"}

Local testing without any cloud account
---------------------------------------
Spin up MinIO standalone (S3-API-compatible) and target it from the DAG:

::

    docker run -d --name minio -p 9000:9000 -p 9001:9001 \\
      -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \\
      minio/minio server /data --console-address ":9001"

Then set Variables (full prefix with scheme, but no Jinja)::

    arize_ax_cloud_target_prefix = s3://airflow-demo
    arize_ax_cloud_storage_options = {
      "key": "minioadmin",
      "secret": "minioadmin",
      "client_kwargs": {"endpoint_url": "http://host.docker.internal:9000"}
    }

Browse uploaded objects at http://localhost:9001 (minioadmin / minioadmin).

The same DAG covers GCS (fake-gcs-server) and Azure (azurite) by swapping
the prefix + storage_options::

    # GCS via fake-gcs-server
    arize_ax_cloud_target_prefix = gs://airflow-demo
    arize_ax_cloud_storage_options = {
      "token": "anon",
      "endpoint_url": "http://host.docker.internal:4443"
    }

    # Azure via azurite (devstoreaccount1 is the default emulator account)
    arize_ax_cloud_target_prefix = abfs://airflow-demo
    arize_ax_cloud_storage_options = {
      "account_name": "devstoreaccount1",
      "account_key": "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
      "connection_string": "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://host.docker.internal:10000/devstoreaccount1;"
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable

try:
    from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.providers.arize_ax.operators.datasets import ArizeAxCreateDatasetOperator
from airflow.providers.arize_ax.operators.spans import ArizeAxSpansExportToParquetOperator

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"


def _check_prereqs(**_ctx) -> bool:
    project_name = Variable.get("arize_ax_project_name", default_var="").strip()
    prefix = Variable.get("arize_ax_cloud_target_prefix", default_var="").strip()
    if not project_name:
        print("Variable arize_ax_project_name is not set — skipping demo.")
        return False
    if not prefix:
        print("Variable arize_ax_cloud_target_prefix is not set — skipping demo.")
        return False
    if not prefix.startswith(("s3://", "gs://", "gcs://", "abfs://", "az://", "azure://")):
        print(
            f"Variable arize_ax_cloud_target_prefix must include a scheme "
            f"(e.g. 's3://demo-bucket' or 'gs://demo-bucket'); got {prefix!r}."
        )
        return False
    if "{{" in prefix or "}}" in prefix:
        print(
            f"Variable arize_ax_cloud_target_prefix contains Jinja syntax — "
            f"Airflow does not recursively render templates inside Variable "
            f"values. Store a plain prefix and let the DAG compose the key. "
            f"Got {prefix!r}."
        )
        return False
    return True


def _resolve_storage_options(**_ctx) -> dict[str, Any] | None:
    raw = Variable.get("arize_ax_cloud_storage_options", default_var=None)
    if not raw:
        return None
    return json.loads(raw)


def _verify_object_landed(**ctx) -> dict[str, Any]:
    """Read the parquet back through fsspec to confirm the upload succeeded."""
    target_uri = ctx["ti"].xcom_pull(task_ids="export_spans_to_cloud")
    storage_options = json.loads(
        Variable.get("arize_ax_cloud_storage_options", default_var="null")
    )
    import fsspec
    import pandas as pd

    with fsspec.open(target_uri, "rb", **(storage_options or {})) as fh:
        df = pd.read_parquet(fh)
    row_count = len(df)
    columns = list(df.columns)
    print(f"[verify] Read back {row_count} rows from {target_uri}")
    print(f"[verify] Columns: {columns[:10]}{'…' if len(columns) > 10 else ''}")
    return {"uri": target_uri, "row_count": row_count, "columns": columns}


def _stage_dataset_examples(**ctx) -> str:
    """Convert the spans parquet → a small examples JSON file, also in cloud storage.

    This demonstrates the canonical "stage objects in cloud, then ingest"
    pattern used by Snowflake / Databricks / BigQuery integrations: the
    spans land in the bucket as parquet, then a follow-up task projects
    the columns we want into a dataset-ready JSON object next to the
    original — also in the bucket. The downstream
    ``ArizeAxCreateDatasetOperator`` reads the object back via
    ``examples_path="<uri>"`` + ``storage_options`` without any local
    download in the DAG code.
    """
    spans_uri = ctx["ti"].xcom_pull(task_ids="export_spans_to_cloud")
    storage_options = ctx["ti"].xcom_pull(task_ids="resolve_storage_options") or None
    import fsspec
    import pandas as pd

    with fsspec.open(spans_uri, "rb", **(storage_options or {})) as fh:
        df = pd.read_parquet(fh)
    # Project a few common columns into dataset-example shape. Any columns
    # missing on a given project are silently dropped — the demo only needs
    # one or two rows to exercise the read-side cloud routing.
    candidate_cols = [
        "attributes.input.value", "input.value", "input",
        "attributes.output.value", "output.value", "output",
    ]
    keep = [c for c in candidate_cols if c in df.columns]
    sample = df[keep].head(5) if keep else df.head(5)
    examples = [{"input": str(v) for k, v in row.items() if "input" in k.lower()} or {"input": ""}
                for row in sample.to_dict(orient="records")]
    if not examples:
        # Fallback so the demo still produces a non-empty dataset payload.
        examples = [{"input": "demo-only example (no input column found in spans)"}]

    examples_uri = spans_uri.rsplit("/", 1)[0] + "/examples-" + ctx["ds_nodash"] + ".json"
    with fsspec.open(examples_uri, "wb", **(storage_options or {})) as fh:
        import json as _json
        fh.write(_json.dumps(examples).encode("utf-8"))
    print(f"[stage] Wrote {len(examples)} examples to {examples_uri}")
    return examples_uri


def _log_summary(**ctx) -> None:
    export = ctx["ti"].xcom_pull(task_ids="export_spans_to_cloud", key="result") or {}
    verify = ctx["ti"].xcom_pull(task_ids="verify_object_landed") or {}
    examples_uri = ctx["ti"].xcom_pull(task_ids="stage_dataset_examples")
    dataset_id = ctx["ti"].xcom_pull(task_ids="load_dataset_from_cloud")
    print("=" * 60)
    print("CLOUD ROUND-TRIP COMPLETE")
    print(f"  Spans parquet : {export.get('path')}")
    print(f"  Wrote         : {export.get('wrote')}")
    print(f"  Rows read back: {verify.get('row_count')}")
    print(f"  Examples URI  : {examples_uri}")
    print(f"  Dataset ID    : {dataset_id}")
    print("  Inspect the object in your cloud console (AWS, GCS, Azure, or MinIO).")
    print("=" * 60)


with DAG(
    dag_id="arize_ax_cloud_export_demo",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "demo", "spans", "cloud", "fsspec"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    resolve_storage_options = PythonOperator(
        task_id="resolve_storage_options",
        python_callable=_resolve_storage_options,
    )

    # The same operator handles local AND remote paths — only ``path`` and
    # ``storage_options`` change between modes. ``storage_options`` is a real
    # template_field so it can flow from upstream XCom like any other kwarg.
    # The full URI is composed from the bucket Variable (just the bucket
    # name) plus the date partition. Composing it here in the DAG (rather
    # than storing it in the Variable) ensures Jinja actually renders the
    # date — Airflow does not recursively expand templates that live
    # inside a Variable's value.
    export_spans_to_cloud = ArizeAxSpansExportToParquetOperator(
        task_id="export_spans_to_cloud",
        space_id=_SPACE_JINJA,
        project_name="{{ var.value.arize_ax_project_name }}",
        start_time="{{ (logical_date - macros.timedelta(days=1)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        path="{{ var.value.arize_ax_cloud_target_prefix }}/spans-{{ ds_nodash }}.parquet",
        storage_options="{{ ti.xcom_pull(task_ids='resolve_storage_options') }}",
    )

    verify_object_landed = PythonOperator(
        task_id="verify_object_landed",
        python_callable=_verify_object_landed,
    )

    stage_dataset_examples = PythonOperator(
        task_id="stage_dataset_examples",
        python_callable=_stage_dataset_examples,
    )

    # Read-side cloud routing: the operator downloads the staged JSON via
    # fsspec (or, for CSV, passes the URI straight to pandas), parses it,
    # and creates a dataset. ``cloud_conn_id`` would be the production
    # equivalent of inline ``storage_options`` — point at any Airflow
    # AWS / GCP / WASB Connection and the resolver maps it onto fsspec
    # kwargs automatically.
    load_dataset_from_cloud = ArizeAxCreateDatasetOperator(
        task_id="load_dataset_from_cloud",
        space_id=_SPACE_JINJA,
        name="cloud-round-trip-{{ ds_nodash }}",
        examples_path="{{ ti.xcom_pull(task_ids='stage_dataset_examples') }}",
        storage_options="{{ ti.xcom_pull(task_ids='resolve_storage_options') }}",
        if_exists="skip",
    )

    log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=_log_summary,
    )

    check_prereqs >> resolve_storage_options >> export_spans_to_cloud
    export_spans_to_cloud >> verify_object_landed >> stage_dataset_examples
    stage_dataset_examples >> load_dataset_from_cloud >> log_summary
