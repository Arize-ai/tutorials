"""
Example DAG: Arize AX ML batch logging and export workflow.

Logs a small batch of synthetic ML predictions with ArizeAxMLLogBatchOperator,
then exports the data for analysis with targeted ``where`` and ``columns``
filters.

The synthetic batch (4 rows) is built inline so the DAG runs end-to-end
without user-supplied data. In production, replace ``_SAMPLE_DATAFRAME`` and
``_SAMPLE_SCHEMA`` with your real predictions + schema.

Requires:
- Airflow connection ``arize_ax_default`` with API key.
- Airflow variable ``arize_ax_space_id``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.providers.arize_ax.operators.ml import (
    ArizeAxMLExportToDataframeOperator,
    ArizeAxMLExportToParquetOperator,
    ArizeAxMLLogBatchOperator,
)
from arize.ml.types import Environments, ModelTypes, Schema

# Tiny synthetic predictions batch — replace with your real DataFrame for production use.
_NOW_TS = int(time.time())
_SAMPLE_DATAFRAME = pd.DataFrame(
    [
        {"prediction_id": "ex-1", "prediction_label": "positive",
         "prediction_score": 0.92, "actual_label": "positive",
         "ts": _NOW_TS, "feature_1": 0.5},
        {"prediction_id": "ex-2", "prediction_label": "negative",
         "prediction_score": 0.71, "actual_label": "negative",
         "ts": _NOW_TS, "feature_1": 0.3},
        {"prediction_id": "ex-3", "prediction_label": "positive",
         "prediction_score": 0.88, "actual_label": "negative",
         "ts": _NOW_TS, "feature_1": 0.6},
        {"prediction_id": "ex-4", "prediction_label": "negative",
         "prediction_score": 0.65, "actual_label": "negative",
         "ts": _NOW_TS, "feature_1": 0.2},
    ]
)
_SAMPLE_SCHEMA = Schema(
    prediction_id_column_name="prediction_id",
    prediction_label_column_name="prediction_label",
    prediction_score_column_name="prediction_score",
    actual_label_column_name="actual_label",
    timestamp_column_name="ts",
    feature_column_names=["feature_1"],
)


with DAG(
    dag_id="example_arize_ax_ml",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "ml"],
    catchup=False,
) as dag:
    log_batch = ArizeAxMLLogBatchOperator(
        task_id="log_predictions",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        model_name="my-classifier",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        dataframe=_SAMPLE_DATAFRAME,
        schema=_SAMPLE_SCHEMA,
        # SDK 8.25+ rejects empty model_version with ValidationFailure; pass a
        # stable label so successive runs append to the same model version.
        extra_log_kwargs={"model_version": "v1"},
    )

    # Just-logged ML data takes a few minutes to become queryable via the
    # Flight export endpoint (model indexing delay). Retry with backoff so a
    # single-DAG-run demo can complete end-to-end once indexing catches up.
    export_df = ArizeAxMLExportToDataframeOperator(
        task_id="export_ml_data",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        model_name="my-classifier",
        environment=Environments.PRODUCTION,
        start_time="{{ (logical_date - macros.timedelta(days=1)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        # Arize exports score_categorical labels as ``scorePredictionLabel``
        # at query time (not ``prediction_label`` as used at log time).
        where="scorePredictionLabel = 'positive'",
        retries=5,
        retry_delay=timedelta(minutes=2),
    )

    export_parquet = ArizeAxMLExportToParquetOperator(
        task_id="export_ml_parquet",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        model_name="my-classifier",
        environment=Environments.PRODUCTION,
        start_time="{{ (logical_date - macros.timedelta(days=1)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
        path="/tmp/ml_export.parquet",
        retries=5,
        retry_delay=timedelta(minutes=2),
    )

    log_batch >> [export_df, export_parquet]
