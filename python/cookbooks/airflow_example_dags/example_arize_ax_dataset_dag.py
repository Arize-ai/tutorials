"""
Example DAG demonstrating Arize AX dataset operators.

Requires an Airflow connection of type `arize_ax` with:
- Password (API key)
- Optional: Host (API host override), Extra JSON: space_id, region

Set the ``arize_ax_space_id`` Airflow Variable (or use connection extra) and ensure the connection exists.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxGetDatasetOperator,
    ArizeAxListDatasetsOperator,
)

with DAG(
    dag_id="example_arize_ax_dataset",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example"],
    catchup=False,
) as dag:
    list_datasets = ArizeAxListDatasetsOperator(
        task_id="list_datasets",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        limit=10,
    )

    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name="airflow-example-dataset",
        examples=[
            {"query": "What is 2+2?", "expected_output": "4"},
            {"query": "Capital of France?", "expected_output": "Paris"},
        ],
        if_exists="skip",
    )

    get_dataset = ArizeAxGetDatasetOperator(
        task_id="get_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
    )

    list_datasets >> create_dataset >> get_dataset
