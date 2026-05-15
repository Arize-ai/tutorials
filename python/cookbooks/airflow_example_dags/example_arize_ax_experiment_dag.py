"""
Example DAG: Arize AX experiment workflow.

Creates a dataset, runs an experiment with a task + evaluator, and retrieves
the experiment results.  Demonstrates the ArizeAxRunExperimentOperator.

Requires:
- Airflow connection ``arize_ax_default`` with API key.
- Airflow variable ``arize_ax_space_id``.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxListExperimentsOperator,
    ArizeAxRunExperimentOperator,
)


def _example_task(dataset_row: dict) -> dict:
    """Trivial task that echoes the expected output."""
    return {"output": dataset_row.get("expected_output", "")}


def _exact_match_evaluator(dataset_row: dict, output: dict):
    """Simple evaluator that checks if output matches expected."""
    from arize.experiments import EvaluationResult

    match = output.get("output") == dataset_row.get("expected_output")
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={dataset_row.get('expected_output')!r}, "
                    f"got={output.get('output')!r}",
    )


with DAG(
    dag_id="example_arize_ax_experiment",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "experiment"],
    catchup=False,
) as dag:
    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name="airflow-experiment-dataset",
        examples=[
            {"query": "What is 2+2?", "expected_output": "4"},
            {"query": "Capital of France?", "expected_output": "Paris"},
            {"query": "Largest planet?", "expected_output": "Jupiter"},
        ],
        if_exists="skip",
    )

    run_experiment = ArizeAxRunExperimentOperator(
        task_id="run_experiment",
        name="airflow-nightly-eval-{{ ds }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        task=_example_task,
        evaluators=[_exact_match_evaluator],
        concurrency=2,
    )

    list_experiments = ArizeAxListExperimentsOperator(
        task_id="list_experiments",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=5,
    )

    create_dataset >> run_experiment >> list_experiments
