"""
Example DAG: Arize AX admin / discovery operators.

Lists projects, spaces, annotation configs, and prompts.  Useful for
inventory audits, drift detection on annotation schemas, or syncing
prompt versions across environments.

Requires:
- Airflow connection ``arize_ax_default`` with API key.
- Airflow variable ``arize_ax_space_id``.
- For prompt tasks: ``pip install 'arize[PromptHub]'``.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.arize_ax.operators.annotations import (
    ArizeAxListAnnotationConfigsOperator,
)
from airflow.providers.arize_ax.operators.projects import (
    ArizeAxCreateProjectOperator,
    ArizeAxListProjectsOperator,
)
from airflow.providers.arize_ax.operators.spaces import (
    ArizeAxListSpacesOperator,
)

with DAG(
    dag_id="example_arize_ax_admin",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "admin"],
    catchup=False,
) as dag:
    list_spaces = ArizeAxListSpacesOperator(
        task_id="list_spaces",
        limit=20,
    )

    list_projects = ArizeAxListProjectsOperator(
        task_id="list_projects",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        limit=20,
    )

    list_annotation_configs = ArizeAxListAnnotationConfigsOperator(
        task_id="list_annotation_configs",
        limit=50,
    )

    create_project = ArizeAxCreateProjectOperator(
        task_id="create_project",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name="airflow-managed-project-{{ ds }}",
        if_exists="skip",
    )

    list_spaces >> [list_projects, list_annotation_configs]
    list_projects >> create_project
