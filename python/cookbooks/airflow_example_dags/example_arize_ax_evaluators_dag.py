"""
Example DAG: Arize AX Eval Hub (LLM template evaluators).

This DAG exercises **all evaluator operators**:

- :class:`~airflow.providers.arize_ax.operators.evaluators.ArizeAxListEvaluatorsOperator`
  twice: a baseline list, then a filtered list after mutations.
- :class:`~airflow.providers.arize_ax.operators.evaluators.ArizeAxCreateEvaluatorOperator`
- :class:`~airflow.providers.arize_ax.operators.evaluators.ArizeAxUpdateEvaluatorOperator`
- :class:`~airflow.providers.arize_ax.operators.evaluators.ArizeAxAddEvaluatorVersionOperator`

**Flow**

1. ``list_evaluators_baseline`` — always runs; lists evaluators in the space (read-only).
2. ``check_ai_integration`` — ``ShortCircuitOperator``; if the integration ID is unset
   (neither worker env nor Airflow Variable ``ARIZE_AI_INTEGRATION_ID``), downstream
   tasks are **skipped** (no API mutations).
3. When integration is configured: build template configs (Python) → create evaluator →
   update description → add a second template version → ``list_evaluators_verify`` with
   ``name_search`` matching the created evaluator name prefix.

Requires:
- Airflow connection ``arize_ax_default`` with API key (and optional host/extra).
- Airflow variable ``arize_ax_space_id`` (space global ID) for templated tasks.
- For the mutation chain: **AI integration global ID** — set either
  ``ARIZE_AI_INTEGRATION_ID`` in the **worker environment** (Docker/Kubernetes) **or**
  an Airflow **Variable** with key ``ARIZE_AI_INTEGRATION_ID`` (Admin → Variables).
  Optional model: env or Variable ``ARIZE_EVALUATOR_MODEL`` (default ``gpt-4o-mini``).

The Evaluators REST API is **alpha**. See:
https://arize.com/docs/api-reference/evaluators/list-evaluators
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable

try:
    from airflow.providers.standard.operators.python import (
        PythonOperator,
        ShortCircuitOperator,
    )
except ImportError:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxAddEvaluatorVersionOperator,
    ArizeAxCreateEvaluatorOperator,
    ArizeAxListEvaluatorsOperator,
    ArizeAxUpdateEvaluatorOperator,
)


def _get_ai_integration_id() -> str:
    """Worker env ``ARIZE_AI_INTEGRATION_ID`` wins; else Airflow Variable of the same key."""
    v = os.environ.get("ARIZE_AI_INTEGRATION_ID", "").strip()
    if v:
        return v
    return Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()


def _get_evaluator_model() -> str:
    """Optional judge model: env or Variable ``ARIZE_EVALUATOR_MODEL``."""
    v = os.environ.get("ARIZE_EVALUATOR_MODEL", "").strip()
    if v:
        return v
    return (
        Variable.get("ARIZE_EVALUATOR_MODEL", default_var="gpt-4o-mini").strip()
        or "gpt-4o-mini"
    )


def _has_ai_integration(**_context: Any) -> bool:
    """Return True when mutation tasks should run (integration ID present)."""
    return bool(_get_ai_integration_id())


def _build_evaluator_template_config_v1(**_context: Any) -> dict[str, Any]:
    """Build initial ``template_config`` for ``ArizeAxCreateEvaluatorOperator``."""
    integration_id = _get_ai_integration_id()
    if not integration_id:
        raise RuntimeError(
            "ARIZE_AI_INTEGRATION_ID must be set in the worker environment or as an "
            "Airflow Variable (Admin → Variables)."
        )
    model = _get_evaluator_model()
    return {
        "template_config": {
            "name": "airflow_example",
            "template": (
                "You are an LLM judge. Given input and output, respond with a JSON object "
                "with keys label (correct or incorrect) and reason. "
                "Input: {input} Output: {output}"
            ),
            "include_explanations": True,
            "use_function_calling_if_available": False,
            "classification_choices": {"incorrect": 0, "correct": 1},
            "llm_config": {
                "ai_integration_id": integration_id,
                "model_name": model,
                "invocation_parameters": {"temperature": 0},
                "provider_parameters": {},
            },
        },
    }


def _build_evaluator_template_config_v2(**context: Any) -> dict[str, Any]:
    """Second template version (slightly different prompt) for add-version task."""
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="build_evaluator_template_v1") or {}
    cfg = dict(payload.get("template_config") or {})
    cfg["template"] = (cfg.get("template") or "") + " Prefer concise reasons."
    return {"template_config": cfg}


# Name prefix for created evaluators; list_evaluators_verify uses the same prefix via name_search.
_EVAL_NAME_PREFIX = "airflow-example-eval"


with DAG(
    dag_id="example_arize_ax_evaluators",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "evaluators"],
    catchup=False,
    doc_md=__doc__,
) as dag:
    list_evaluators_baseline = ArizeAxListEvaluatorsOperator(
        task_id="list_evaluators_baseline",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        limit=25,
    )

    check_ai_integration = ShortCircuitOperator(
        task_id="check_ai_integration",
        python_callable=_has_ai_integration,
    )

    build_evaluator_template_v1 = PythonOperator(
        task_id="build_evaluator_template_v1",
        python_callable=_build_evaluator_template_config_v1,
    )

    build_evaluator_template_v2 = PythonOperator(
        task_id="build_evaluator_template_v2",
        python_callable=_build_evaluator_template_config_v2,
    )

    create_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_evaluator",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name=f"{_EVAL_NAME_PREFIX}-{{{{ ts_nodash }}}}",
        commit_message="Initial version from example_arize_ax_evaluators DAG",
        description="Example LLM-as-judge evaluator created by Airflow",
        template_config_task_id="build_evaluator_template_v1",
        template_config_key="template_config",
    )

    update_evaluator_metadata = ArizeAxUpdateEvaluatorOperator(
        task_id="update_evaluator_metadata",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        description="Metadata updated by example_arize_ax_evaluators DAG",
    )

    add_evaluator_version = ArizeAxAddEvaluatorVersionOperator(
        task_id="add_evaluator_version",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        commit_message="Second template from example DAG",
        template_config_task_id="build_evaluator_template_v2",
        template_config_key="template_config",
    )

    # Second list call: exercises ListEvaluatorsOperator with name filter (API `name` param).
    # Matches evaluators whose name contains this prefix (same prefix as create_evaluator).
    list_evaluators_verify = ArizeAxListEvaluatorsOperator(
        task_id="list_evaluators_verify",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        limit=50,
        name_search=_EVAL_NAME_PREFIX,
    )

    (
        list_evaluators_baseline
        >> check_ai_integration
        >> build_evaluator_template_v1
        >> create_evaluator
        >> update_evaluator_metadata
        >> build_evaluator_template_v2
        >> add_evaluator_version
        >> list_evaluators_verify
    )
