"""Self-learning agent demo — closed-loop prompt optimization from production feedback.

End-to-end story implemented with three native Arize provider operators and
**Airflow dynamic task mapping** so every system prompt in your project is
optimized in parallel — one task instance per agent.

Pipeline::

    curate_feedback_dataset  ──┬─►  optimize_prompt  (mapped: one per prompt group)
    (one Arize op)             │           │
                               │           ▼
                               └───────►  create_optimized_prompt  (mapped)

1. **curate_feedback_dataset** — ``ArizeAxCurateFeedbackDatasetOperator``
   fetches LLM spans from your project, reads only OpenInference standard
   fields (``llm.input_messages`` for system prompts, ``llm.output_messages``
   for outputs, row-level ``evaluations`` + ``annotations`` for feedback),
   groups them by system-prompt fingerprint (or by a user-supplied metadata
   key like ``"agent"``), and emits one ``{prompt, dataset, group_key}`` per
   group ready for downstream mapping. Framework-agnostic.
2. **optimize_prompt** — ``ArizeAxOptimizePromptOperator`` expanded with
   ``.partial(...).expand_kwargs(curate.output)``. One task instance per
   prompt group. Each instance calls the Arize Prompt Learning SDK to
   produce a revised system prompt informed by that group's own feedback.
3. **create_optimized_prompt** — ``ArizeAxCreatePromptOperator`` expanded
   over the optimize output so every improved prompt lands in the Arize
   Prompt Hub. The created prompt name includes the ``group_key`` and the
   current date so each agent gets a uniquely-named prompt per run; on
   same-day retries ``if_exists="skip"`` returns the existing prompt ID
   without modifying anything. To version-bump under a single stable name
   instead (one prompt that accumulates revisions over time), use
   ``if_exists="add_version"``.

User configuration:
- Airflow connection ``arize_ax_default`` with API key.
- Variable ``arize_ax_project_id`` — base64 project ID to optimize.
- Variable ``arize_ax_space_id`` — space the new prompts get written to.
- Variable ``arize_ax_lookback_days`` (optional, default 30) — how far back
  to read production feedback.
- Variable ``arize_ax_group_by_metadata_key`` (optional, default unset) —
  set to e.g. ``"agent"`` or ``"langgraph_node"`` if your spans carry a
  per-agent metadata tag (yields readable group keys); otherwise spans
  group by system-prompt fingerprint.
- ``OPENAI_API_KEY`` in the worker environment.
- ``prompt-learning-enhanced`` installed in the worker — installed
  separately because PyPI rejects direct-URL deps so the provider can no
  longer declare it as an extra (Python 3.12+ recommended)::

      pip install 'arize-phoenix-evals>=2.0,<3.0' \\
                  'prompt-learning-enhanced @ git+https://github.com/Arize-ai/prompt-learning.git'

Point the DAG at a project and trigger it — every system prompt with
production feedback in the lookback window gets optimized and a new
version lands in Prompt Hub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from airflow import DAG
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxOptimizePromptOperator,
)
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxCurateFeedbackDatasetOperator,
)

OPTIMIZATION_MODEL = "gpt-4o"


def _build_create_prompt_kwargs(result: dict[str, Any]) -> dict[str, Any]:
    """Per-group kwargs for ArizeAxCreatePromptOperator.expand_kwargs(...).

    Defined at module level (rather than inline as a lambda on the .map call)
    so Airflow's mapped-task serialization works across all executors —
    lambdas pickle inconsistently under stricter executors.
    """
    group_key = result.get("group_key") or "default"
    return {
        "name": f"optimized-{group_key}-{datetime.now(timezone.utc):%Y%m%d}",
        "messages": result["messages"],
    }


with DAG(
    dag_id="example_arize_ax_prompt_optimization_with_feedback",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "example", "prompt_optimization", "self_learning"],
    catchup=False,
    doc_md=__doc__,
) as dag:
    curate = ArizeAxCurateFeedbackDatasetOperator(
        task_id="curate_feedback_dataset",
        project_id="{{ var.value.get('arize_ax_project_id', '') }}",
        start_time=(
            "{{ (logical_date - macros.timedelta("
            "days=(var.value.get('arize_ax_lookback_days', 30) | int)"
            ")).isoformat() }}"
        ),
        end_time="{{ logical_date.isoformat() }}",
        group_by_metadata_key="{{ var.value.get('arize_ax_group_by_metadata_key', '') or None }}",
        min_records_per_group=3,
    )

    # ArizeAxOptimizePromptOperator reads OPENAI_API_KEY from the worker
    # environment automatically when openai_api_key is omitted, so callers
    # don't need to bake the scheduler env into the DAG.
    optimize = ArizeAxOptimizePromptOperator.partial(
        task_id="optimize_prompt",
        model_choice=OPTIMIZATION_MODEL,
        feedback_columns=["feedback"],
        output_column="output",
    ).expand_kwargs(curate.output)

    create_optimized_prompt = ArizeAxCreatePromptOperator.partial(
        task_id="create_optimized_prompt",
        space_id="{{ var.value.get('arize_ax_space_id', '') or None }}",
        provider="open_ai",
        input_variable_format="mustache",
        model=OPTIMIZATION_MODEL,
        description=(
            "Optimized via ArizeAxOptimizePromptOperator from production "
            "eval + annotation feedback."
        ),
        if_exists="skip",
    ).expand_kwargs(optimize.output.map(_build_create_prompt_kwargs))
