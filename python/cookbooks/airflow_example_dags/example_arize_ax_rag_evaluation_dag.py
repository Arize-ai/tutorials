"""
RAG Quality Evaluation DAG: automated daily faithfulness and relevance scoring.

This DAG implements an automated daily pipeline that:
1. Exports production RAG spans (RETRIEVER + LLM) from the previous 24 hours.
2. Packages them into an evaluation dataset.
3. Runs two Python evaluator experiments:
   - **faithfulness**: does the LLM output stay grounded in the retrieved
     context?  Score 1.0 = fully grounded, 0.0 = hallucinated.
   - **context_relevance**: is the retrieved context actually relevant to the
     user query?  Score 1.0 = highly relevant, 0.0 = irrelevant.
4. Scores both experiments and logs a quality summary.

Metrics computed
----------------
- ``faithfulness``: mean faithfulness score across all evaluated spans.
- ``context_relevance``: mean context relevance score across evaluated spans.

The ``report_rag_quality`` task pushes a summary dict to XCom so downstream
alerting DAGs (e.g. a Slack notifier) can read it with
``xcom_pull(dag_id="arize_ax_rag_evaluation", task_ids="report_rag_quality")``.

Variables
---------
- ``arize_ax_project_name`` — Arize project **name** containing production RAG spans.
  If absent, the first project returned by list_projects is used.
- ``arize_ax_space_id``     — Arize space ID used to export spans and create datasets.
  Required: ArizeAxListProjectsOperator and ArizeAxCreateDatasetOperator will raise
  AirflowException with a clear message if this Variable is absent.

Schedule
--------
Daily at midnight UTC (``@daily``).  Each run evaluates the spans from the
previous calendar day via the Jinja macros ``data_interval_start`` and
``data_interval_end``.

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Variable ``arize_ax_space_id`` — required by list_projects and create_dataset operators.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxAppendDatasetExamplesOperator,
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxGetExperimentScoreOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.projects import (
    ArizeAxListProjectsOperator,
)
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxSpansExportToParquetOperator,
)
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

_TMP_PARQUET = "/tmp/rag_spans_{{ ds }}.parquet"
_RAG_SPAN_WHERE = (
    "attributes.openinference.span.kind='RETRIEVER' "
    "OR attributes.openinference.span.kind='LLM'"
)


def _extract_project_name(**ctx) -> str:
    """Return the project name from ``arize_ax_project_name`` Variable.

    Falls back to the name of the first project returned by
    ArizeAxListProjectsOperator if the Variable is not set.
    """
    project_name_var = Variable.get("arize_ax_project_name", default_var=None)
    if project_name_var:
        return project_name_var

    result = ctx["ti"].xcom_pull(task_ids="check_project_exists")
    items = result.get("items", []) if isinstance(result, dict) else []
    if not items:
        raise ValueError(
            "No projects found. Set Variable 'arize_ax_project_name' to your project name."
        )
    first_item = items[0]
    name = first_item.get("name") if isinstance(first_item, dict) else None
    if not name:
        raise ValueError(
            "Project item has no 'name' field. "
            "Set Variable 'arize_ax_project_name' explicitly."
        )
    return str(name)


def _check_has_spans(**ctx) -> bool:
    """Return True if the parquet export produced at least one span."""
    parquet_path = ctx["ti"].xcom_pull(task_ids="export_rag_spans")
    if not parquet_path:
        print("[check_has_spans] export task returned no path -- short-circuiting.")
        return False
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        count = len(df)
        print(f"[check_has_spans] parquet has {count} rows.")
        return count > 0
    except Exception as exc:
        print(f"[check_has_spans] could not read parquet: {exc} -- short-circuiting.")
        return False


def _prepare_rag_examples(**ctx) -> list[dict[str, Any]]:
    """Load the exported parquet and extract input / output / context fields.

    Returns a list of dataset example dicts ready for
    ArizeAxAppendDatasetExamplesOperator.

    Expected span columns (best-effort -- missing columns yield empty strings):
    - ``input.value``   — the user query sent to the LLM / retriever
    - ``output.value``  — the LLM response
    - ``attributes.retrieval.documents`` — JSON list of retrieved document texts
    """
    import json

    import pandas as pd

    parquet_path = ctx["ti"].xcom_pull(task_ids="export_rag_spans")
    if not parquet_path:
        raise ValueError("No parquet path in XCom from export_rag_spans.")

    df = pd.read_parquet(parquet_path)
    examples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        input_val = str(row.get("input.value", row.get("input", "")))
        output_val = str(row.get("output.value", row.get("output", "")))
        raw_docs = row.get("attributes.retrieval.documents", row.get("retrieval.documents", ""))
        if isinstance(raw_docs, str):
            try:
                docs = json.loads(raw_docs)
                context = " ".join(
                    d.get("document.content", "") if isinstance(d, dict) else str(d)
                    for d in (docs if isinstance(docs, list) else [])
                )
            except (json.JSONDecodeError, TypeError):
                context = raw_docs
        elif isinstance(raw_docs, list):
            context = " ".join(
                d.get("document.content", "") if isinstance(d, dict) else str(d)
                for d in raw_docs
            )
        else:
            context = ""

        if not input_val and not output_val:
            continue  # skip empty rows

        examples.append({
            "query": input_val,
            "response": output_val,
            "context": context,
        })

    print(f"[prepare_rag_examples] produced {len(examples)} dataset examples.")
    return examples


def _faithfulness_evaluator(dataset_row: dict, output: Any) -> Any:
    """Score whether the LLM response stays grounded in the retrieved context.

    Uses a simple heuristic: if any non-trivial word from the response also
    appears in the context, score = 1.0 (grounded); otherwise 0.0 (potential
    hallucination).  Replace this with a real LLM-as-judge in production.
    """
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")

    response = dataset_row.get("response", "")
    context = dataset_row.get("context", "")

    if not response or not context:
        return EvaluationResult(
            score=0.5,
            label="unknown",
            explanation="insufficient data to evaluate faithfulness",
        )

    response_words = {
        w.lower() for w in response.split() if len(w) > 4
    }
    context_words = {
        w.lower() for w in context.split() if len(w) > 4
    }
    overlap = response_words & context_words
    grounded = len(overlap) >= max(1, len(response_words) // 4)

    return EvaluationResult(
        score=1.0 if grounded else 0.0,
        label="grounded" if grounded else "hallucinated",
        explanation=(
            f"Overlap {len(overlap)}/{len(response_words)} meaningful words; "
            f"threshold={max(1, len(response_words) // 4)}."
        ),
    )


def _context_relevance_evaluator(dataset_row: dict, output: Any) -> Any:
    """Score whether the retrieved context is relevant to the user query.

    Uses a simple heuristic: if any word from the query (>3 chars) appears in
    the context, score = 1.0 (relevant); otherwise 0.0 (irrelevant).  Replace
    this with a real LLM-as-judge in production.
    """
    from arize.experiments import EvaluationResult

    query = dataset_row.get("query", "")
    context = dataset_row.get("context", "")

    if not query or not context:
        return EvaluationResult(
            score=0.5,
            label="unknown",
            explanation="insufficient data to evaluate context relevance",
        )

    query_words = {w.lower() for w in query.split() if len(w) > 3}
    context_words = {w.lower() for w in context.split() if len(w) > 3}
    overlap = query_words & context_words
    relevant = bool(overlap)

    return EvaluationResult(
        score=1.0 if relevant else 0.0,
        label="relevant" if relevant else "irrelevant",
        explanation=f"Query–context word overlap: {overlap or 'none'}.",
    )


# ---------------------------------------------------------------------------
# Identity task: passes span data through to the experiment
# ---------------------------------------------------------------------------
def _rag_task(dataset_row: dict) -> dict:
    """Identity task: surface the span's response field as the experiment output."""
    return {
        "output": dataset_row.get("response", ""),
        "query": dataset_row.get("query", ""),
        "context": dataset_row.get("context", ""),
    }


def _report_rag_quality(**ctx) -> dict[str, Any]:
    """Log a RAG quality summary and push scores to XCom for downstream alerting."""
    ti = ctx["ti"]
    ds = ctx.get("ds", "unknown")
    faithfulness_scores = ti.xcom_pull(task_ids="score_faithfulness") or {}
    relevance_scores = ti.xcom_pull(task_ids="score_relevance") or {}

    faithfulness = faithfulness_scores.get("_faithfulness_evaluator")
    relevance = relevance_scores.get("_context_relevance_evaluator")

    summary = {
        "date": ds,
        "faithfulness_score": faithfulness,
        "context_relevance_score": relevance,
    }

    print("=" * 60)
    print(f"RAG QUALITY REPORT — {ds}")
    print(f"  Faithfulness score   : {faithfulness}")
    print(f"  Context relevance    : {relevance}")
    if faithfulness is not None and faithfulness < 0.7:
        print("  WARNING: faithfulness below 0.7 — possible hallucination increase.")
    if relevance is not None and relevance < 0.7:
        print("  WARNING: context relevance below 0.7 — retrieval quality degraded.")
    print("=" * 60)
    return summary


with DAG(
    dag_id="arize_ax_rag_evaluation",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "rag", "evaluation", "daily"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    # Stage 1 — Discover project
    check_project_exists = ArizeAxListProjectsOperator(
        task_id="check_project_exists",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        limit=50,
    )

    extract_project_name = PythonOperator(
        task_id="extract_project_name",
        python_callable=_extract_project_name,
    )

    # Stage 2 — Export RAG spans from the previous 24 hours
    export_rag_spans = ArizeAxSpansExportToParquetOperator(
        task_id="export_rag_spans",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        project_name="{{ ti.xcom_pull(task_ids='extract_project_name') }}",
        path=_TMP_PARQUET,
        where=_RAG_SPAN_WHERE,
        # Manual triggers leave data_interval_start == data_interval_end (both
        # equal logical_date), and the SDK's exporter validates start < end
        # strictly. Compute a 24h window off logical_date so manual triggers
        # work; scheduled runs still see (close to) one schedule interval.
        start_time="{{ (logical_date - macros.timedelta(hours=24)).isoformat() }}",
        end_time="{{ logical_date.isoformat() }}",
    )

    check_has_spans = ShortCircuitOperator(
        task_id="check_has_spans",
        python_callable=_check_has_spans,
    )

    # Stage 3 — Build evaluation dataset
    create_rag_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_rag_eval_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name="rag-eval-{{ ds }}",
        # SDK 8.25+ rejects empty examples list at create time. Seed with a
        # single placeholder; real RAG examples are appended by
        # prepare_rag_examples once spans are exported.
        examples=[{"input": "_seed_", "expected": "_seed_"}],
        if_exists="skip",
    )

    prepare_rag_examples = PythonOperator(
        task_id="prepare_rag_examples",
        python_callable=_prepare_rag_examples,
    )

    append_rag_examples = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_rag_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_rag_eval_dataset') }}",
        examples="{{ ti.xcom_pull(task_ids='prepare_rag_examples') }}",
    )

    # Stage 4 — Run evaluation experiments
    run_faithfulness_eval = ArizeAxRunExperimentOperator(
        task_id="run_faithfulness_eval",
        name="rag-faithfulness-{{ ds }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_rag_eval_dataset') }}",
        task=_rag_task,
        evaluators=[_faithfulness_evaluator],
        concurrency=4,
    )

    run_relevance_eval = ArizeAxRunExperimentOperator(
        task_id="run_relevance_eval",
        name="rag-relevance-{{ ds }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_rag_eval_dataset') }}",
        task=_rag_task,
        evaluators=[_context_relevance_evaluator],
        concurrency=4,
    )

    # Stage 5 — Score experiments
    score_faithfulness = ArizeAxGetExperimentScoreOperator(
        task_id="score_faithfulness",
        experiment_id="{{ ti.xcom_pull(task_ids='run_faithfulness_eval') }}",
        aggregation="mean",
    )

    score_relevance = ArizeAxGetExperimentScoreOperator(
        task_id="score_relevance",
        experiment_id="{{ ti.xcom_pull(task_ids='run_relevance_eval') }}",
        aggregation="mean",
    )

    # Stage 6 — Report
    report_rag_quality = PythonOperator(
        task_id="report_rag_quality",
        python_callable=_report_rag_quality,
        trigger_rule="all_done",
    )

    # Wiring
    check_project_exists >> extract_project_name >> export_rag_spans >> check_has_spans

    check_has_spans >> create_rag_eval_dataset >> prepare_rag_examples >> append_rag_examples

    append_rag_examples >> [run_faithfulness_eval, run_relevance_eval]

    run_faithfulness_eval >> score_faithfulness
    run_relevance_eval >> score_relevance

    [score_faithfulness, score_relevance] >> report_rag_quality
