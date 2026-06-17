"""
RAG Quality Evaluation DAG (server-side, self-contained):
faithfulness and context-relevance scoring for a context-augmented LLM.

This DAG demonstrates RAG-style evaluation end-to-end without depending on
any production span pipeline. It builds a small synthetic dataset of
``(query, context, expected_output)`` rows, registers a server-side
``run_experiment`` task that drives a context-aware ``gpt-5.5`` LLM call,
and chains two ``template_evaluation`` LLM-as-judges that run server-side
after each row's LLM response is produced:

- **faithfulness** — does the model's answer stay grounded in the
  retrieved context, or does it hallucinate beyond it?
- **context_relevance** — was the retrieved context actually relevant to
  the user query in the first place?

It is fully self-contained — no pre-existing project, dataset, evaluator,
or production spans required. Everything is provisioned and torn down on
every run.

Pipeline stages
---------------
0. **check_prereqs** — short-circuit if ``ARIZE_AI_INTEGRATION_ID`` isn't
   resolvable.
1. **create_rag_dataset** — provision a tiny synthetic RAG dataset.
2. **build_faithfulness_judge_config / create_faithfulness_evaluator /
   create_faithfulness_eval_task** — the LLM-as-judge that scores whether
   each response is grounded in its context.
3. **build_relevance_judge_config / create_relevance_evaluator /
   create_relevance_eval_task** — the LLM-as-judge that scores whether
   each context is relevant to its query.
4. **build_rag_run_config / create_rag_run_exp_task** — register a
   ``run_experiment`` task whose system prompt asks the model to answer
   strictly from the provided context.
5. **run_rag_experiment.{trigger,wait,get_result}** — fire the experiment
   with **both** judges chained via ``evaluation_task_ids``; Arize
   executes faithfulness + context_relevance server-side after each row.
6. **score_faithfulness / score_context_relevance** — aggregate the two
   eval metrics from the chained experiment runs.
7. **report_rag_quality** — log a quality summary and push it to XCom for
   downstream alerting.
8. **cleanup_*** — delete the per-run-ephemeral evaluators, eval tasks,
   run-experiment task, and dataset.

Schedule: ``@daily`` (each manual or scheduled run is fully self-contained).

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_project_id`` — project ID used to scope the eval tasks.
- ``arize_ai_integration_id`` *or* env var ``ARIZE_AI_INTEGRATION_ID`` —
  OpenAI integration in Arize with gpt-5.5 access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxGetExperimentScoreOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxCreateTaskOperator,
    ArizeAxDeleteTaskOperator,
)
from airflow.providers.arize_ax.utils.task_groups import (
    arize_ax_chained_experiment_eval,
)
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

# Synthetic RAG-style dataset: each row has a query, a retrieved context
# block, and a reference answer. The LLM is asked to answer from the
# context; the two judges score faithfulness (output vs context) and
# context_relevance (context vs query).
RAG_EVAL_EXAMPLES = [
    {
        "query": "What is the capital of France?",
        "context": (
            "France is a country in Western Europe. Its capital and "
            "largest city is Paris, on the Seine river."
        ),
        "expected_output": "Paris",
    },
    {
        "query": "How many continents are there?",
        "context": (
            "Geographers traditionally divide the world into seven "
            "continents: Africa, Antarctica, Asia, Australia, Europe, "
            "North America, and South America."
        ),
        "expected_output": "7",
    },
    {
        "query": "Who wrote Hamlet?",
        "context": (
            "Hamlet is a tragedy written by the English playwright "
            "William Shakespeare around the year 1600."
        ),
        "expected_output": "William Shakespeare",
    },
    {
        "query": "What element has atomic number 1?",
        "context": (
            "Hydrogen, with chemical symbol H and atomic number 1, is "
            "the lightest element in the periodic table."
        ),
        "expected_output": "Hydrogen",
    },
    {
        "query": "What is the largest planet in our solar system?",
        "context": (
            "Jupiter is the fifth planet from the Sun and the largest "
            "in our solar system — over twice as massive as all the "
            "other planets combined."
        ),
        "expected_output": "Jupiter",
    },
]

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_RUN_SUFFIX = (
    "{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') "
    "| replace('.', '') | replace('_', '') }}"
)

FAITHFULNESS_TEMPLATE = (
    "[Context]: {context}\n"
    "[Output]: {output}\n\n"
    "Is the output fully grounded in the context (every claim supported), "
    "or does it introduce information not present in the context?\n"
    "Reply with ONLY 'faithful' (no fabricated content) or 'hallucinated' "
    "(includes claims not supported by the context)."
)

CONTEXT_RELEVANCE_TEMPLATE = (
    "[Question]: {query}\n"
    "[Context]: {context}\n\n"
    "Is the context relevant to answering the question?\n"
    "Reply with ONLY 'relevant' (the context contains information that "
    "addresses the question) or 'irrelevant' (the context does not help "
    "answer the question)."
)

# The run_experiment system prompt enforces the RAG contract: answer only
# from the provided context. The user message bundles query + context
# together using mustache substitution.
RAG_SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "you don't know. Give only the direct factual answer — no extra "
    "explanation or punctuation."
)
RAG_USER_TEMPLATE = "Context: {{context}}\n\nQuestion: {{query}}"


def _resolve_integration_id() -> str:
    return (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )


def _check_prereqs(**_ctx) -> bool:
    if not _resolve_integration_id():
        print(
            "ARIZE_AI_INTEGRATION_ID env var or arize_ai_integration_id "
            "Variable must be set."
        )
        return False
    return True


def _build_judge_config(name: str, template: str, choices: dict[str, float]):
    """Factory: build a template_config dict for one of the two RAG judges."""

    def _callable(**_ctx) -> dict[str, Any]:
        return {
            "name": name,
            "template": template,
            "include_explanations": True,
            "use_function_calling_if_available": False,
            "classification_choices": choices,
            "llm_config": {
                "ai_integration_id": _resolve_integration_id(),
                "model_name": "gpt-5.4-mini",
                "invocation_parameters": {},
                "provider_parameters": {},
            },
        }

    _callable.__name__ = f"_build_{name}_judge_config"
    return _callable


def _build_rag_run_config(**_ctx) -> dict[str, Any]:
    return {
        "experiment_type": "llm_generation",
        "ai_integration_id": _resolve_integration_id(),
        "model_name": "gpt-5.5",
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_USER_TEMPLATE},
        ],
        "input_variable_format": "mustache",
        "invocation_parameters": {},
        "provider_parameters": {},
    }


def _report_rag_quality(**ctx) -> dict[str, Any]:
    """Log RAG quality summary; push to XCom for downstream alerting."""
    ti = ctx["ti"]
    ds = ctx.get("ds", "unknown")
    faithfulness_scores = ti.xcom_pull(task_ids="score_faithfulness") or {}
    relevance_scores = ti.xcom_pull(task_ids="score_context_relevance") or {}

    summary = {
        "date": ds,
        "faithfulness_scores": faithfulness_scores,
        "context_relevance_scores": relevance_scores,
    }

    print("=" * 60)
    print(f"RAG QUALITY REPORT — {ds}")
    print(f"  Faithfulness        : {faithfulness_scores}")
    print(f"  Context relevance   : {relevance_scores}")
    print("=" * 60)
    return summary


with DAG(
    dag_id="arize_ax_rag_evaluation",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "rag", "evaluation", "eval-hub", "self-contained"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    # ----- Phase 1: ephemeral RAG dataset ----------------------------------
    create_rag_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_rag_dataset",
        space_id=_SPACE_JINJA,
        name=f"rag-eval-dataset-{_RUN_SUFFIX}",
        examples=RAG_EVAL_EXAMPLES,
        if_exists="skip",
    )

    # ----- Phase 2: two LLM-as-judges (faithfulness + context_relevance) ----
    build_faithfulness_judge_config = PythonOperator(
        task_id="build_faithfulness_judge_config",
        python_callable=_build_judge_config(
            "faithfulness", FAITHFULNESS_TEMPLATE,
            {"faithful": 1.0, "hallucinated": 0.0},
        ),
    )
    create_faithfulness_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_faithfulness_evaluator",
        space_id=_SPACE_JINJA,
        name=f"rag_faithfulness_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial faithfulness judge",
        template_config_task_id="build_faithfulness_judge_config",
        description="LLM-as-judge for RAG response groundedness in context.",
    )
    create_faithfulness_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_faithfulness_eval_task",
        name=f"rag-faithfulness-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_faithfulness_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    build_relevance_judge_config = PythonOperator(
        task_id="build_relevance_judge_config",
        python_callable=_build_judge_config(
            "context_relevance", CONTEXT_RELEVANCE_TEMPLATE,
            {"relevant": 1.0, "irrelevant": 0.0},
        ),
    )
    create_relevance_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_relevance_evaluator",
        space_id=_SPACE_JINJA,
        name=f"rag_context_relevance_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial context_relevance judge",
        template_config_task_id="build_relevance_judge_config",
        description="LLM-as-judge for RAG retrieval-context relevance.",
    )
    create_relevance_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_relevance_eval_task",
        name=f"rag-context-relevance-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_relevance_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    # ----- Phase 3: RAG run_experiment task ---------------------------------
    build_rag_run_config = PythonOperator(
        task_id="build_rag_run_config",
        python_callable=_build_rag_run_config,
    )
    create_rag_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_rag_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"rag-run-exp-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_rag_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_rag_run_config') }}",
        if_exists="skip",
    )

    # ----- Phase 4: chained trigger + wait + get_result ---------------------
    # Both judges fan out server-side after each row's LLM response is
    # generated. Eval Hub executes them and writes per-row labels/scores
    # into experiment_runs.additional_properties["eval.<name>.*"].
    run_rag_experiment = arize_ax_chained_experiment_eval(
        group_id="run_rag_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_rag_run_exp_task') }}",
        experiment_name=f"rag-eval-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_faithfulness_eval_task') }}",
            "{{ ti.xcom_pull(task_ids='create_relevance_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    # ----- Phase 5: score both metrics --------------------------------------
    _RAG_EXP_ID = (
        "{{ ti.xcom_pull(task_ids='run_rag_experiment.trigger', "
        "key='result')['experiment_id'] }}"
    )
    score_faithfulness = ArizeAxGetExperimentScoreOperator(
        task_id="score_faithfulness",
        experiment_id=_RAG_EXP_ID,
        aggregation="mean",
    )
    score_context_relevance = ArizeAxGetExperimentScoreOperator(
        task_id="score_context_relevance",
        experiment_id=_RAG_EXP_ID,
        aggregation="mean",
    )

    # ----- Phase 6: report --------------------------------------------------
    report_rag_quality = PythonOperator(
        task_id="report_rag_quality",
        python_callable=_report_rag_quality,
        trigger_rule="all_done",
    )

    # ----- Phase 7: cleanup -------------------------------------------------
    cleanup_faithfulness_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_faithfulness_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_faithfulness_eval_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_relevance_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_relevance_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_relevance_eval_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_rag_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_rag_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_rag_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_faithfulness_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_faithfulness_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_faithfulness_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_relevance_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_relevance_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_relevance_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_rag_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # Wiring
    check_prereqs >> [
        create_rag_dataset,
        build_faithfulness_judge_config,
        build_relevance_judge_config,
        build_rag_run_config,
    ]
    build_faithfulness_judge_config >> create_faithfulness_evaluator >> create_faithfulness_eval_task
    build_relevance_judge_config >> create_relevance_evaluator >> create_relevance_eval_task
    [create_rag_dataset, build_rag_run_config] >> create_rag_run_exp_task
    [
        create_rag_run_exp_task,
        create_faithfulness_eval_task,
        create_relevance_eval_task,
    ] >> run_rag_experiment
    run_rag_experiment >> [score_faithfulness, score_context_relevance] >> report_rag_quality
    report_rag_quality >> [
        cleanup_faithfulness_eval_task,
        cleanup_relevance_eval_task,
        cleanup_rag_run_exp_task,
        cleanup_faithfulness_evaluator,
        cleanup_relevance_evaluator,
        cleanup_dataset,
    ]
