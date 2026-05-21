"""
LLM Experiment Comparison DAG (server-side): runs 3 real experiments with
different model + prompt combinations against the same dataset, scored
end-to-end on the Arize Eval Hub so you can compare them side-by-side on
the Arize dashboard.

Experiment matrix
-----------------
+---+--------------+-----------+--------------------------------------+
| # | Model        | Prompt    | Purpose                              |
+---+--------------+-----------+--------------------------------------+
| 1 | gpt-4o-mini  | concise   | Baseline: cheap model, simple prompt |
| 2 | gpt-4.1      | concise   | Better model, same prompt            |
| 3 | gpt-4o-mini  | detailed  | Same cheap model, better prompt      |
+---+--------------+-----------+--------------------------------------+

Comparing (1 vs 2): model quality impact (holding prompt constant)
Comparing (1 vs 3): prompt engineering impact (holding model constant)

Three server-side LLM-as-Judge evaluators run on every experiment via
chained ``template_evaluation`` tasks (Eval Hub executes them server-side
after each run_experiment terminates):
  - **qa_correctness**: is the answer factually correct? (correct / incorrect)
  - **hallucination**: does the output contain fabricated information?
    (factual / hallucinated)
  - **relevance**: does the output address the question? (relevant / irrelevant)

Each judge produces score + label + explanation columns on the experiment
runs, giving rich per-row insights into each model's behavior.

Prereqs
-------
1. Airflow connection ``arize_ax_default`` with a valid API key.
2. Airflow Variable ``arize_ax_space_id`` (or ``default_space`` on the connection extras).
3. Airflow Variable ``arize_ai_integration_id`` (or env-var
   ``ARIZE_AI_INTEGRATION_ID``) pointing at an OpenAI integration with a
   real API key — used for **both** the run_experiment LLM calls and the
   chained LLM-as-judge evaluators. Server-side execution means there are
   no ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` on the Airflow worker.

For an Anthropic / multi-provider variant, fork this DAG and add an
additional AI integration ID + a second model_name + an
``ArizeAxCreateRunExperimentTaskOperator`` per provider.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
    ArizeAxGetDatasetOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxListExperimentsOperator,
)
from airflow.providers.arize_ax.operators.spaces import (
    ArizeAxListSpacesOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxCreateTaskOperator,
    ArizeAxDeleteTaskOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxDatasetReadySensor,
)
from airflow.providers.arize_ax.utils.task_groups import (
    arize_ax_chained_experiment_eval,
)
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor

INSPECTION_WINDOW_MINUTES = 15

# Single source of truth for the space all resources live in. Avoids the
# trap of letting `list_spaces`'s "first" entry pick a different space than
# the connection's default_space — which causes `create_task` to 404 looking
# for the evaluator in the wrong space.
_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

# Per-run suffix derived from run_id (works regardless of whether the DAG
# was triggered with --logical-date). We use this to give every ephemeral
# resource a name that's unique to the run, so we never collide with a
# tombstoned resource from a prior run that the Arize API still treats as
# name-taken (a soft-delete leakage we observed on evaluators).
_RUN_SUFFIX = (
    "{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') "
    "| replace('.', '') | replace('_', '') }}"
)

PROMPT_CONCISE = (
    "You are a helpful assistant. Answer the question in as few words as "
    "possible. Give only the direct answer with no explanation."
)

PROMPT_DETAILED = (
    "You are an expert knowledge assistant. Think step by step before "
    "answering. After your reasoning, provide the final answer on its own "
    "line prefixed with 'Answer: '. Be precise and accurate."
)

# (name, model, system_prompt) — single-provider (OpenAI) to keep the demo
# runnable on a single AI integration.
EXPERIMENT_CONFIGS: list[dict[str, str]] = [
    {"name": "gpt4omini-concise",  "model": "gpt-4o-mini", "prompt": PROMPT_CONCISE},
    {"name": "gpt41-concise",      "model": "gpt-4.1",     "prompt": PROMPT_CONCISE},
    {"name": "gpt4omini-detailed", "model": "gpt-4o-mini", "prompt": PROMPT_DETAILED},
]

DATASET_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 15 multiplied by 37?", "expected_output": "555"},
    {"query": "Who wrote Romeo and Juliet?", "expected_output": "William Shakespeare"},
    {"query": "What is the chemical formula for water?", "expected_output": "H2O"},
    {"query": "In what year did World War II end?", "expected_output": "1945"},
    {"query": "What is the largest planet in our solar system?", "expected_output": "Jupiter"},
    {"query": "What programming language was created by Guido van Rossum?", "expected_output": "Python"},
    {"query": "How many sides does a hexagon have?", "expected_output": "6"},
]


# Evaluator template strings. The `template_evaluation` config uses
# f-string single-brace substitution (server-validated at create time);
# the `run_experiment` task's `messages` field uses mustache double-brace
# substitution. The two surfaces have different rules.
QA_CORRECTNESS_TEMPLATE = (
    "You are evaluating a question-answering system.\n\n"
    "[Question]: {query}\n"
    "[Expected Answer]: {expected_output}\n"
    "[System Output]: {output}\n\n"
    "The system output is correct if it conveys the same factual answer as the "
    "expected answer, even if the phrasing differs. Minor wording differences "
    "are acceptable. Reply with ONLY 'correct' or 'incorrect'."
)

HALLUCINATION_TEMPLATE = (
    "You are evaluating whether an AI response contains hallucinated "
    "(fabricated or factually wrong) information.\n\n"
    "[Question]: {query}\n"
    "[Expected Answer]: {expected_output}\n"
    "[System Output]: {output}\n\n"
    "Reply with ONLY 'factual' (all claims accurate or consistent with the "
    "expected answer) or 'hallucinated' (contains made-up facts, wrong "
    "numbers, or fabricated details)."
)

RELEVANCE_TEMPLATE = (
    "You are evaluating whether an AI response is relevant to the question.\n\n"
    "[Question]: {query}\n"
    "[System Output]: {output}\n\n"
    "Reply with ONLY 'relevant' (directly addresses the question, even if it "
    "includes extra detail) or 'irrelevant' (does not attempt to answer the "
    "question or goes completely off-topic)."
)


def _resolve_integration_id() -> str:
    return (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )


def _build_judge_template_config(name: str, template: str, choices: dict[str, float]):
    """Factory: build a template_config dict for one of the three judges."""

    def _callable(**_ctx) -> dict[str, Any]:
        integration_id = _resolve_integration_id()
        return {
            "name": name,
            "template": template,
            "include_explanations": True,
            "use_function_calling_if_available": False,
            "classification_choices": choices,
            "llm_config": {
                "ai_integration_id": integration_id,
                "model_name": "gpt-4o-mini",
                "invocation_parameters": {"temperature": 0},
                "provider_parameters": {},
            },
        }

    _callable.__name__ = f"_build_{name}_template_config"
    return _callable


def _build_run_config(model: str, system_prompt: str):
    """Factory: build a LlmGenerationRunConfig dict for one experiment variant."""

    def _callable(**_ctx) -> dict[str, Any]:
        integration_id = _resolve_integration_id()
        return {
            "experiment_type": "llm_generation",
            "ai_integration_id": integration_id,
            "model_name": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                # Mustache + double-brace: the run_experiment surface uses
                # mustache substitution (different rule than the evaluator
                # template, which uses single-brace f-string).
                {"role": "user", "content": "{{query}}"},
            ],
            "input_variable_format": "mustache",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        }

    _callable.__name__ = f"_build_run_config_{model.replace('-', '_')}"
    return _callable


def _verify_experiment_results(**ctx) -> dict[str, Any]:
    """Log a summary of the chained experiment runs for the Airflow UI."""
    ti = ctx["ti"]
    summary: dict[str, Any] = {}
    for cfg in EXPERIMENT_CONFIGS:
        group_id = f"exp_{cfg['name'].replace('-', '_')}"
        triggered = ti.xcom_pull(task_ids=f"{group_id}.trigger", key="result") or {}
        summary[cfg["name"]] = {
            "experiment_id": triggered.get("experiment_id") if isinstance(triggered, dict) else None,
            "chained_run_id": triggered.get("id") if isinstance(triggered, dict) else None,
        }
    print("=" * 60)
    print("LLM EXPERIMENT COMPARISON COMPLETE (server-side)")
    for name, info in summary.items():
        print(f"  {name:25} experiment={info['experiment_id']}")
    print("  Compare them in Arize: Datasets → <dataset> → Experiments tab.")
    print("=" * 60)
    return summary


with DAG(
    dag_id="arize_ax_e2e_llm_experiment_comparison",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "e2e", "experiments", "llm", "comparison", "eval-hub"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    # ----- Phase 1: dataset + space lookup ----------------------------------
    list_spaces = ArizeAxListSpacesOperator(
        task_id="list_spaces",
        limit=10,
    )

    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id=_SPACE_JINJA,
        name="llm-experiments-{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}",
        examples=DATASET_EXAMPLES,
        if_exists="skip",
    )

    get_dataset = ArizeAxGetDatasetOperator(
        task_id="get_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
    )

    dataset_ready = ArizeAxDatasetReadySensor(
        task_id="dataset_ready_sensor",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        min_examples=len(DATASET_EXAMPLES),
        poke_interval=5,
        timeout=120,
        mode="poke",
    )

    # ----- Phase 2: three server-side judges (template_evaluation tasks) -----
    build_qa_config = PythonOperator(
        task_id="build_qa_config",
        python_callable=_build_judge_template_config(
            "qa_correctness", QA_CORRECTNESS_TEMPLATE,
            {"correct": 1.0, "incorrect": 0.0},
        ),
    )
    create_qa_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_qa_evaluator",
        space_id=_SPACE_JINJA,
        name=f"qa_correctness_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial qa_correctness judge",
        template_config_task_id="build_qa_config",
        description="LLM-as-judge for factual correctness.",
    )

    build_hallu_config = PythonOperator(
        task_id="build_hallu_config",
        python_callable=_build_judge_template_config(
            "hallucination", HALLUCINATION_TEMPLATE,
            {"factual": 1.0, "hallucinated": 0.0},
        ),
    )
    create_hallu_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_hallu_evaluator",
        space_id=_SPACE_JINJA,
        name=f"hallucination_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial hallucination judge",
        template_config_task_id="build_hallu_config",
        description="LLM-as-judge for fabricated content.",
    )

    build_relevance_config = PythonOperator(
        task_id="build_relevance_config",
        python_callable=_build_judge_template_config(
            "relevance", RELEVANCE_TEMPLATE,
            {"relevant": 1.0, "irrelevant": 0.0},
        ),
    )
    create_relevance_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_relevance_evaluator",
        space_id=_SPACE_JINJA,
        name=f"relevance_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial relevance judge",
        template_config_task_id="build_relevance_config",
        description="LLM-as-judge for question relevance.",
    )

    create_qa_task = ArizeAxCreateTaskOperator(
        task_id="create_qa_task",
        name="llm-exp-qa-{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_qa_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )
    create_hallu_task = ArizeAxCreateTaskOperator(
        task_id="create_hallu_task",
        name="llm-exp-hallu-{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_hallu_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )
    create_relevance_task = ArizeAxCreateTaskOperator(
        task_id="create_relevance_task",
        name="llm-exp-relevance-{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_relevance_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    # ----- Phase 3: 3 run_experiment tasks (server-side LLM calls) ----------
    build_run_config_tasks: dict[str, PythonOperator] = {}
    create_run_exp_tasks: dict[str, ArizeAxCreateRunExperimentTaskOperator] = {}
    chain_groups: list = []
    cleanup_run_exp_tasks: list = []

    for cfg in EXPERIMENT_CONFIGS:
        suffix = cfg["name"].replace("-", "_")

        build_cfg = PythonOperator(
            task_id=f"build_run_config_{suffix}",
            python_callable=_build_run_config(cfg["model"], cfg["prompt"]),
        )
        build_run_config_tasks[cfg["name"]] = build_cfg

        create_task = ArizeAxCreateRunExperimentTaskOperator(
            task_id=f"create_run_exp_task_{suffix}",
            space_id=_SPACE_JINJA,
            name=f"llm-exp-task-{cfg['name']}-{{{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}}}",
            dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
            run_configuration=(
                f"{{{{ ti.xcom_pull(task_ids='build_run_config_{suffix}') }}}}"
            ),
            if_exists="skip",
        )
        create_run_exp_tasks[cfg["name"]] = create_task

        # Chain all three evaluators after each run_experiment.
        group = arize_ax_chained_experiment_eval(
            group_id=f"exp_{suffix}",
            task_id_param=(
                f"{{{{ ti.xcom_pull(task_ids='create_run_exp_task_{suffix}') }}}}"
            ),
            experiment_name=f"{cfg['name']}-{{{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') | replace('.', '') | replace('_', '') }}}}",
            space_id=_SPACE_JINJA,
            evaluation_task_ids=[
                "{{ ti.xcom_pull(task_ids='create_qa_task') }}",
                "{{ ti.xcom_pull(task_ids='create_hallu_task') }}",
                "{{ ti.xcom_pull(task_ids='create_relevance_task') }}",
            ],
            wait_for_completion=True,
            sensor_timeout=900,
            sensor_poke_interval=15,
            fail_on_run_error=True,
        )
        chain_groups.append(group)

        cleanup_run_exp_tasks.append(
            ArizeAxDeleteTaskOperator(
                task_id=f"cleanup_run_exp_task_{suffix}",
                task_id_param=(
                    f"{{{{ ti.xcom_pull(task_ids='create_run_exp_task_{suffix}') }}}}"
                ),
                ignore_if_missing=True,
                trigger_rule="all_done",
            )
        )

        # Wire deps within the per-experiment branch
        [dataset_ready, build_cfg] >> create_task
        [
            create_task,
            create_qa_task,
            create_hallu_task,
            create_relevance_task,
        ] >> group

    list_experiments = ArizeAxListExperimentsOperator(
        task_id="list_experiments",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=20,
        trigger_rule="all_done",
    )

    verify = PythonOperator(
        task_id="verify_results",
        python_callable=_verify_experiment_results,
        trigger_rule="all_done",
    )

    inspection_window = TimeDeltaSensor(
        task_id="inspection_window",
        delta=timedelta(minutes=INSPECTION_WINDOW_MINUTES),
    )

    # ----- Phase 4: cleanup -------------------------------------------------
    cleanup_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_qa_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_qa_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_qa_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_hallu_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_hallu_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_hallu_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_relevance_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_relevance_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_relevance_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_qa_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_qa_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_qa_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_hallu_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_hallu_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_hallu_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_relevance_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_relevance_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_relevance_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # ----- Top-level wiring -------------------------------------------------
    list_spaces >> create_dataset >> [get_dataset, dataset_ready]
    list_spaces >> [build_qa_config, build_hallu_config, build_relevance_config]
    build_qa_config >> create_qa_evaluator >> create_qa_task
    build_hallu_config >> create_hallu_evaluator >> create_hallu_task
    build_relevance_config >> create_relevance_evaluator >> create_relevance_task

    chain_groups >> list_experiments >> verify >> inspection_window
    inspection_window >> [
        cleanup_dataset,
        *cleanup_run_exp_tasks,
        cleanup_qa_task,
        cleanup_hallu_task,
        cleanup_relevance_task,
        cleanup_qa_evaluator,
        cleanup_hallu_evaluator,
        cleanup_relevance_evaluator,
    ]
