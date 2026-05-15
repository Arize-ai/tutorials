"""
LLM Experiment Comparison DAG: runs 5 real experiments with different model and
prompt combinations against the same dataset, so you can compare scores
side-by-side on the Arize dashboard.

Experiment matrix
-----------------
+---+--------------------+-----------+--------------------------------------+
| # | Model              | Prompt    | Purpose                              |
+---+--------------------+-----------+--------------------------------------+
| 1 | gpt-4o-mini        | concise   | Baseline: cheap model, simple prompt |
| 2 | gpt-4o             | concise   | Better model, same prompt            |
| 3 | claude-3-haiku     | concise   | Different provider, same prompt      |
| 4 | gpt-4o-mini        | detailed  | Same cheap model, better prompt      |
| 5 | claude-3-haiku     | detailed  | Cross-provider prompt impact         |
+---+--------------------+-----------+--------------------------------------+

Comparing (1 vs 2): model quality impact (holding prompt constant)
Comparing (1 vs 3): provider comparison (same tier, different vendor)
Comparing (1 vs 4): prompt engineering impact (holding model constant)
Comparing (3 vs 5): prompt impact on same model, different provider

Three LLM-as-Judge evaluators run on every experiment (using GPT-4o-mini
as the judge via ``arize-phoenix-evals >= 3.0.0``):
  - **qa_correctness**: is the answer factually correct? (correct / incorrect)
  - **hallucination**: does the output contain fabricated information?
    (factual / hallucinated)
  - **relevance**: does the output address the question? (relevant / irrelevant)

These evaluators produce score + label + explanation columns on the Arize
dashboard, giving rich per-row insights into each model's behavior.

Prompts are created in the Arize Prompt Hub using the
``ArizeAxCreatePromptOperator`` and cleaned up at the end with
``ArizeAxDeletePromptOperator``. Note: the Arize SDK ``experiments.run()`` API
does not currently accept a prompt_id or prompt_version_id parameter, so
experiments run via this DAG are not automatically linked to those Prompt Hub
prompts in the Arize UI (you may see "No prompt template available" on the
experiment row). Linking prompts to experiments is supported when running from
the Arize Prompt Playground or when the API adds support. See
``docs/arize-experiments-and-prompts-analysis.md`` for a full analysis of the
API/SDK and options.

Requires:
  - Airflow connection ``arize_ax_default`` with valid API key + space_id.
  - Environment variables ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` set in
    the Airflow worker (e.g. via docker-compose environment section).
  - Python packages: ``openai``, ``anthropic``, ``arize-phoenix-evals``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
    ArizeAxGetDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxListExperimentsOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.projects import (
    ArizeAxCreateProjectOperator,
    ArizeAxDeleteProjectOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxDeletePromptOperator,
)
from airflow.providers.arize_ax.operators.spaces import (
    ArizeAxListSpacesOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxDatasetReadySensor,
)

try:
    from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor
except ImportError:
    from airflow.sensors.time_delta import TimeDeltaSensor

INSPECTION_WINDOW_MINUTES = 15

PROMPT_CONCISE = (
    "You are a helpful assistant. Answer the question in as few words as "
    "possible. Give only the direct answer with no explanation."
)

PROMPT_DETAILED = (
    "You are an expert knowledge assistant. Think step by step before "
    "answering. After your reasoning, provide the final answer on its own "
    "line prefixed with 'Answer: '. Be precise and accurate."
)

EXPERIMENT_CONFIGS: list[dict[str, str]] = [
    {"name": "gpt4omini-concise", "provider": "openai", "model": "gpt-4o-mini", "prompt": PROMPT_CONCISE},
    {"name": "gpt4o-concise", "provider": "openai", "model": "gpt-4o", "prompt": PROMPT_CONCISE},
    {"name": "haiku-concise", "provider": "anthropic", "model": "claude-3-haiku-20240307", "prompt": PROMPT_CONCISE},
    {"name": "gpt4omini-detailed", "provider": "openai", "model": "gpt-4o-mini", "prompt": PROMPT_DETAILED},
    {"name": "haiku-detailed", "provider": "anthropic", "model": "claude-3-haiku-20240307", "prompt": PROMPT_DETAILED},
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


def _make_llm_task(provider: str, model: str, system_prompt: str):
    """Return a task callable that calls the specified LLM."""

    def _task(dataset_row: dict) -> str:
        query = dataset_row.get("query", "")
        if provider == "openai":
            import openai

            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        elif provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text.strip()

        raise ValueError(f"Unknown provider: {provider}")

    _task.__name__ = f"task_{provider}_{model.replace('-', '_').replace('.', '_')}"
    return _task


QA_CORRECTNESS_TEMPLATE = (
    "You are evaluating a question-answering system.\n\n"
    "[BEGIN DATA]\n"
    "[Question]: {query}\n"
    "[Expected Answer]: {expected_output}\n"
    "[System Output]: {output}\n"
    "[END DATA]\n\n"
    "The system output is correct if it conveys the same factual answer as the "
    "expected answer, even if the phrasing differs. Minor wording differences "
    "are acceptable.\n\n"
    "Explain your reasoning in 1-2 sentences, then provide a single-word "
    "LABEL on its own line: either 'correct' or 'incorrect'."
)

HALLUCINATION_TEMPLATE = (
    "You are evaluating whether an AI response contains hallucinated "
    "(fabricated or factually wrong) information.\n\n"
    "[BEGIN DATA]\n"
    "[Question]: {query}\n"
    "[Expected Answer]: {expected_output}\n"
    "[System Output]: {output}\n"
    "[END DATA]\n\n"
    "The output is 'factual' if all claims are accurate or consistent with "
    "the expected answer. The output is 'hallucinated' if it contains any "
    "made-up facts, wrong numbers, or fabricated details.\n\n"
    "Explain your reasoning in 1-2 sentences, then provide a single-word "
    "LABEL on its own line: either 'factual' or 'hallucinated'."
)

RELEVANCE_TEMPLATE = (
    "You are evaluating whether an AI response is relevant to the question.\n\n"
    "[BEGIN DATA]\n"
    "[Question]: {query}\n"
    "[System Output]: {output}\n"
    "[END DATA]\n\n"
    "The output is 'relevant' if it directly addresses the question, even if "
    "it includes extra detail. The output is 'irrelevant' if it does not "
    "attempt to answer the question or goes completely off-topic.\n\n"
    "Explain your reasoning in 1-2 sentences, then provide a single-word "
    "LABEL on its own line: either 'relevant' or 'irrelevant'."
)


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluator functions (arize-phoenix-evals >= 3.0.0)
# Requires OPENAI_API_KEY env var. Falls back to keyword-heuristic scoring
# when arize-phoenix-evals is unavailable or OPENAI_API_KEY is not set.
# ---------------------------------------------------------------------------
def _make_llm():
    """Construct a phoenix.evals 3.x LLM instance (OpenAI gpt-4o-mini)."""
    from phoenix.evals import LLM
    return LLM(provider="openai", model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])


def _qa_correctness(output: str, dataset_row: dict):
    """LLM-as-Judge correctness using arize-phoenix-evals 3.x create_classifier."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")

    try:
        from phoenix.evals import create_classifier
        evaluator = create_classifier(
            name="qa_correctness",
            prompt_template=QA_CORRECTNESS_TEMPLATE,
            llm=_make_llm(),
            choices={"correct": 1.0, "incorrect": 0.0},
        )
        scores = evaluator.evaluate({
            "query": dataset_row.get("query", ""),
            "expected_output": dataset_row.get("expected_output", ""),
            "output": output,
        })
        s = scores[0]
        return EvaluationResult(
            score=float(s.score) if s.score is not None else 0.0,
            label=s.label or "incorrect",
            explanation=s.explanation or "LLM-as-judge (qa_correctness)",
        )
    except Exception as exc:
        expected = str(dataset_row.get("expected_output", "")).strip().lower()
        actual = str(output).strip().lower()
        match = bool(expected and expected in actual)
        return EvaluationResult(
            score=1.0 if match else 0.0,
            label="correct" if match else "incorrect",
            explanation=f"Heuristic fallback ({exc.__class__.__name__}): expected in output={match}",
        )


def _hallucination_check(output: str, dataset_row: dict):
    """LLM-as-Judge hallucination using arize-phoenix-evals 3.x create_classifier."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")

    try:
        from phoenix.evals import create_classifier
        evaluator = create_classifier(
            name="hallucination",
            prompt_template=HALLUCINATION_TEMPLATE,
            llm=_make_llm(),
            choices={"factual": 1.0, "hallucinated": 0.0},
        )
        scores = evaluator.evaluate({
            "query": dataset_row.get("query", ""),
            "expected_output": dataset_row.get("expected_output", ""),
            "output": output,
        })
        s = scores[0]
        return EvaluationResult(
            score=float(s.score) if s.score is not None else 0.0,
            label=s.label or "hallucinated",
            explanation=s.explanation or "LLM-as-judge (hallucination)",
        )
    except Exception as exc:
        expected_words = {w.lower() for w in str(dataset_row.get("expected_output", "")).split() if len(w) > 4}
        output_words = {w.lower() for w in str(output).split() if len(w) > 4}
        overlap = expected_words & output_words
        factual = bool(overlap) or not expected_words
        return EvaluationResult(
            score=1.0 if factual else 0.0,
            label="factual" if factual else "hallucinated",
            explanation=f"Heuristic fallback ({exc.__class__.__name__}): keyword overlap={overlap or 'none'}",
        )


def _relevance_check(output: str, dataset_row: dict):
    """LLM-as-Judge relevance using arize-phoenix-evals 3.x create_classifier."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")

    try:
        from phoenix.evals import create_classifier
        evaluator = create_classifier(
            name="relevance",
            prompt_template=RELEVANCE_TEMPLATE,
            llm=_make_llm(),
            choices={"relevant": 1.0, "irrelevant": 0.0},
        )
        scores = evaluator.evaluate({
            "query": dataset_row.get("query", ""),
            "output": output,
        })
        s = scores[0]
        return EvaluationResult(
            score=float(s.score) if s.score is not None else 0.0,
            label=s.label or "irrelevant",
            explanation=s.explanation or "LLM-as-judge (relevance)",
        )
    except Exception as exc:
        query_words = {w.lower() for w in str(dataset_row.get("query", "")).split() if len(w) > 3}
        output_words = {w.lower() for w in str(output).split() if len(w) > 3}
        overlap = query_words & output_words
        relevant = bool(overlap)
        return EvaluationResult(
            score=1.0 if relevant else 0.0,
            label="relevant" if relevant else "irrelevant",
            explanation=f"Heuristic fallback ({exc.__class__.__name__}): query–output overlap={overlap or 'none'}",
        )


def _verify_experiment_results(**ctx) -> dict[str, Any]:
    """Summarise which experiment tasks produced non-None XCom values."""
    ti = ctx["ti"]
    task_ids = [f"exp_{cfg['name'].replace('-', '_')}" for cfg in EXPERIMENT_CONFIGS]
    task_ids += [
        "create_project", "create_dataset", "list_experiments",
        "create_prompt_concise", "create_prompt_detailed",
    ]
    checks = {tid: ti.xcom_pull(task_ids=tid) is not None for tid in task_ids}
    passed = sum(checks.values())
    print(f"Verification: {passed}/{len(checks)} tasks produced output")
    for tid, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {tid}")
    return {"passed": passed, "total": len(checks), "checks": checks}


def _cleanup_experiments(**ctx) -> None:
    """Delete all experiments for the dataset, then the dataset and project."""
    from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook

    hook = ArizeAxHook()
    ds_id = ctx["ti"].xcom_pull(task_ids="create_dataset")
    if not ds_id or str(ds_id) == "None":
        print("No dataset ID -- skipping experiment cleanup.")
        return

    try:
        result = hook.list_experiments(dataset_id=str(ds_id), limit=50)
        items = result.get("items", []) if isinstance(result, dict) else []
        for exp in items:
            exp_id = exp.get("id") if isinstance(exp, dict) else None
            if exp_id:
                try:
                    hook.delete_experiment(experiment_id=str(exp_id))
                    print(f"Deleted experiment {exp_id}")
                except Exception as exc:
                    print(f"Failed to delete experiment {exp_id} (non-fatal): {exc}")
    except Exception as exc:
        print(f"Failed to list experiments for cleanup (non-fatal): {exc}")


with DAG(
    dag_id="arize_ax_e2e_llm_experiment_comparison",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "e2e", "llm", "experiments"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Phase 1 -- Discovery
    list_spaces = ArizeAxListSpacesOperator(task_id="list_spaces", limit=10)

    # Phase 2 -- Resource setup
    create_project = ArizeAxCreateProjectOperator(
        task_id="create_project",
        space_id="{{ ti.xcom_pull(task_ids='list_spaces', key='first_id') }}",
        name="e2e-llm-{{ ts_nodash }}",
        if_exists="skip",
    )

    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id="{{ ti.xcom_pull(task_ids='list_spaces', key='first_id') }}",
        name="e2e-qa-{{ ts_nodash }}",
        examples=DATASET_EXAMPLES,
        if_exists="skip",
    )

    get_dataset = ArizeAxGetDatasetOperator(
        task_id="get_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
    )

    # Phase 3 -- Create prompts in Prompt Hub
    create_prompt_concise = ArizeAxCreatePromptOperator(
        task_id="create_prompt_concise",
        space_id="{{ ti.xcom_pull(task_ids='list_spaces', key='first_id') }}",
        name="e2e-concise-{{ ts_nodash }}",
        messages=[
            {"role": "system", "content": PROMPT_CONCISE},
            {"role": "user", "content": "{query}"},
        ],
        provider="open_ai",
        input_variable_format="f_string",
        model="gpt-4o-mini",
        commit_message="Concise prompt from e2e run {{ ts_nodash }}",
        description="Auto-generated concise prompt",
        if_exists="skip",
    )

    create_prompt_detailed = ArizeAxCreatePromptOperator(
        task_id="create_prompt_detailed",
        space_id="{{ ti.xcom_pull(task_ids='list_spaces', key='first_id') }}",
        name="e2e-detailed-{{ ts_nodash }}",
        messages=[
            {"role": "system", "content": PROMPT_DETAILED},
            {"role": "user", "content": "{query}"},
        ],
        provider="open_ai",
        input_variable_format="f_string",
        model="gpt-4o-mini",
        commit_message="Detailed prompt from e2e run {{ ts_nodash }}",
        description="Auto-generated detailed prompt",
        if_exists="skip",
    )

    # Phase 4 -- Sensor gate
    dataset_ready = ArizeAxDatasetReadySensor(
        task_id="dataset_ready_sensor",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        min_examples=len(DATASET_EXAMPLES),
        poke_interval=5,
        timeout=120,
        mode="poke",
    )

    # Phase 5 -- Run 5 experiments (parallel after dataset is ready)
    experiment_tasks = []
    for cfg in EXPERIMENT_CONFIGS:
        task_id = f"exp_{cfg['name'].replace('-', '_')}"
        exp_task = ArizeAxRunExperimentOperator(
            task_id=task_id,
            name=f"{cfg['name']}-{{{{ ts_nodash }}}}",
            dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
            task=_make_llm_task(cfg["provider"], cfg["model"], cfg["prompt"]),
            evaluators=[_qa_correctness, _hallucination_check, _relevance_check],
            concurrency=4,
        )
        experiment_tasks.append(exp_task)

    # Phase 5b -- List all experiments for the dataset
    list_experiments = ArizeAxListExperimentsOperator(
        task_id="list_experiments",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=20,
        trigger_rule="all_done",
    )

    # Phase 6 -- Verification + Inspection
    verify = PythonOperator(
        task_id="verify_results",
        python_callable=_verify_experiment_results,
        trigger_rule="all_done",
    )

    inspection_window = TimeDeltaSensor(
        task_id="inspection_window",
        delta=timedelta(minutes=INSPECTION_WINDOW_MINUTES),
        mode="reschedule",
        trigger_rule="all_done",
    )

    # Phase 7 -- Cleanup (always runs)
    cleanup_exps = PythonOperator(
        task_id="cleanup_experiments",
        python_callable=_cleanup_experiments,
        trigger_rule="all_done",
    )

    delete_prompt_concise = ArizeAxDeletePromptOperator(
        task_id="cleanup_prompt_concise",
        prompt_id="{{ ti.xcom_pull(task_ids='create_prompt_concise') }}",
        trigger_rule="all_done",
    )

    delete_prompt_detailed = ArizeAxDeletePromptOperator(
        task_id="cleanup_prompt_detailed",
        prompt_id="{{ ti.xcom_pull(task_ids='create_prompt_detailed') }}",
        trigger_rule="all_done",
    )

    cleanup_ds = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    cleanup_proj = ArizeAxDeleteProjectOperator(
        task_id="cleanup_project",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # Wiring

    # Phase 1 → 2
    list_spaces >> [create_project, create_dataset]
    create_dataset >> get_dataset

    # Phase 2 → 3 (create prompts after space_id is known)
    list_spaces >> [create_prompt_concise, create_prompt_detailed]

    # Phase 2 → 4 (sensor gate after dataset exists)
    get_dataset >> dataset_ready

    # Phase 4 → 5 (all experiments run in parallel after dataset is ready)
    for exp_task in experiment_tasks:
        dataset_ready >> exp_task

    # Phase 5 → 5b (list all experiments after they complete)
    experiment_tasks >> list_experiments

    # Phase 5 + extras → 6 (verify after everything)
    [list_experiments, create_prompt_concise, create_prompt_detailed, create_project] >> verify

    # Phase 6 → 6b → 7
    verify >> inspection_window
    inspection_window >> cleanup_exps >> [delete_prompt_concise, delete_prompt_detailed] >> cleanup_ds >> cleanup_proj
