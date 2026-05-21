"""Self-Optimizing Loop demo — fully self-contained closed-loop prompt
improvement: create a deliberately-broken starter prompt, run a baseline
experiment, optimize the prompt from baseline feedback via the Arize
Prompt Learning SDK, run a candidate experiment with the optimized
prompt, gate promotion on an LLM-as-judge metric, and promote on win.

The demo is designed to **reliably end green** when prerequisites are
configured: the verbose starter prompt produces multi-paragraph answers
that the terseness LLM judge rates low; the optimizer reliably reduces
the prompt to a concise form; the candidate scores high. If you change
the starter prompt or the dataset, the gate may fire and the DAG will
end red — that is correct behavior, not a bug.

Pipeline (12 stages)::

    check_prereqs               (ShortCircuit — OPENAI_API_KEY + space + SDK import)
        │
        ▼
    create_demo_dataset         (10 terse-answer trivia rows; if_exists="skip")
        │
        ▼
    create_initial_prompt       (verbose-by-design starter; if_exists="skip")
        │
        ▼
    run_baseline_experiment     (ArizeAxRunExperimentOperator, baseline_task + LLM judge)
        │
        ▼
    optimize_and_store          (PythonOperator — Prompt Learning SDK + Variable.set)
        │
        ▼
    push_optimized_prompt       (ArizeAxCreatePromptOperator if_exists="add_version" — pushes v2 under same name)
        │
        ▼
    run_candidate_experiment    (ArizeAxRunExperimentOperator, candidate_task + LLM judge)
        │
        ▼
    compare_experiments         (ArizeAxCompareExperimentsOperator, fail_on_regression=True)
        │
        ▼
    promote_prompt              (ArizeAxPromotePromptOperator, label="production")
        │
        ▼
    summarize_loop              (PythonOperator — print baseline/candidate/delta)
        │
        ▼
    cleanup_dataset             (ArizeAxDeleteDatasetOperator, trigger_rule="all_done",
                                 gated by arize_ax_self_optimizing_cleanup Variable)

Prerequisites
-------------
1. Airflow connection ``arize_ax_default`` with a valid API key.
2. Airflow Variable ``arize_ax_space_id`` (or ``default_space`` on the connection extras).
3. ``OPENAI_API_KEY`` in the **worker environment** — used both by the
   experiment tasks (worker-side LLM call) and by the Prompt Learning SDK
   during optimization.
4. ``prompt-learning-enhanced`` installed on the worker (separate install
   because PyPI rejects direct-URL deps, so the provider can no longer
   declare it as an extra)::

       pip install 'arize-phoenix-evals>=2.0,<3.0' \\
                   'prompt-learning-enhanced @ git+https://github.com/Arize-ai/prompt-learning.git'

5. ``openai`` Python SDK installed on the worker (typically already present).

Optional Variables
------------------
- ``arize_ax_self_optimizing_cleanup`` — set to ``"true"`` to delete the
  demo dataset on DAG completion. Default ``"false"`` so you can inspect
  the artifacts in the Arize UI between runs.
- ``arize_ax_self_optimizing_model`` — the OpenAI model used by the
  experiment tasks (default ``"gpt-4o-mini"``). The optimizer always
  uses ``gpt-4o`` regardless.

This demo runs **client-side experiments** via ``ArizeAxRunExperimentOperator``.
For a server-side (Eval Hub) variant, see ``example_arize_ax_e2e_dag.py``,
which uses the ``arize_ax_chained_experiment_eval`` TaskGroup helper to
trigger a ``run_experiment`` task with chained evaluation tasks via
``evaluation_task_ids``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxCompareExperimentsOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxPromotePromptOperator,
)
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_PROMPT_NAME = "arize-ax-self-optimizing-loop-demo"
_DATASET_NAME_TEMPLATE = "arize-ax-self-optimizing-loop-{{ ds_nodash }}"
_OPTIMIZED_MESSAGES_VAR = "arize_ax_self_optimizing_optimized_messages"

# Deliberately verbose starter — produces multi-paragraph answers that fail
# an exact-match evaluator over terse expected answers. The optimizer
# reliably reduces this to a concise form once it sees the failures.
_INITIAL_SYSTEM_PROMPT = (
    "You are a verbose, educational assistant. Always explain your "
    "reasoning in multiple paragraphs before giving the final answer. "
    "Include historical context and related facts. Your response should "
    "feel like a textbook entry."
)

# 10 trivia rows with one-word expected answers. Each row uses ``input`` and
# ``expected_output`` columns to match Arize experiment conventions.
_DEMO_EXAMPLES = [
    {"input": "What is the capital of France?", "expected_output": "Paris"},
    {"input": "Chemical symbol for water?", "expected_output": "H2O"},
    {"input": "Largest planet in the solar system?", "expected_output": "Jupiter"},
    {"input": "Who wrote Hamlet?", "expected_output": "Shakespeare"},
    {"input": "First president of the United States?", "expected_output": "Washington"},
    {"input": "How many continents are there?", "expected_output": "7"},
    {"input": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"input": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"input": "Speed of light in km/s (approx.)?", "expected_output": "300000"},
    {"input": "Year humans first landed on the Moon?", "expected_output": "1969"},
]


# ---------------------------------------------------------------------------
# Prereq guard
# ---------------------------------------------------------------------------
def _check_prereqs(**_ctx) -> bool:
    """Return True iff every prerequisite is satisfied; otherwise short-circuit."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY not set in worker environment — skipping demo.")
        return False
    if not Variable.get("arize_ax_space_id", default_var="").strip():
        print("Airflow Variable arize_ax_space_id not set — skipping demo.")
        return False
    try:
        import optimizer_sdk.prompt_learning_optimizer  # noqa: F401
    except ImportError:
        print(
            "prompt-learning-enhanced SDK is not installed on the worker. "
            "Install with: pip install 'arize-phoenix-evals>=2.0,<3.0' "
            "'prompt-learning-enhanced @ git+https://github.com/Arize-ai/prompt-learning.git'"
        )
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        print("openai SDK is not installed on the worker — required for experiment tasks.")
        return False
    return True


# ---------------------------------------------------------------------------
# Task callables for the experiments
# ---------------------------------------------------------------------------
def _call_openai_with_messages(messages: list[dict[str, str]]) -> str:
    """Single LLM call. Kept tiny so the demo is easy to read."""
    import openai

    model = Variable.get("arize_ax_self_optimizing_model", default_var="gpt-4o-mini")
    response = openai.OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def baseline_task(dataset_row: dict) -> dict[str, Any]:
    """Run the deliberately-verbose starter prompt against a dataset row."""
    messages = [
        {"role": "system", "content": _INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": dataset_row.get("input", "")},
    ]
    return {"output": _call_openai_with_messages(messages)}


def candidate_task(dataset_row: dict) -> dict[str, Any]:
    """Run the optimized prompt (stored in Airflow Variable) against a dataset row."""
    raw = Variable.get(_OPTIMIZED_MESSAGES_VAR, default_var=None)
    if not raw:
        raise RuntimeError(
            f"Variable {_OPTIMIZED_MESSAGES_VAR!r} is not set — "
            "the store_optimized_prompt stage must run before candidate_task."
        )
    optimized_messages = json.loads(raw)
    user_msg = {"role": "user", "content": dataset_row.get("input", "")}
    # Optimizer typically returns a single system message; append the user
    # question so the call has both roles.
    messages = list(optimized_messages) + [user_msg]
    return {"output": _call_openai_with_messages(messages)}


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator scoring an assistant's response to a "
    "trivia question. Score the response on BOTH correctness AND "
    "terseness. First count the words in the response, then score using "
    "the rubric below. Reply with ONLY a single line in the exact form: "
    "score=<float between 0.0 and 1.0>; label=<concise|brief|verbose|incorrect>; "
    "reason=<word count + short explanation>.\n\n"
    "Scoring rubric (response word count matters — count words carefully):\n"
    "  - score=1.0, label=concise: response is 1-30 words AND contains the "
    "expected answer. A short factual answer with at most one supporting "
    "sentence fits here.\n"
    "  - score=0.5, label=brief: response is 31-100 words AND contains the "
    "expected answer. A short paragraph with limited context fits here.\n"
    "  - score=0.0, label=verbose: response is MORE THAN 100 words "
    "(regardless of correctness). Multi-paragraph explanations, historical "
    "context tangents, or textbook-style answers fall here.\n"
    "  - score=0.0, label=incorrect: response does not contain the "
    "expected answer at all.\n\n"
    "Be strict about word count: count every word, do not round generously. "
    "The grade reflects format quality, not just factual accuracy."
)


def _parse_judge_response(text: str) -> tuple[float, str, str]:
    """Parse 'score=...; label=...; reason=...' into (score, label, reason)."""
    score = 0.0
    label = "incorrect"
    reason = text.strip()
    for part in text.split(";"):
        key, _, value = part.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key == "score":
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 0.0
        elif key == "label":
            label = value or label
        elif key == "reason":
            reason = value or reason
    score = max(0.0, min(1.0, score))
    return score, label, reason


def llm_judge_terseness(dataset_row: dict, output: Any) -> Any:
    """LLM-as-judge: scores response on correctness AND terseness via gpt-4o-mini.

    Cheap and fast (one extra LLM call per dataset row, gpt-4o-mini at
    temperature=0). The rubric narrows variance enough that the demo
    reliably ends green when the optimizer wins.
    """
    from arize.experiments import EvaluationResult

    expected = str(dataset_row.get("expected_output", "")).strip()
    question = str(dataset_row.get("input", "")).strip()
    actual = str((output or {}).get("output", "") if isinstance(output, dict) else (output or "")).strip()

    user_msg = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Assistant response: {actual}"
    )
    raw = _call_openai_with_messages(
        [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    score, label, reason = _parse_judge_response(raw)
    return EvaluationResult(score=score, label=label, explanation=reason)


# ---------------------------------------------------------------------------
# Optimization step — runs ArizeAxOptimizePromptOperator under the hood and
# persists the result in an Airflow Variable so candidate_task can pick it up
# without an Airflow context (Arize SDK task callables only receive the
# dataset row).
# ---------------------------------------------------------------------------
def _optimize_and_store(**ctx) -> dict[str, Any]:
    """Optimize the starter prompt against baseline feedback and store the result.

    The Prompt Learning SDK's ``PromptLearningOptimizer.optimize`` requires a
    pandas DataFrame (it calls ``dataset.columns`` during input validation),
    so we materialize the baseline's row-dict records into one here.
    """
    import pandas as pd
    from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook

    baseline_result = ctx["ti"].xcom_pull(task_ids="run_baseline_experiment", key="result") or {}
    if not isinstance(baseline_result, dict):
        raise RuntimeError(
            f"Unexpected baseline result XCom shape: {type(baseline_result).__name__}."
        )
    records = baseline_result.get("dataframe_records") or []
    if not records:
        raise RuntimeError(
            "Baseline experiment produced no dataframe_records — cannot optimize."
        )

    df = pd.DataFrame(records)
    print(f"[optimize] Baseline DataFrame: {len(df)} rows, columns={list(df.columns)}")

    hook = ArizeAxHook()
    optimization = hook.optimize_prompt(
        prompt=_INITIAL_SYSTEM_PROMPT,
        dataset=df,
        output_column="output",
        feedback_columns=[
            "eval.llm_judge_terseness.label",
            "eval.llm_judge_terseness.explanation",
        ],
        model_choice="gpt-4o",
    )

    messages = optimization.get("messages") or []
    if not messages:
        raise RuntimeError(
            "Prompt Learning SDK returned no optimized messages — "
            f"raw output: {optimization!r}"
        )

    Variable.set(_OPTIMIZED_MESSAGES_VAR, json.dumps(messages))
    print(f"[optimize] Optimized system prompt → {messages[0].get('content', '')[:200]!r}…")
    return optimization


# ---------------------------------------------------------------------------
# Summary callable
# ---------------------------------------------------------------------------
def _summarize_loop(**ctx) -> dict[str, Any]:
    """Log baseline / candidate / delta and a Prompt Hub pointer."""
    baseline_id = ctx["ti"].xcom_pull(task_ids="run_baseline_experiment")
    candidate_id = ctx["ti"].xcom_pull(task_ids="run_candidate_experiment")
    compare = ctx["ti"].xcom_pull(task_ids="compare_experiments") or {}
    promoted_version_id = ctx["ti"].xcom_pull(task_ids="promote_prompt")
    print("=" * 60)
    print("SELF-OPTIMIZING LOOP COMPLETE")
    print(f"  Baseline experiment id : {baseline_id}")
    print(f"  Candidate experiment id: {candidate_id}")
    print(f"  Comparison verdict     : {compare}")
    print(f"  Prompt name            : {_PROMPT_NAME}")
    print(f"  Promoted version id    : {promoted_version_id}")
    print("  Next step              : inspect the prompt in Arize Prompt Hub.")
    print("=" * 60)
    return {
        "baseline_experiment_id": baseline_id,
        "candidate_experiment_id": candidate_id,
        "compare": compare,
        "prompt_name": _PROMPT_NAME,
        "prompt_version_id": promoted_version_id,
    }


def _should_cleanup(**_ctx) -> bool:
    return Variable.get("arize_ax_self_optimizing_cleanup", default_var="false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="arize_ax_self_optimizing_loop",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "demo", "prompt", "optimization", "self-learning"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    create_demo_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_demo_dataset",
        space_id=_SPACE_JINJA,
        name=_DATASET_NAME_TEMPLATE,
        examples=_DEMO_EXAMPLES,
        if_exists="skip",
    )

    create_initial_prompt = ArizeAxCreatePromptOperator(
        task_id="create_initial_prompt",
        space_id=_SPACE_JINJA,
        name=_PROMPT_NAME,
        messages=[{"role": "system", "content": _INITIAL_SYSTEM_PROMPT}],
        model="gpt-4o-mini",
        commit_message="initial verbose starter prompt (deliberately broken for terse Q&A)",
        if_exists="skip",
    )

    run_baseline_experiment = ArizeAxRunExperimentOperator(
        task_id="run_baseline_experiment",
        name="self-optimizing-baseline-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        task=baseline_task,
        evaluators=[llm_judge_terseness],
        concurrency=4,
    )

    optimize_and_store = PythonOperator(
        task_id="optimize_and_store",
        python_callable=_optimize_and_store,
    )

    # Reuse ArizeAxCreatePromptOperator with ``if_exists="add_version"``:
    # when the prompt name already exists, the operator catches the 409,
    # resolves the existing prompt_id, and pushes the new messages as a
    # fresh version (visible in Prompt Hub as v2 alongside the original).
    push_optimized_prompt = ArizeAxCreatePromptOperator(
        task_id="push_optimized_prompt",
        space_id=_SPACE_JINJA,
        name=_PROMPT_NAME,
        messages_task_id="optimize_and_store",
        messages_key="messages",
        model="gpt-4o-mini",
        commit_message="optimized via Prompt Learning SDK from baseline feedback ({{ ts_nodash }})",
        if_exists="add_version",
    )

    run_candidate_experiment = ArizeAxRunExperimentOperator(
        task_id="run_candidate_experiment",
        name="self-optimizing-candidate-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        task=candidate_task,
        evaluators=[llm_judge_terseness],
        concurrency=4,
    )

    compare_experiments = ArizeAxCompareExperimentsOperator(
        task_id="compare_experiments",
        candidate_experiment_id="{{ ti.xcom_pull(task_ids='run_candidate_experiment') }}",
        baseline_experiment_id="{{ ti.xcom_pull(task_ids='run_baseline_experiment') }}",
        metric_names=["llm_judge_terseness"],
        pass_threshold=0.0,
        fail_on_regression=True,
    )

    promote_prompt = ArizeAxPromotePromptOperator(
        task_id="promote_prompt",
        prompt_name=_PROMPT_NAME,
        label="production",
        space_id=_SPACE_JINJA,
    )

    summarize_loop = PythonOperator(
        task_id="summarize_loop",
        python_callable=_summarize_loop,
    )

    should_cleanup = ShortCircuitOperator(
        task_id="should_cleanup",
        python_callable=_should_cleanup,
        trigger_rule="all_done",
    )

    cleanup_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        ignore_if_missing=True,
    )

    # Wiring
    check_prereqs >> create_demo_dataset >> create_initial_prompt
    create_initial_prompt >> run_baseline_experiment >> optimize_and_store
    optimize_and_store >> push_optimized_prompt >> run_candidate_experiment
    run_candidate_experiment >> compare_experiments >> promote_prompt
    promote_prompt >> summarize_loop >> should_cleanup >> cleanup_dataset
