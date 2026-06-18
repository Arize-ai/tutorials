"""
Prompt A/B Testing DAG: compare N prompts against an evaluation dataset and
promote the winner to the "production" label in Arize Prompt Hub.

Pipeline stages
---------------
1. **create_eval_dataset** — create (or skip if exists) the evaluation dataset.
2. **append_eval_examples** — load evaluation examples into the dataset.
3. **compare_prompts** — run one experiment per prompt and rank by mean eval
   score (ArizeAxComparePromptsOperator).
4. **extract_winner** — pull the winning prompt name from compare_prompts XCom.
5. **gate_has_winner** — short-circuit if no winner was determined.
6. **promote_winner** — tag the winning prompt version with label="production"
   (ArizeAxPromotePromptOperator).
7. **log_results** — log the full rankings summary.

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_dataset_id`` — dataset ID; if absent a new dataset is created.
- ``arize_ax_prompt_names`` — prompt names to compare. Accepts either a
  comma-separated list (``"prompt-a,prompt-b"``) or a JSON array
  (``'["prompt-a","prompt-b"]'``). Default:
  ``["prompt-v1", "prompt-v2", "prompt-v3"]``.

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - At least two prompts already created in Arize Prompt Hub with the names
    listed in ``arize_ax_prompt_names``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxAppendDatasetExamplesOperator,
    ArizeAxCreateDatasetOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxComparePromptsOperator,
    ArizeAxPromotePromptOperator,
)
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

AB_DATASET_NAME = "prompt-ab-eval-dataset"

# Sample evaluation examples used to score each prompt.
AB_EVAL_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"query": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
    {"query": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"query": "How many continents are there?", "expected_output": "7"},
]

# Default prompt names compared when Variable is not set.
_DEFAULT_PROMPT_NAMES = ["prompt-v1", "prompt-v2", "prompt-v3"]


def _resolve_prompt_names() -> list[str]:
    """Read the ``arize_ax_prompt_names`` Variable, accepting JSON or CSV."""
    raw = Variable.get("arize_ax_prompt_names", default_var=None)
    if not raw or not str(raw).strip():
        return _DEFAULT_PROMPT_NAMES
    s = str(raw).strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                names = [str(n).strip() for n in parsed if str(n).strip()]
                if names:
                    return names
        except json.JSONDecodeError:
            pass
    names = [n.strip() for n in s.split(",") if n.strip()]
    return names or _DEFAULT_PROMPT_NAMES


def _ab_task(dataset_row: dict) -> dict:
    """Minimal task that echoes expected output to demonstrate the pipeline.

    Replace with a real LLM call in production, e.g.::

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": dataset_row["query"]}],
        )
        return {"output": response.choices[0].message.content}
    """
    return {"output": dataset_row.get("expected_output", "")}


def _ab_evaluator(dataset_row: dict, output: Any):
    """Accuracy evaluator: 1.0 if output matches expected, else 0.0."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="no output")
    actual = output.get("output", "") if isinstance(output, dict) else str(output)
    expected = dataset_row.get("expected_output", "")
    match = str(actual).strip().lower() == str(expected).strip().lower()
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={expected!r}, got={actual!r}",
    )


def _extract_winner(**ctx) -> str | None:
    """Pull the winning prompt name from compare_prompts XCom."""
    result = ctx["ti"].xcom_pull(task_ids="compare_prompts")
    if not isinstance(result, dict):
        print(f"[extract_winner] unexpected XCom type: {type(result).__name__!r}")
        return None
    winner = result.get("winner")
    rankings = result.get("rankings", [])
    print(f"[extract_winner] winner={winner!r}")
    for entry in rankings:
        print(
            f"  rank={entry.get('rank')}  prompt={entry.get('prompt_name')!r}"
            f"  mean_score={entry.get('mean_score')}"
        )
    return winner


def _gate_has_winner(**ctx) -> bool:
    """Return True iff a clear winner was produced."""
    winner = ctx["ti"].xcom_pull(task_ids="extract_winner")
    if winner:
        print(f"[gate] winner={winner!r} -- proceeding to promotion.")
        return True
    print("[gate] no winner determined -- short-circuiting.")
    return False


def _log_results(**ctx) -> dict[str, Any]:
    """Log the A/B test summary."""
    result = ctx["ti"].xcom_pull(task_ids="compare_prompts") or {}
    winner = result.get("winner")
    rankings = result.get("rankings", [])
    print("=" * 60)
    print("PROMPT A/B TEST COMPLETE")
    print(f"  Winner: {winner!r}")
    for entry in rankings:
        print(
            f"  rank={entry.get('rank')}  {entry.get('prompt_name')!r}"
            f"  mean_score={entry.get('mean_score'):.4f}"
            f"  exp_id={entry.get('experiment_id')}"
        )
    print("=" * 60)
    return {"winner": winner, "rankings": rankings}


with DAG(
    dag_id="arize_ax_prompt_ab_test",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "prompts", "ab_test", "experiments"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # Stage 1 — Dataset setup
    create_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_eval_dataset",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        name=AB_DATASET_NAME,
        # SDK 8.25+ rejects examples=[] with a client-side ValueError before
        # the API is called, so if_exists="skip" can't intercept it. Seed
        # with a single placeholder example; real examples are added by the
        # subsequent append step.
        examples=[{"input": "_seed_", "expected": "_seed_"}],
        if_exists="skip",
    )

    append_eval_examples = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_eval_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        examples=AB_EVAL_EXAMPLES,
    )

    # Stage 2 — Compare prompts
    compare_prompts = ArizeAxComparePromptsOperator(
        task_id="compare_prompts",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        prompt_names=_resolve_prompt_names(),
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        task=_ab_task,
        evaluators=[_ab_evaluator],
        experiment_name_prefix="prompt-ab",
        concurrency=2,
    )

    # Stage 3 — Extract winner and gate
    extract_winner = PythonOperator(
        task_id="extract_winner",
        python_callable=_extract_winner,
    )

    gate_has_winner = ShortCircuitOperator(
        task_id="gate_has_winner",
        python_callable=_gate_has_winner,
    )

    # Stage 4 — Promote winning prompt
    promote_winner = ArizeAxPromotePromptOperator(
        task_id="promote_winner",
        space_id="{{ var.value.get('arize_ax_space_id', None) }}",
        prompt_name="{{ ti.xcom_pull(task_ids='extract_winner') }}",
        label="production",
    )

    # Stage 5 — Log results
    log_results = PythonOperator(
        task_id="log_results",
        python_callable=_log_results,
    )

    # Wiring
    create_eval_dataset >> append_eval_examples >> compare_prompts
    compare_prompts >> extract_winner >> gate_has_winner >> promote_winner >> log_results
