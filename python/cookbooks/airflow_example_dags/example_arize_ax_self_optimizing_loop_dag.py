"""Self-Optimizing Loop demo — fully self-contained closed-loop prompt
improvement, **executed server-side via the Arize Eval Hub**: create a
deliberately-broken starter prompt, run a baseline experiment server-side,
optimize the prompt from baseline feedback via the Arize Prompt Learning
SDK, run a candidate experiment with the optimized prompt server-side,
gate promotion on an LLM-as-judge metric, and promote on win.

The demo is designed to **reliably end green** when prerequisites are
configured: the verbose starter prompt produces multi-paragraph answers
that the terseness LLM judge rates low; the optimizer reliably reduces
the prompt to a concise form; the candidate scores high. If you change
the starter prompt or the dataset, the gate may fire and the DAG will
end red — that is correct behavior, not a bug.

Pipeline (server-side variant)::

    check_prereqs                   (ShortCircuit — AI integration + space + SDK)
        │
        ▼
    create_demo_dataset             (10 terse-answer trivia rows; if_exists="skip")
        │
        ▼
    create_initial_prompt           (verbose-by-design starter; if_exists="skip")
        │
        ▼
    create_terseness_evaluator      (server-side template_evaluation evaluator)
        │
        ▼
    create_terseness_eval_task      (template_evaluation task wrapping the evaluator)
        │
        ▼
    create_baseline_run_exp_task    (run_experiment task with v1 prompt baked in)
        │
        ▼
    baseline.{trigger,wait,get_result}   (TaskGroup — server-side chained eval)
        │
        ▼
    fetch_baseline_records          (PythonOperator — list_experiment_runs into df)
        │
        ▼
    optimize_and_store              (Prompt Learning SDK + Variable.set)
        │
        ▼
    push_optimized_prompt           (ArizeAxCreatePromptOperator if_exists="add_version")
        │
        ▼
    create_candidate_run_exp_task   (run_experiment task with v2 prompt baked in; dynamic)
        │
        ▼
    candidate.{trigger,wait,get_result}  (TaskGroup — server-side chained eval)
        │
        ▼
    compare_experiments             (gate on terseness metric; fail_on_regression=True)
        │
        ▼
    promote_prompt                  (label="production")
        │
        ▼
    summarize_loop                  (print baseline/candidate/delta)
        │
        ▼
    should_cleanup → cleanup_*      (delete dataset + ephemeral tasks/evaluator)

Prerequisites
-------------
1. Airflow connection ``arize_ax_default`` with a valid API key.
2. Airflow Variable ``arize_ax_space_id`` (or ``default_space`` on the connection extras).
3. Airflow Variable ``arize_ai_integration_id`` (or env-var
   ``ARIZE_AI_INTEGRATION_ID``) pointing at an AI integration with a real
   provider API key. Used for **both** the run_experiment LLM call AND
   the chained terseness evaluator. Server-side execution means there's
   no ``OPENAI_API_KEY`` on the Airflow worker.
4. ``prompt-learning-enhanced`` installed on the worker (the optimizer
   step still runs locally; only the experiments + eval go server-side)::

       pip install 'arize-phoenix-evals>=2.0,<3.0' \\
                   'prompt-learning-enhanced @ git+https://github.com/Arize-ai/prompt-learning.git'

Optional Variables
------------------
- ``arize_ax_self_optimizing_cleanup`` — set to ``"true"`` to delete the
  demo dataset + ephemeral task resources on DAG completion. Default
  ``"false"`` so you can inspect the artifacts in the Arize UI between
  runs.
- ``arize_ax_self_optimizing_model`` — the model used by the server-side
  experiment tasks (default ``"gpt-4o-mini"``). The optimizer always
  uses ``gpt-4o`` regardless.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxCompareExperimentsOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxPromotePromptOperator,
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

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"
_INTEGRATION_ID_JINJA = (
    "{{ var.value.get('arize_ai_integration_id', '') "
    "or var.value.get('ARIZE_AI_INTEGRATION_ID', '') }}"
)

_PROMPT_NAME = "arize-ax-self-optimizing-loop-demo"
_DATASET_NAME_TEMPLATE = "arize-ax-self-optimizing-loop-{{ ds_nodash }}"
_OPTIMIZED_MESSAGES_VAR = "arize_ax_self_optimizing_optimized_messages"
_EVAL_NAME = "terseness_judge"  # column name on experiment runs / metric name for compare

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

# Terseness judge rubric, encoded as a server-side template_evaluation
# template. The evaluator template uses **f-string single-brace**
# substitution (Arize backend validates this at create time). Note that
# the `run_experiment` task's `messages` field uses **mustache
# double-brace** — the two surfaces have different substitution rules.
_TERSENESS_JUDGE_TEMPLATE = (
    "You are a strict evaluator scoring an assistant's response to a "
    "trivia question. Score the response on BOTH correctness AND terseness.\n\n"
    "Question: {input}\n"
    "Expected answer: {expected_output}\n"
    "Assistant response: {output}\n\n"
    "First count the words in the assistant response, then choose a label "
    "using the rubric below.\n\n"
    "Rubric (response word count matters — count carefully):\n"
    "  - concise: 1-30 words AND contains the expected answer.\n"
    "  - brief: 31-100 words AND contains the expected answer.\n"
    "  - verbose: more than 100 words (regardless of correctness).\n"
    "  - incorrect: response does not contain the expected answer at all.\n\n"
    "Be strict about word count. Reply with ONLY the label."
)


def _check_prereqs(**_ctx) -> bool:
    """Return True iff every prerequisite is satisfied; otherwise short-circuit."""
    if not Variable.get("arize_ax_space_id", default_var="").strip():
        print("Airflow Variable arize_ax_space_id not set — skipping demo.")
        return False
    integration = (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )
    if not integration:
        print(
            "AI integration Variable not set — set 'arize_ai_integration_id' or "
            "'ARIZE_AI_INTEGRATION_ID' to point at an integration with a real "
            "provider API key."
        )
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
    return True


def _build_evaluator_template_config(**ctx) -> dict[str, Any]:
    """Build the terseness-judge template_config — referenced by create_terseness_evaluator."""
    integration_id = (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )
    return {
        "name": _EVAL_NAME,
        "template": _TERSENESS_JUDGE_TEMPLATE,
        "include_explanations": True,
        "use_function_calling_if_available": False,
        "classification_choices": {
            "concise": 1.0,
            "brief": 0.5,
            "verbose": 0.0,
            "incorrect": 0.0,
        },
        "llm_config": {
            "ai_integration_id": integration_id,
            "model_name": Variable.get(
                "arize_ax_self_optimizing_model", default_var="gpt-4o-mini",
            ),
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        },
    }


def _run_configuration_for_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Shared helper: build a LlmGenerationRunConfig dict around a messages list."""
    integration_id = (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )
    return {
        "experiment_type": "llm_generation",
        "ai_integration_id": integration_id,
        "model_name": Variable.get(
            "arize_ax_self_optimizing_model", default_var="gpt-4o-mini",
        ),
        # Mustache + double-brace: matches Arize backend substitution.
        "messages": messages,
        "input_variable_format": "mustache",
        "invocation_parameters": {"temperature": 0},
        "provider_parameters": {},
    }


def _build_baseline_run_config(**_ctx) -> dict[str, Any]:
    """run_configuration for the baseline experiment (verbose-by-design starter)."""
    return _run_configuration_for_messages(
        [
            {"role": "system", "content": _INITIAL_SYSTEM_PROMPT},
            {"role": "user", "content": "{{input}}"},
        ],
    )


def _build_candidate_run_config(**_ctx) -> dict[str, Any]:
    """run_configuration for the candidate experiment using the optimized prompt.

    Reads optimized messages from the Variable stored by ``optimize_and_store``;
    appends a user-role line that pulls the dataset's ``input`` column via
    mustache substitution.
    """
    raw = Variable.get(_OPTIMIZED_MESSAGES_VAR, default_var=None)
    if not raw:
        raise RuntimeError(
            f"Variable {_OPTIMIZED_MESSAGES_VAR!r} not set — "
            "optimize_and_store must run before this step."
        )
    optimized_messages = json.loads(raw)
    return _run_configuration_for_messages(
        list(optimized_messages) + [{"role": "user", "content": "{{input}}"}],
    )


def _records_from_experiment_runs(experiment_id: str) -> list[dict[str, Any]]:
    """Flatten experiment runs (server-side) into a list of dicts the
    Prompt Learning optimizer can ingest.

    For each run, combines the dataset-example fields (input,
    expected_output) with the LLM output and chained-eval columns
    (label, explanation). The optimizer reads `feedback_columns` from
    the resulting DataFrame.
    """
    hook = ArizeAxHook()
    # Brief grace period for eval scores to attach to runs after the
    # experiment terminates (the chained eval task fires async on the
    # backend; helper's wait_for_completion gates on task-run status
    # but not on eval score persistence).
    time.sleep(15)
    response = hook.list_experiment_runs(experiment_id=experiment_id, limit=50, all=True)
    items = response.get("items", []) if isinstance(response, dict) else []
    records: list[dict[str, Any]] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        ap = r.get("additional_properties") or {}
        ap = ap if isinstance(ap, dict) else {}
        rec: dict[str, Any] = {
            "output": r.get("output"),
            # Dataset example fields land in additional_properties under
            # their original column names on server-side runs.
            "input": ap.get("input"),
            "expected_output": ap.get("expected_output"),
        }
        # Eval columns from the chained template_evaluation task:
        # eval.<name>.label / .score / .explanation.
        for key in ap.keys():
            if key.startswith("eval."):
                rec[key] = ap[key]
        records.append(rec)
    return records


def _optimize_and_store(**ctx) -> dict[str, Any]:
    """Optimize the starter prompt against server-side baseline feedback."""
    import pandas as pd

    # Fetch the chained get_result XCom (TaskGroup .get_result task pushes
    # the get_task_run dict; experiment_id lives on its sibling .trigger
    # XCom under the 'result' key).
    ti = ctx["ti"]
    triggered = ti.xcom_pull(
        task_ids="run_baseline_experiment.trigger", key="result",
    ) or {}
    experiment_id = triggered.get("experiment_id") if isinstance(triggered, dict) else None
    if not experiment_id:
        raise RuntimeError(
            "Could not resolve baseline experiment_id from "
            "run_baseline_experiment.trigger XCom."
        )

    records = _records_from_experiment_runs(experiment_id)
    if not records:
        raise RuntimeError(
            f"Baseline experiment {experiment_id} returned no runs — "
            "cannot optimize."
        )
    df = pd.DataFrame(records)
    print(f"[optimize] Baseline DataFrame: {len(df)} rows, columns={list(df.columns)}")

    hook = ArizeAxHook()
    optimization = hook.optimize_prompt(
        prompt=_INITIAL_SYSTEM_PROMPT,
        dataset=df,
        output_column="output",
        feedback_columns=[
            f"eval.{_EVAL_NAME}.label",
            f"eval.{_EVAL_NAME}.explanation",
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


def _summarize_loop(**ctx) -> dict[str, Any]:
    """Log baseline / candidate / delta and a Prompt Hub pointer."""
    ti = ctx["ti"]
    baseline_triggered = ti.xcom_pull(
        task_ids="run_baseline_experiment.trigger", key="result",
    ) or {}
    candidate_triggered = ti.xcom_pull(
        task_ids="run_candidate_experiment.trigger", key="result",
    ) or {}
    baseline_id = baseline_triggered.get("experiment_id") if isinstance(baseline_triggered, dict) else None
    candidate_id = candidate_triggered.get("experiment_id") if isinstance(candidate_triggered, dict) else None
    compare = ti.xcom_pull(task_ids="compare_experiments") or {}
    promoted_version_id = ti.xcom_pull(task_ids="promote_prompt")
    print("=" * 60)
    print("SELF-OPTIMIZING LOOP COMPLETE (server-side)")
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
    return Variable.get(
        "arize_ax_self_optimizing_cleanup", default_var="false",
    ).strip().lower() == "true"


with DAG(
    dag_id="arize_ax_self_optimizing_loop",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "demo", "prompt", "optimization", "self-learning", "eval-hub"],
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

    build_evaluator_config = PythonOperator(
        task_id="build_evaluator_config",
        python_callable=_build_evaluator_template_config,
    )

    create_terseness_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_terseness_evaluator",
        space_id=_SPACE_JINJA,
        name=_EVAL_NAME,
        evaluator_type="template",
        commit_message="initial terseness judge for self-optimizing-loop demo",
        template_config_task_id="build_evaluator_config",
        description="Word-count + correctness rubric for trivia-style answers.",
        if_exists="skip",
    )

    create_terseness_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_terseness_eval_task",
        name="self-optimizing-terseness-task-{{ ts_nodash }}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_terseness_evaluator') }}"},
        ],
        # Project-scoped: server-side eval runs against the run's spans.
        # We point at the experiment_traces_project_id created by the
        # baseline run_experiment task at trigger time (via the chained
        # path); for a project-scoped template_eval the project_id is
        # required at creation. The baseline run_experiment creates its
        # own ephemeral project; we use the configured arize_ax_project_id
        # Variable as the scoping target so the eval task can be created
        # before the experiments fire.
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    build_baseline_run_config = PythonOperator(
        task_id="build_baseline_run_config",
        python_callable=_build_baseline_run_config,
    )

    create_baseline_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_baseline_run_exp_task",
        space_id=_SPACE_JINJA,
        name="self-optimizing-baseline-task-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_baseline_run_config') }}",
        if_exists="skip",
    )

    # Baseline server-side experiment + chained terseness eval.
    run_baseline_experiment = arize_ax_chained_experiment_eval(
        group_id="run_baseline_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        experiment_name="self-optimizing-baseline-{{ ts_nodash }}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_terseness_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    optimize_and_store = PythonOperator(
        task_id="optimize_and_store",
        python_callable=_optimize_and_store,
    )

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

    build_candidate_run_config = PythonOperator(
        task_id="build_candidate_run_config",
        python_callable=_build_candidate_run_config,
    )

    create_candidate_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_candidate_run_exp_task",
        space_id=_SPACE_JINJA,
        name="self-optimizing-candidate-task-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_demo_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_candidate_run_config') }}",
        if_exists="skip",
    )

    run_candidate_experiment = arize_ax_chained_experiment_eval(
        group_id="run_candidate_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_candidate_run_exp_task') }}",
        experiment_name="self-optimizing-candidate-{{ ts_nodash }}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_terseness_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    compare_experiments = ArizeAxCompareExperimentsOperator(
        task_id="compare_experiments",
        candidate_experiment_id=(
            "{{ ti.xcom_pull(task_ids='run_candidate_experiment.trigger', "
            "key='result')['experiment_id'] }}"
        ),
        baseline_experiment_id=(
            "{{ ti.xcom_pull(task_ids='run_baseline_experiment.trigger', "
            "key='result')['experiment_id'] }}"
        ),
        metric_names=[_EVAL_NAME],
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
    cleanup_baseline_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_baseline_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        ignore_if_missing=True,
    )
    cleanup_candidate_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_candidate_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_candidate_run_exp_task') }}",
        ignore_if_missing=True,
    )
    cleanup_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_terseness_eval_task') }}",
        ignore_if_missing=True,
    )
    cleanup_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_terseness_evaluator') }}",
        ignore_if_missing=True,
    )

    # Wiring
    check_prereqs >> create_demo_dataset >> create_initial_prompt
    create_initial_prompt >> build_evaluator_config >> create_terseness_evaluator
    create_terseness_evaluator >> create_terseness_eval_task
    [create_demo_dataset, build_baseline_run_config] >> create_baseline_run_exp_task
    [create_terseness_eval_task, create_baseline_run_exp_task] >> run_baseline_experiment
    run_baseline_experiment >> optimize_and_store >> push_optimized_prompt
    push_optimized_prompt >> build_candidate_run_config >> create_candidate_run_exp_task
    [create_terseness_eval_task, create_candidate_run_exp_task] >> run_candidate_experiment
    run_candidate_experiment >> compare_experiments >> promote_prompt
    promote_prompt >> summarize_loop
    summarize_loop >> should_cleanup
    should_cleanup >> [
        cleanup_dataset,
        cleanup_baseline_task,
        cleanup_candidate_task,
        cleanup_eval_task,
        cleanup_evaluator,
    ]
