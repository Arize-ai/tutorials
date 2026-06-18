"""
LLM CI/CD Gate DAG (server-side, self-contained): automated evaluation-gated
promotion for LLM changes.

The DAG creates both a *baseline* and a *candidate* experiment server-side
(Arize Eval Hub drives the LLM calls and the accuracy judge), scores both,
and raises ``AirflowException`` if the candidate doesn't beat the baseline
by the configured threshold. On pass it optionally promotes a prompt.

It is fully self-contained — no pre-existing baseline experiment, prompt,
or dataset is required. The DAG provisions everything it needs on each
run, demonstrates the gate, then cleans up the ephemeral resources.

Pipeline stages
---------------
0. **check_prereqs** — short-circuit if ``ARIZE_AI_INTEGRATION_ID`` isn't
   resolvable (an OpenAI integration is needed for both the experiment LLM
   calls and the judge).
1. **create_eval_dataset** — provision a small evaluation dataset.
2. **build_accuracy_judge_config / create_accuracy_evaluator** — build the
   accuracy LLM-as-judge config and create an evaluator with a per-run-unique
   name (avoids Arize's soft-delete name-tombstone trap).
3. **create_accuracy_eval_task** — wrap the evaluator in a
   ``template_evaluation`` task scoped to the configured project.
4. **build_baseline_run_config / create_baseline_run_exp_task** — register a
   ``run_experiment`` task driven by a *weaker* config (gpt-4o-mini + terse
   system prompt) to act as the baseline.
5. **run_baseline_experiment.{trigger,wait,get_result}** — fire the baseline
   experiment with the accuracy judge chained server-side.
6. **build_candidate_run_config / create_candidate_run_exp_task** — register
   a ``run_experiment`` task driven by a *stronger* config (gpt-4.1 +
   careful system prompt).
7. **run_candidate_experiment.{trigger,wait,get_result}** — fire the
   candidate experiment with the same chained accuracy judge.
8. **score_baseline / score_candidate** — aggregate evaluation scores from
   both experiments (``ArizeAxGetExperimentScoreOperator``).
9. **compare_experiments** — pass/fail verdict with
   ``fail_on_regression=True``.
10. **check_prompt_configured** — short-circuit promotion if no
    ``arize_ax_prompt_name`` Variable is set (gate-only mode).
11. **promote_prompt** — tag the configured prompt with the
    ``arize_ax_prompt_label`` label.
12. **promote_notification** — log the promotion outcome.
13. **cleanup_*** — delete per-run-ephemeral evaluator, eval task, the two
    run-experiment tasks, and the dataset.

Variables
---------
- ``arize_ax_space_id`` — Arize space ID (required).
- ``arize_ax_project_id`` — project ID used to scope the accuracy eval task
  (required by the Eval Hub for ``template_evaluation`` tasks).
- ``arize_ai_integration_id`` *or* env var ``ARIZE_AI_INTEGRATION_ID`` —
  OpenAI integration in Arize with gpt-4.1 + gpt-4o-mini access.
- ``arize_ax_prompt_name`` — optional. When set, the DAG promotes that
  prompt version on gate-pass.
- ``arize_ax_prompt_label`` — label to apply (default ``"production"``).

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
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
    ArizeAxCompareExperimentsOperator,
    ArizeAxGetExperimentScoreOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxCreatePromptOperator,
    ArizeAxDeletePromptOperator,
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

CICD_PASS_THRESHOLD = 0.0  # candidate score must be >= baseline + threshold

CICD_EVAL_EXAMPLES = [
    {"query": "What is the capital of France?", "expected_output": "Paris"},
    {"query": "What is 7 multiplied by 8?", "expected_output": "56"},
    {"query": "Who wrote Hamlet?", "expected_output": "William Shakespeare"},
    {"query": "What element has atomic number 1?", "expected_output": "Hydrogen"},
    {"query": "How many continents are there?", "expected_output": "7"},
]

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_RUN_SUFFIX = (
    "{{ run_id | replace(':', '') | replace('-', '') | replace('+', '') "
    "| replace('.', '') | replace('_', '') }}"
)

ACCURACY_TEMPLATE = (
    "You are evaluating a question-answering system.\n\n"
    "[Question]: {query}\n"
    "[Expected Answer]: {expected_output}\n"
    "[System Output]: {output}\n\n"
    "The system output is correct if it conveys the same factual answer as "
    "the expected answer, even if phrased differently. Reply with ONLY "
    "'correct' or 'incorrect'."
)

BASELINE_SYSTEM_PROMPT = "Answer."  # deliberately terse → expected to underperform
CANDIDATE_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer the question with only "
    "the direct factual answer — no extra explanation or punctuation."
)


def _resolve_integration_id() -> str:
    return (
        Variable.get("arize_ai_integration_id", default_var="").strip()
        or Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()
    )


def _check_prereqs(**_ctx) -> bool:
    """Short-circuit when the AI integration isn't configured."""
    if not _resolve_integration_id():
        print(
            "ARIZE_AI_INTEGRATION_ID env var or arize_ai_integration_id "
            "Variable must be set to a valid Arize OpenAI integration ID."
        )
        return False
    return True


def _build_accuracy_judge_config(**_ctx) -> dict[str, Any]:
    return {
        "name": "cicd_accuracy_judge",
        "template": ACCURACY_TEMPLATE,
        "include_explanations": True,
        "use_function_calling_if_available": False,
        "classification_choices": {"correct": 1.0, "incorrect": 0.0},
        "llm_config": {
            "ai_integration_id": _resolve_integration_id(),
            "model_name": "gpt-4o-mini",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        },
    }


def _build_run_config(model: str, system_prompt: str):
    def _callable(**_ctx) -> dict[str, Any]:
        return {
            "experiment_type": "llm_generation",
            "ai_integration_id": _resolve_integration_id(),
            "model_name": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                # run_experiment messages use mustache substitution.
                {"role": "user", "content": "{{query}}"},
            ],
            "input_variable_format": "mustache",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        }

    _callable.__name__ = f"_build_run_config_{model.replace('-', '_').replace('.', '_')}"
    return _callable


def _promote_notification(**ctx) -> dict[str, Any]:
    """Log the promotion outcome."""
    ti = ctx["ti"]
    trigger_result = ti.xcom_pull(
        task_ids="run_candidate_experiment.trigger", key="result"
    ) or {}
    candidate_id = (
        trigger_result.get("experiment_id") if isinstance(trigger_result, dict) else None
    )
    scores = ti.xcom_pull(task_ids="score_candidate") or {}
    promote_result = ti.xcom_pull(task_ids="promote_prompt") or {}
    promoted_version_id = (
        promote_result.get("prompt_id") if isinstance(promote_result, dict) else None
    )
    prompt_label = (
        promote_result.get("label") if isinstance(promote_result, dict) else "production"
    )
    prompt_name = "cicd-demo-prompt-<run_suffix>"
    print("=" * 60)
    print("CANDIDATE PROMOTED")
    print(f"  Experiment ID : {candidate_id}")
    print(f"  Scores        : {scores}")
    print(f"  Prompt name   : {prompt_name}")
    print(f"  Prompt label  : {prompt_label}")
    print(f"  Prompt verId  : {promoted_version_id}")
    print("=" * 60)
    return {
        "promoted": True,
        "experiment_id": candidate_id,
        "scores": scores,
        "prompt_name": prompt_name,
        "prompt_label": prompt_label,
        "prompt_version_id": promoted_version_id,
    }


_BASELINE_EXP_ID = (
    "{{ ti.xcom_pull(task_ids='run_baseline_experiment.trigger', "
    "key='result')['experiment_id'] }}"
)
_CANDIDATE_EXP_ID = (
    "{{ ti.xcom_pull(task_ids='run_candidate_experiment.trigger', "
    "key='result')['experiment_id'] }}"
)


with DAG(
    dag_id="arize_ax_llm_cicd_gate",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    tags=["arize_ax", "cicd", "llm", "experiments", "eval-hub", "self-contained"],
    catchup=False,
    doc_md=__doc__,
    render_template_as_native_obj=True,
) as dag:

    check_prereqs = ShortCircuitOperator(
        task_id="check_prereqs",
        python_callable=_check_prereqs,
    )

    # ----- Phase 1: ephemeral dataset --------------------------------------
    create_eval_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_eval_dataset",
        space_id=_SPACE_JINJA,
        name=f"cicd-eval-dataset-{_RUN_SUFFIX}",
        examples=CICD_EVAL_EXAMPLES,
        if_exists="skip",
    )

    # ----- Phase 2: accuracy judge ------------------------------------------
    build_accuracy_judge_config = PythonOperator(
        task_id="build_accuracy_judge_config",
        python_callable=_build_accuracy_judge_config,
    )
    create_accuracy_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_accuracy_evaluator",
        space_id=_SPACE_JINJA,
        name=f"cicd_accuracy_judge_{_RUN_SUFFIX}",
        evaluator_type="template",
        commit_message="initial cicd accuracy judge",
        template_config_task_id="build_accuracy_judge_config",
        description="LLM-as-judge for CI/CD gate factual correctness.",
    )
    create_accuracy_eval_task = ArizeAxCreateTaskOperator(
        task_id="create_accuracy_eval_task",
        name=f"cicd-accuracy-task-{_RUN_SUFFIX}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_accuracy_evaluator') }}"},
        ],
        project_id="{{ var.value.get('arize_ax_project_id', '') or None }}",
        is_continuous=False,
        sampling_rate=1.0,
    )

    # ----- Phase 3: baseline experiment (weaker config) --------------------
    # Baseline + candidate both use gpt-4.1 (proven stable in earlier
    # end-to-end testing); the comparison isolates prompt-engineering
    # impact rather than model quality.
    build_baseline_run_config = PythonOperator(
        task_id="build_baseline_run_config",
        python_callable=_build_run_config("gpt-4.1", BASELINE_SYSTEM_PROMPT),
    )
    create_baseline_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_baseline_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"cicd-baseline-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_baseline_run_config') }}",
        if_exists="skip",
    )
    run_baseline_experiment = arize_ax_chained_experiment_eval(
        group_id="run_baseline_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        experiment_name=f"cicd-baseline-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    # ----- Phase 4: candidate experiment (stronger config) ------------------
    build_candidate_run_config = PythonOperator(
        task_id="build_candidate_run_config",
        python_callable=_build_run_config("gpt-4.1", CANDIDATE_SYSTEM_PROMPT),
    )
    create_candidate_run_exp_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_candidate_run_exp_task",
        space_id=_SPACE_JINJA,
        name=f"cicd-candidate-task-{_RUN_SUFFIX}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_candidate_run_config') }}",
        if_exists="skip",
    )
    run_candidate_experiment = arize_ax_chained_experiment_eval(
        group_id="run_candidate_experiment",
        task_id_param="{{ ti.xcom_pull(task_ids='create_candidate_run_exp_task') }}",
        experiment_name=f"cicd-candidate-{_RUN_SUFFIX}",
        space_id=_SPACE_JINJA,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ],
        wait_for_completion=True,
        sensor_timeout=900,
        sensor_poke_interval=15,
        fail_on_run_error=True,
    )

    # ----- Phase 5: score + compare + gate ----------------------------------
    score_baseline = ArizeAxGetExperimentScoreOperator(
        task_id="score_baseline",
        experiment_id=_BASELINE_EXP_ID,
        aggregation="mean",
    )
    score_candidate = ArizeAxGetExperimentScoreOperator(
        task_id="score_candidate",
        experiment_id=_CANDIDATE_EXP_ID,
        aggregation="mean",
    )
    compare_experiments = ArizeAxCompareExperimentsOperator(
        task_id="compare_experiments",
        candidate_experiment_id=_CANDIDATE_EXP_ID,
        baseline_experiment_id=_BASELINE_EXP_ID,
        pass_threshold=CICD_PASS_THRESHOLD,
        aggregation="mean",
        fail_on_regression=True,
    )

    # ----- Phase 6: promotion (fully self-contained) ------------------------
    # Create an ephemeral demo prompt so the promotion path always exercises
    # end-to-end without needing any external Variable.
    create_demo_prompt = ArizeAxCreatePromptOperator(
        task_id="create_demo_prompt",
        space_id=_SPACE_JINJA,
        name=f"cicd-demo-prompt-{_RUN_SUFFIX}",
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": "{query}"},
        ],
        model="gpt-4.1",
        commit_message="initial cicd-demo prompt version",
        input_variable_format="f_string",
    )
    promote_prompt = ArizeAxPromotePromptOperator(
        task_id="promote_prompt",
        prompt_name=f"cicd-demo-prompt-{_RUN_SUFFIX}",
        label="production",
        space_id=_SPACE_JINJA,
    )
    promote_notification = PythonOperator(
        task_id="promote_notification",
        python_callable=_promote_notification,
    )

    # ----- Phase 7: cleanup -------------------------------------------------
    cleanup_accuracy_eval_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_accuracy_eval_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_accuracy_eval_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_baseline_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_baseline_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_baseline_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_candidate_run_exp_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_candidate_run_exp_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_candidate_run_exp_task') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_accuracy_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_accuracy_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_accuracy_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_eval_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    cleanup_demo_prompt = ArizeAxDeletePromptOperator(
        task_id="cleanup_demo_prompt",
        prompt_id="{{ ti.xcom_pull(task_ids='create_demo_prompt') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # Wiring — baseline and candidate experiments run *serially*, not in
    # parallel. We saw the Arize backend silently stall one of two
    # concurrent run_experiment triggers (status stuck at "running" with
    # zero success/error counters), so we wait for the baseline chain to
    # terminate before kicking off the candidate.
    check_prereqs >> create_eval_dataset
    check_prereqs >> [build_accuracy_judge_config, build_baseline_run_config, build_candidate_run_config]
    build_accuracy_judge_config >> create_accuracy_evaluator >> create_accuracy_eval_task
    [create_eval_dataset, build_baseline_run_config] >> create_baseline_run_exp_task
    [create_eval_dataset, build_candidate_run_config] >> create_candidate_run_exp_task
    [create_baseline_run_exp_task, create_accuracy_eval_task] >> run_baseline_experiment
    run_baseline_experiment >> score_baseline
    # Candidate trigger blocks on baseline terminal — avoid concurrent runs.
    [run_baseline_experiment, create_candidate_run_exp_task, create_accuracy_eval_task] >> run_candidate_experiment
    run_candidate_experiment >> score_candidate
    [score_baseline, score_candidate] >> compare_experiments
    # Self-contained promotion path: create_demo_prompt happens up-front so
    # the prompt always exists when promote_prompt fires.
    check_prereqs >> create_demo_prompt
    compare_experiments >> promote_prompt
    create_demo_prompt >> promote_prompt >> promote_notification
    # Cleanup runs unconditionally (trigger_rule="all_done").
    promote_notification >> [
        cleanup_accuracy_eval_task,
        cleanup_baseline_run_exp_task,
        cleanup_candidate_run_exp_task,
        cleanup_accuracy_evaluator,
        cleanup_dataset,
        cleanup_demo_prompt,
    ]
