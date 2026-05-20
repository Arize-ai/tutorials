"""
End-to-end test DAG: exercises every Arize AX operator and sensor that can be
exercised in a single self-contained run.

Each DAG run creates its own isolated resources using ``{{ ts_nodash }}`` as a
unique suffix, so consecutive runs never collide. All create operators use
``if_exists="skip"`` so retries within the same run are safe.

After verification, a configurable inspection window (default 15 minutes)
pauses the DAG so you can inspect resources on the Arize dashboard.
Set ``INSPECTION_WINDOW_MINUTES = 0`` to skip the pause for CI runs.

Cleanup tasks at the end remove every resource created during the run. They
use ``trigger_rule="all_done"`` so cleanup always runs, even on partial failure.

Optional Airflow Variables (phases that depend on them short-circuit cleanly):
  - ``arize_ai_integration_id``: enables Phase 5 (LLM evaluator lifecycle),
    Phase 6's comparison sub-phase, and Phase 7 (tasks lifecycle). These
    phases need a real (working) AI integration since the evaluator/task
    issues live LLM calls. The dummy integration created in Phase 2 is for
    CRUD coverage only.
  - ``arize_annotator_email``: enables Phase 8's annotation-queue sub-phase.
    Without it the annotation-config CRUD still runs.

Operators exercised (~70):
  Discovery (9): ListSpaces, ListAIIntegrations, ListProjects, ListDatasets,
    ListEvaluators, ListTasks, ListAnnotationConfigs, ListAnnotationQueues,
    ListAPIKeys
  AI Integration CRUD (4): CreateAIIntegration (dummy), GetAIIntegration,
    UpdateAIIntegration, DeleteAIIntegration
  API Key CRUD (3): CreateAPIKey, RefreshAPIKey, DeleteAPIKey
  Datasets (10): CreateDataset, GetDataset, AppendDatasetExamples,
    ListDatasetExamples, ExportDatasetExamplesToFile, EvalDatasetHealth,
    SmartDatasetRefresh, AnnotateDatasetExamples, DeleteDataset, GetProject
  Evaluators (LLM, gated, 7): CreateEvaluator, GetEvaluator, UpdateEvaluator,
    AddEvaluatorVersion, ListEvaluatorVersions, GetEvaluatorVersion,
    DeleteEvaluator
  Experiments (12): RunExperiment, GetExperiment, ListExperiments,
    ListExperimentRuns, GetExperimentScore, CompareExperiments,
    DetectEvalDrift, EvaluatorCalibration (LLM-gated), BehavioralRegression,
    EvalBudgetAllocator, ExportExperimentRunsToFile, AnnotateExperimentRuns,
    DeleteExperiment
  Tasks (LLM, gated, 9): CreateTask, GetTask, ListTaskRuns,
    TriggerTaskRun, GetTaskRun, CancelTaskRun, UpdateTask, DeleteTask,
    CreateRunExperimentTask
  Annotations (11): CreateAnnotationConfig, DeleteAnnotationConfig,
    CreateAnnotationQueue (gated), GetAnnotationQueue, UpdateAnnotationQueue,
    AddAnnotationQueueRecords, ListAnnotationQueueRecords, AnnotateQueueRecord,
    AssignQueueRecord, DeleteAnnotationQueueRecords, DeleteAnnotationQueue
  Prompts (6): CreatePrompt, GetPrompt, ListPrompts, ComparePrompts (gated),
    PromotePrompt, DeletePrompt
  Spans (1, alpha): ListSpans

Sensors exercised (7):
  DatasetReadySensor, ExperimentCompleteSensor, ExperimentRunCountSensor,
  EvaluationScoreSensor, SpanCountSensor, SpanIngestionSensor,
  AnnotationQueueSensor, TaskRunSensor (gated)

Not covered (need real production span/ML state or admin-level mutations):
  Span log/update/export operators (need OpenInference DataFrames with platform
    schema and real span IDs),
  Span curation/export/metrics/adaptive sampling (need a project with real
    production spans),
  ML log/export operators (need ML DataFrames + schemas),
  Spaces Create/Update/Delete (would mutate Arize tenancy — admin only;
    ListSpaces/GetSpace are exercised),
  CreateExperimentOperator (its experiment_runs param needs runtime
    example IDs; RunExperiment covers the equivalent end-to-end path).

Requires:
  - Airflow connection ``arize_ax_default`` with a valid API key.
  - Connection extra must include ``space_id``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook
from airflow.providers.arize_ax.operators.ai_integrations import (
    ArizeAxCreateAIIntegrationOperator,
    ArizeAxDeleteAIIntegrationOperator,
    ArizeAxGetAIIntegrationOperator,
    ArizeAxListAIIntegrationsOperator,
    ArizeAxUpdateAIIntegrationOperator,
)
from airflow.providers.arize_ax.operators.annotations import (
    ArizeAxAddAnnotationQueueRecordsOperator,
    ArizeAxAnnotateQueueRecordOperator,
    ArizeAxAssignQueueRecordOperator,
    ArizeAxCreateAnnotationConfigOperator,
    ArizeAxCreateAnnotationQueueOperator,
    ArizeAxDeleteAnnotationConfigOperator,
    ArizeAxDeleteAnnotationQueueOperator,
    ArizeAxDeleteAnnotationQueueRecordsOperator,
    ArizeAxGetAnnotationQueueOperator,
    ArizeAxListAnnotationConfigsOperator,
    ArizeAxListAnnotationQueueRecordsOperator,
    ArizeAxListAnnotationQueuesOperator,
    ArizeAxUpdateAnnotationQueueOperator,
)
from airflow.providers.arize_ax.operators.api_keys import (
    ArizeAxCreateAPIKeyOperator,
    ArizeAxDeleteAPIKeyOperator,
    ArizeAxListAPIKeysOperator,
    ArizeAxRefreshAPIKeyOperator,
)
from airflow.providers.arize_ax.operators.datasets import (
    ArizeAxAnnotateDatasetExamplesOperator,
    ArizeAxAppendDatasetExamplesOperator,
    ArizeAxCreateDatasetOperator,
    ArizeAxDeleteDatasetOperator,
    ArizeAxEvalDatasetHealthOperator,
    ArizeAxExportDatasetExamplesToFileOperator,
    ArizeAxGetDatasetOperator,
    ArizeAxListDatasetExamplesOperator,
    ArizeAxListDatasetsOperator,
    ArizeAxSmartDatasetRefreshOperator,
)
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxAddEvaluatorVersionOperator,
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
    ArizeAxGetEvaluatorOperator,
    ArizeAxGetEvaluatorVersionOperator,
    ArizeAxListEvaluatorsOperator,
    ArizeAxListEvaluatorVersionsOperator,
    ArizeAxUpdateEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.experiments import (
    ArizeAxAnnotateExperimentRunsOperator,
    ArizeAxBehavioralRegressionOperator,
    ArizeAxCompareExperimentsOperator,
    ArizeAxDeleteExperimentOperator,
    ArizeAxDetectEvalDriftOperator,
    ArizeAxEvalBudgetAllocatorOperator,
    ArizeAxEvaluatorCalibrationOperator,
    ArizeAxExportExperimentRunsToFileOperator,
    ArizeAxGetExperimentOperator,
    ArizeAxGetExperimentScoreOperator,
    ArizeAxListExperimentRunsOperator,
    ArizeAxListExperimentsOperator,
    ArizeAxRunExperimentOperator,
)
from airflow.providers.arize_ax.operators.projects import (
    ArizeAxCreateProjectOperator,
    ArizeAxDeleteProjectOperator,
    ArizeAxGetProjectOperator,
    ArizeAxListProjectsOperator,
)
from airflow.providers.arize_ax.operators.prompts import (
    ArizeAxComparePromptsOperator,
    ArizeAxCreatePromptOperator,
    ArizeAxDeletePromptOperator,
    ArizeAxGetPromptOperator,
    ArizeAxListPromptsOperator,
    ArizeAxPromotePromptOperator,
)
from airflow.providers.arize_ax.operators.spaces import (
    ArizeAxListSpacesOperator,
)
from airflow.providers.arize_ax.operators.spans import (
    ArizeAxListSpansOperator,
)
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxCancelTaskRunOperator,
    ArizeAxCreateRunExperimentTaskOperator,
    ArizeAxCreateTaskOperator,
    ArizeAxDeleteTaskOperator,
    ArizeAxGetTaskOperator,
    ArizeAxGetTaskRunOperator,
    ArizeAxListTaskRunsOperator,
    ArizeAxListTasksOperator,
    ArizeAxTriggerTaskRunOperator,
    ArizeAxUpdateTaskOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import (
    ArizeAxAnnotationQueueSensor,
    ArizeAxDatasetReadySensor,
    ArizeAxEvaluationScoreSensor,
    ArizeAxExperimentCompleteSensor,
    ArizeAxExperimentRunCountSensor,
    ArizeAxSpanCountSensor,
    ArizeAxSpanIngestionSensor,
    ArizeAxTaskRunSensor,
)
from airflow.providers.arize_ax.utils.task_groups import (
    arize_ax_chained_experiment_eval,
)
from airflow.providers.standard.operators.python import (
    PythonOperator,
    ShortCircuitOperator,
)


def _echo_task(dataset_row: dict) -> dict:
    """Trivial task that echoes the expected output."""
    return {"output": dataset_row.get("expected_output", "no-answer")}


def _exact_match_evaluator(dataset_row: dict, output: dict):
    """Evaluator: strict equality between output and expected answer."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")
    actual = output.get("output") if isinstance(output, dict) else str(output)
    expected = dataset_row.get("expected_output", "")
    match = actual == expected
    return EvaluationResult(
        score=1.0 if match else 0.0,
        label="correct" if match else "incorrect",
        explanation=f"expected={expected!r}, got={actual!r}",
    )


def _contains_answer_evaluator(dataset_row: dict, output: dict):
    """Evaluator: lenient check -- expected answer appears within output."""
    from arize.experiments import EvaluationResult

    if output is None:
        return EvaluationResult(score=0.0, label="error", explanation="task produced no output")
    actual = output.get("output", "") if isinstance(output, dict) else str(output)
    expected = dataset_row.get("expected_output", "")
    found = expected.lower() in actual.lower()
    return EvaluationResult(
        score=1.0 if found else 0.0,
        label="contains" if found else "missing",
        explanation=f"looking for {expected!r} in {actual!r}",
    )


def _get_first_var(*names: str) -> str:
    """Return the first non-empty Airflow Variable from *names* (case variants)."""
    from airflow.models import Variable

    for name in names:
        try:
            val = Variable.get(name, default_var="")
        except Exception:
            val = ""
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _has_ai_integration_var(**_ctx) -> bool:
    """Short-circuit predicate: True if the AI-integration Variable is set.

    Accepts either ``arize_ai_integration_id`` (lowercase — preferred to match
    the other ``arize_ax_*`` variables) or ``ARIZE_AI_INTEGRATION_ID``
    (uppercase — matches the env-var name used by the system test suite).
    """
    val = _get_first_var("arize_ai_integration_id", "ARIZE_AI_INTEGRATION_ID")
    if not val:
        print(
            "[gate] Neither 'arize_ai_integration_id' nor "
            "'ARIZE_AI_INTEGRATION_ID' is set — downstream LLM phases "
            "(evaluators, tasks, LLM-evaluator comparison) will be skipped."
        )
        return False
    return True


def _has_annotator_email_var(**_ctx) -> bool:
    """Short-circuit predicate: True if the annotator-email Variable is set."""
    val = _get_first_var("arize_annotator_email", "ARIZE_ANNOTATOR_EMAIL")
    if not val:
        print(
            "[gate] Neither 'arize_annotator_email' nor "
            "'ARIZE_ANNOTATOR_EMAIL' is set — annotation-queue sub-phase "
            "will be skipped (config CRUD still runs)."
        )
        return False
    return True


def _make_xcom_present_gate(*, task_ids: str, key: str | None = None) -> Any:
    """Return a callable that short-circuits when an upstream XCom is missing.

    The Airflow caveat this guards against: when a task is skipped (e.g. by an
    earlier ShortCircuit gate) its XCom is None. Downstream tasks with
    ``trigger_rule="all_done"`` will still run, and Jinja renders None as the
    literal string ``"None"``, which most Arize endpoints then reject with a
    confusing 400 "Invalid ... ID format" or even 500. Use this guard between
    upstream creates/lists and downstream consumers that must not run on
    placeholder IDs.
    """
    def _gate(**ctx: Any) -> bool:
        ti = ctx["ti"]
        val = (
            ti.xcom_pull(task_ids=task_ids, key=key)
            if key is not None
            else ti.xcom_pull(task_ids=task_ids)
        )
        if val is None:
            print(
                f"[gate] no XCom from task_ids={task_ids!r} key={key!r}; "
                "skipping downstream consumers."
            )
            return False
        if isinstance(val, str) and (not val or val.strip().lower() in {"none", ""}):
            print(
                f"[gate] XCom from {task_ids!r} resolved to placeholder "
                f"{val!r}; skipping downstream consumers."
            )
            return False
        return True
    return _gate


def _build_run_configuration(integration_id: str | None = None, **ctx) -> dict[str, Any]:
    """Build a run configuration dict for CreateRunExperimentTask.

    Resolution order for the AI integration:
      1. The ``arize_ai_integration_id`` Airflow Variable (preferred). This
         is expected to point at an integration with a REAL API key so the
         LLM-generation step can actually call the provider.
      2. Fallback: the freshly-created ``create_ai_integration`` integration
         passed via ``op_kwargs``. **Note**: the DAG creates that fresh
         integration with a *dummy* API key (``sk-e2e-dummy-do-not-use``)
         to exercise the create/get/delete lifecycle without leaking real
         credentials, so a run_experiment using it will fail at the LLM
         call. Only used here when the Variable isn't set.

    Returns the LlmGenerationRunConfig-shaped dict directly. The hook layer
    calls ``RunConfiguration.from_dict`` for the typed wrapping internally
    (SDK 8.27+), so user code doesn't need to import the typed model.
    """
    var_integration_id = _get_first_var(
        "arize_ai_integration_id", "ARIZE_AI_INTEGRATION_ID",
    )
    if var_integration_id:
        integration_id = var_integration_id
    if not integration_id:
        raise RuntimeError(
            "build_run_config: an AI-integration ID is required. Either let "
            "the DAG create one via 'create_ai_integration' (it is wired up "
            "automatically), or set the 'arize_ai_integration_id' Airflow "
            "Variable."
        )
    # Use mustache + double-brace substitution for the run_experiment
    # template — produces real LLM output (single-brace / f_string can
    # leave the placeholder unsubstituted depending on backend handling).
    return {
        "experiment_type": "llm_generation",
        "ai_integration_id": integration_id,
        "model_name": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Echo: {{query}}"}],
        "input_variable_format": "mustache",
        "invocation_parameters": {"temperature": 0},
        "provider_parameters": {},
    }


def _build_evaluator_template_config(**ctx) -> dict[str, Any]:
    """Build a template-evaluator config using the configured AI integration."""
    integration_id = _get_first_var(
        "arize_ai_integration_id", "ARIZE_AI_INTEGRATION_ID",
    )
    if not integration_id:
        raise RuntimeError(
            "AI-integration Variable required for evaluator "
            "(set 'arize_ai_integration_id' or 'ARIZE_AI_INTEGRATION_ID')."
        )
    suffix = (ctx.get("ts_nodash") or "x")[-8:]
    return {
        "name": f"e2e_eval_{suffix}",
        # Template variables must match the experiment run's input column
        # names. This DAG's dataset uses ``query`` (the dataset's input
        # column) — referencing ``{input}`` here would fail substitution
        # with "Task was unable to handle template variable: input".
        # ``{output}`` is the LLM's output and is always available.
        "template": (
            "Evaluate the response. Input: {query}\nOutput: {output}\n"
            "Respond with JSON label correct or incorrect."
        ),
        "include_explanations": False,
        "use_function_calling_if_available": False,
        "classification_choices": {"incorrect": 0, "correct": 1},
        "llm_config": {
            "ai_integration_id": integration_id,
            "model_name": "gpt-4o-mini",
            "invocation_parameters": {"temperature": 0},
            "provider_parameters": {},
        },
    }


# Task IDs whose XCom values _verify_results checks. Exposed as a module-level
# constant so unit tests can derive expected counts without hardcoding numbers.
VERIFY_TASK_IDS: list[str] = [
    # Phase 1 -- discovery
    "list_spaces",
    "list_ai_integrations",
    "list_annotation_configs",
    "list_annotation_queues",
    "list_api_keys",
    "list_evaluators",
    "list_tasks",
    # Phase 2-3 -- AI integration + API key CRUD
    "create_ai_integration",
    "get_ai_integration",
    "create_api_key",
    # Phase 4 -- project + dataset
    "create_project",
    "get_project",
    "create_dataset",
    "get_dataset",
    "append_examples",
    "list_dataset_examples",
    "export_dataset_examples_to_file",
    "eval_dataset_health",
    # Phase 6 -- experiment + scoring
    "run_experiment",
    "get_experiment",
    "list_experiment_runs",
    "get_experiment_score",
    "compare_experiments",
    "detect_eval_drift",
    "behavioral_regression",
    "eval_budget_allocator",
    "export_experiment_runs_to_file",
    # Phase 8 -- annotations
    "create_annotation_config",
    # Phase 9 -- prompts
    "create_prompt",
    "get_prompt",
    # Phase 10 -- spans
    "list_spans",
]


def _verify_results(**ctx) -> dict[str, Any]:
    """Summarise which tasks produced non-None XCom values."""
    ti = ctx["ti"]
    checks = {tid: ti.xcom_pull(task_ids=tid) is not None for tid in VERIFY_TASK_IDS}
    passed = sum(checks.values())
    return {"passed": passed, "total": len(checks), "checks": checks}


def _inspection_pause(**ctx) -> None:
    """Sleep for the configured number of minutes so the user can inspect
    resources on the platform before cleanup deletes them."""
    minutes = ctx["params"].get("inspection_minutes", INSPECTION_WINDOW_MINUTES)
    minutes = max(int(minutes), 0)
    if minutes == 0:
        print("Inspection window is 0 minutes -- skipping pause.")
        return
    print(
        f"Pausing {minutes} minutes for manual inspection. "
        f"Resources will be cleaned up after this task completes."
    )
    time.sleep(minutes * 60)
    print("Inspection window finished -- proceeding to cleanup.")


INSPECTION_WINDOW_MINUTES = 15
# Prefer the user-configured ``arize_ax_space_id`` Airflow Variable so the
# e2e DAG operates in the space the operator was wired against (where the
# user's AI integration, datasets, projects, etc. live). Fall back to the
# first space returned by ``list_spaces`` only if the Variable is unset —
# convenient for first-time setup but historically a footgun, since
# ``list_spaces.first_id`` can resolve to a different space than the one
# the user has configured elsewhere, leading to cross-space "AI integration
# not found" / "no spans" surprises.
SPACE_ID = (
    "{{ var.value.get('arize_ax_space_id', '') "
    "or ti.xcom_pull(task_ids='list_spaces', key='first_id') }}"
)


with DAG(
    dag_id="arize_ax_e2e_full_provider_test",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    tags=["arize_ax", "e2e", "test"],
    catchup=False,
    doc_md=__doc__,
    params={"inspection_minutes": INSPECTION_WINDOW_MINUTES},
    # render_template_as_native_obj makes Jinja XCom pulls return native Python
    # types (dict/list/etc.) instead of stringified versions. Required for
    # operators that take dict params (e.g. CreateRunExperimentTask's
    # run_configuration) where the dict is built by an upstream PythonOperator.
    render_template_as_native_obj=True,
    # Many tasks here hit alpha/beta Arize APIs that occasionally return
    # transient backend errors (Flight internal errors on experiment init,
    # 5xx blips, etc.). One retry with a short delay handles the bulk of those
    # without burying real test signal under noise.
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=15),
    },
) as dag:

    # Phase 1 -- Discovery (read-only listings)
    list_spaces = ArizeAxListSpacesOperator(task_id="list_spaces", limit=10)
    list_ai_integrations = ArizeAxListAIIntegrationsOperator(
        task_id="list_ai_integrations", space_id=SPACE_ID, limit=10,
    )
    list_annotation_configs = ArizeAxListAnnotationConfigsOperator(
        task_id="list_annotation_configs", limit=10,
    )
    list_annotation_queues = ArizeAxListAnnotationQueuesOperator(
        task_id="list_annotation_queues", space=SPACE_ID, limit=10,
    )
    list_api_keys = ArizeAxListAPIKeysOperator(
        task_id="list_api_keys", limit=10,
    )
    list_evaluators = ArizeAxListEvaluatorsOperator(
        task_id="list_evaluators", space_id=SPACE_ID, limit=10,
    )
    list_tasks = ArizeAxListTasksOperator(
        task_id="list_tasks", space_id=SPACE_ID, limit=10,
    )

    # Phase 2 -- AI Integration CRUD (dummy key, fully self-contained)
    create_ai_integration = ArizeAxCreateAIIntegrationOperator(
        task_id="create_ai_integration",
        space_id=SPACE_ID,
        name="e2e-ai-integration-{{ ts_nodash }}",
        # API expects exact SDK enum value casing: "openAI", not "openai".
        provider="openAI",
        api_key="sk-e2e-dummy-do-not-use",
        # API rejects integrations with zero models; enable defaults so the
        # integration is well-formed without hardcoding model names here.
        enable_default_models=True,
        if_exists="skip",
    )
    get_ai_integration = ArizeAxGetAIIntegrationOperator(
        task_id="get_ai_integration",
        integration_id="{{ ti.xcom_pull(task_ids='create_ai_integration') }}",
    )
    update_ai_integration = ArizeAxUpdateAIIntegrationOperator(
        task_id="update_ai_integration",
        integration_id="{{ ti.xcom_pull(task_ids='create_ai_integration') }}",
        space_id=SPACE_ID,
        api_key="sk-e2e-dummy-rotated",
    )

    # Phase 3 -- API Key CRUD
    create_api_key = ArizeAxCreateAPIKeyOperator(
        task_id="create_api_key",
        name="e2e-api-key-{{ ts_nodash }}",
        description="Created by arize_ax_e2e_full_provider_test DAG; safe to delete.",
    )
    refresh_api_key = ArizeAxRefreshAPIKeyOperator(
        task_id="refresh_api_key",
        api_key_id="{{ ti.xcom_pull(task_ids='create_api_key') }}",
    )

    # Phase 4 -- Project + dataset setup + enrichment
    create_project = ArizeAxCreateProjectOperator(
        task_id="create_project",
        space_id=SPACE_ID,
        name="e2e-proj-{{ ts_nodash }}",
        if_exists="skip",
    )
    get_project = ArizeAxGetProjectOperator(
        task_id="get_project",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
    )
    list_projects = ArizeAxListProjectsOperator(
        task_id="list_projects", space_id=SPACE_ID, limit=5,
    )

    create_dataset = ArizeAxCreateDatasetOperator(
        task_id="create_dataset",
        space_id=SPACE_ID,
        name="e2e-ds-{{ ts_nodash }}",
        examples=[
            {"query": "What is 2+2?", "expected_output": "4"},
            {"query": "Capital of France?", "expected_output": "Paris"},
            {"query": "Largest planet?", "expected_output": "Jupiter"},
        ],
        if_exists="skip",
    )
    get_dataset = ArizeAxGetDatasetOperator(
        task_id="get_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
    )
    list_datasets = ArizeAxListDatasetsOperator(
        task_id="list_datasets", space_id=SPACE_ID, limit=5,
    )
    append_examples = ArizeAxAppendDatasetExamplesOperator(
        task_id="append_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        examples=[
            {"query": "Boiling point of water?", "expected_output": "100C"},
            {"query": "Speed of light?", "expected_output": "299792458 m/s"},
        ],
    )
    list_dataset_examples = ArizeAxListDatasetExamplesOperator(
        task_id="list_dataset_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=50,
    )
    export_dataset_examples_to_file = ArizeAxExportDatasetExamplesToFileOperator(
        task_id="export_dataset_examples_to_file",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        path="/tmp/arize_e2e_dataset_examples_{{ ts_nodash }}.json",
    )
    eval_dataset_health = ArizeAxEvalDatasetHealthOperator(
        task_id="eval_dataset_health",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
    )
    smart_dataset_refresh = ArizeAxSmartDatasetRefreshOperator(
        task_id="smart_dataset_refresh",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        max_new_examples=5,
        # Project is fresh, no spans yet — the operator should no-op gracefully.
    )

    # Phase 4 sensors
    dataset_ready = ArizeAxDatasetReadySensor(
        task_id="dataset_ready_sensor",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        min_examples=1,
        poke_interval=5,
        timeout=60,
        mode="poke",
    )
    span_count_sensor = ArizeAxSpanCountSensor(
        task_id="span_count_sensor",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        min_count=0,
        poke_interval=5,
        timeout=30,
        mode="poke",
        soft_fail=True,
    )
    span_ingestion_sensor = ArizeAxSpanIngestionSensor(
        task_id="span_ingestion_sensor",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        stable_for_pokes=1,
        poke_interval=5,
        timeout=30,
        mode="poke",
        soft_fail=True,
    )

    # Phase 5 -- Evaluator lifecycle (LLM, gated by arize_ai_integration_id)
    gate_evaluator = ShortCircuitOperator(
        task_id="gate_evaluator_phase",
        python_callable=_has_ai_integration_var,
        ignore_downstream_trigger_rules=False,
    )
    build_evaluator_config = PythonOperator(
        task_id="build_evaluator_config",
        python_callable=_build_evaluator_template_config,
    )
    create_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_evaluator",
        space_id=SPACE_ID,
        name="e2e-evaluator-{{ ts_nodash }}",
        commit_message="e2e initial version",
        template_config_task_id="build_evaluator_config",
        description="Created by arize_ax_e2e_full_provider_test DAG; safe to delete.",
        if_exists="skip",
    )
    get_evaluator = ArizeAxGetEvaluatorOperator(
        task_id="get_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
    )
    update_evaluator = ArizeAxUpdateEvaluatorOperator(
        task_id="update_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        description="patched by e2e DAG",
    )
    add_evaluator_version = ArizeAxAddEvaluatorVersionOperator(
        task_id="add_evaluator_version",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        commit_message="e2e second version",
        template_config_task_id="build_evaluator_config",
    )
    list_evaluator_versions = ArizeAxListEvaluatorVersionsOperator(
        task_id="list_evaluator_versions",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        limit=10,
    )
    # Defensive: skip get_evaluator_version when the upstream list has no
    # versions yet. Note: even with a valid first_id, the API has been observed
    # to return 400 with an empty detail — that's an Arize bug to file
    # separately; this gate just keeps the DAG green when the precondition
    # genuinely isn't met.
    gate_get_evaluator_version = ShortCircuitOperator(
        task_id="gate_get_evaluator_version",
        python_callable=_make_xcom_present_gate(
            task_ids="list_evaluator_versions", key="first_id",
        ),
        ignore_downstream_trigger_rules=False,
    )
    get_evaluator_version = ArizeAxGetEvaluatorVersionOperator(
        task_id="get_evaluator_version",
        version_id="{{ ti.xcom_pull(task_ids='list_evaluator_versions', key='first_id') }}",
    )

    # Phase 6 -- Experiment lifecycle + scoring + comparison
    run_experiment = ArizeAxRunExperimentOperator(
        task_id="run_experiment",
        name="e2e-run-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        task=_echo_task,
        evaluators=[_exact_match_evaluator, _contains_answer_evaluator],
        concurrency=2,
    )
    run_experiment_v2 = ArizeAxRunExperimentOperator(
        task_id="run_experiment_v2",
        name="e2e-run-v2-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        task=_echo_task,
        evaluators=[_exact_match_evaluator, _contains_answer_evaluator],
        concurrency=2,
    )
    experiment_complete = ArizeAxExperimentCompleteSensor(
        task_id="experiment_complete_sensor",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        poke_interval=5,
        timeout=120,
        mode="poke",
    )
    experiment_run_count_sensor = ArizeAxExperimentRunCountSensor(
        task_id="experiment_run_count_sensor",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        min_runs=1,
        poke_interval=5,
        timeout=60,
        mode="poke",
    )
    evaluation_score_sensor = ArizeAxEvaluationScoreSensor(
        task_id="evaluation_score_sensor",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        # Arize records metric names from the evaluator function's __name__,
        # so this must match `_exact_match_evaluator` exactly.
        metric_name="_exact_match_evaluator",
        min_score=0.0,
        poke_interval=5,
        timeout=60,
        mode="poke",
        soft_fail=True,
    )
    get_experiment = ArizeAxGetExperimentOperator(
        task_id="get_experiment",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
    )
    list_experiments = ArizeAxListExperimentsOperator(
        task_id="list_experiments",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        limit=10,
    )
    list_experiment_runs = ArizeAxListExperimentRunsOperator(
        task_id="list_experiment_runs",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        limit=50,
    )
    get_experiment_score = ArizeAxGetExperimentScoreOperator(
        task_id="get_experiment_score",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
    )
    compare_experiments = ArizeAxCompareExperimentsOperator(
        task_id="compare_experiments",
        baseline_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        candidate_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment_v2') }}",
        pass_threshold=-1.0,  # tolerate any score; we just want to exercise the operator
    )
    detect_eval_drift = ArizeAxDetectEvalDriftOperator(
        task_id="detect_eval_drift",
        baseline_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        current_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment_v2') }}",
        drift_threshold=1.0,  # very lenient; we're not asserting drift behaviour here
    )
    behavioral_regression = ArizeAxBehavioralRegressionOperator(
        task_id="behavioral_regression",
        baseline_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        candidate_experiment_id="{{ ti.xcom_pull(task_ids='run_experiment_v2') }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        significance_threshold=1.0,  # lenient
    )
    eval_budget_allocator = ArizeAxEvalBudgetAllocatorOperator(
        task_id="eval_budget_allocator",
        space_id=SPACE_ID,
        total_budget_spans=1000,
        project_ids=["{{ ti.xcom_pull(task_ids='create_project') }}"],
    )
    export_experiment_runs_to_file = ArizeAxExportExperimentRunsToFileOperator(
        task_id="export_experiment_runs_to_file",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        path="/tmp/arize_e2e_experiment_runs_{{ ts_nodash }}.json",
    )
    annotate_experiment_runs = ArizeAxAnnotateExperimentRunsOperator(
        task_id="annotate_experiment_runs",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        annotations=[
            # record_id will be patched at template time from list_experiment_runs.first_id.
            # Use a categorical-style value: only label needed (score derived).
            {
                "record_id": (
                    "{{ ti.xcom_pull(task_ids='list_experiment_runs', "
                    "key='first_id') }}"
                ),
                "values": [
                    {"name": "e2e-annot-cfg-{{ ts_nodash }}", "label": "correct"}
                ],
            },
        ],
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        space_id=SPACE_ID,
    )

    # LLM-gated calibration uses the experiment we ran above as the candidate.
    gate_calibration = ShortCircuitOperator(
        task_id="gate_calibration",
        python_callable=_has_ai_integration_var,
        ignore_downstream_trigger_rules=False,
    )
    evaluator_calibration = ArizeAxEvaluatorCalibrationOperator(
        task_id="evaluator_calibration",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        annotation_name="e2e-annot-cfg-{{ ts_nodash }}",
        # Arize records metric names from the evaluator function's __name__,
        # so this must match `_exact_match_evaluator` exactly.
        metric_name="_exact_match_evaluator",
        min_samples=1,
        # Don't fail the run if calibration is poor — annotations are sparse here.
    )

    # Annotate dataset examples (HITL) — categorical config with label only.
    annotate_dataset_examples = ArizeAxAnnotateDatasetExamplesOperator(
        task_id="annotate_dataset_examples",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        annotations=[
            {
                "record_id": (
                    "{{ ti.xcom_pull(task_ids='list_dataset_examples', "
                    "key='first_id') }}"
                ),
                "values": [
                    {"name": "e2e-annot-cfg-{{ ts_nodash }}", "label": "correct"}
                ],
            },
        ],
        space_id=SPACE_ID,
    )

    # Phase 7 -- Tasks lifecycle (LLM, gated by arize_ai_integration_id)
    gate_tasks = ShortCircuitOperator(
        task_id="gate_tasks_phase",
        python_callable=_has_ai_integration_var,
        ignore_downstream_trigger_rules=False,
    )
    create_task = ArizeAxCreateTaskOperator(
        task_id="create_task",
        name="e2e-eval-task-{{ ts_nodash }}",
        eval_task_type="template_evaluation",
        evaluators=[
            {"evaluator_id": "{{ ti.xcom_pull(task_ids='create_evaluator') }}"},
        ],
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        is_continuous=False,
        sampling_rate=1.0,
    )
    get_task = ArizeAxGetTaskOperator(
        task_id="get_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
    )
    update_task = ArizeAxUpdateTaskOperator(
        task_id="update_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
        sampling_rate=0.5,
        space_id=SPACE_ID,
    )
    trigger_task_run = ArizeAxTriggerTaskRunOperator(
        task_id="trigger_task_run",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
        space_id=SPACE_ID,
        # Our DAG creates a fresh empty project — no spans yet means trigger
        # has nothing to evaluate. Treat that as a skip instead of a failure
        # so the rest of the lifecycle (list_runs, cancel, etc.) flows.
        skip_on_no_data=True,
    )
    list_task_runs = ArizeAxListTaskRunsOperator(
        task_id="list_task_runs",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
        limit=10,
    )
    get_task_run = ArizeAxGetTaskRunOperator(
        task_id="get_task_run",
        run_id="{{ ti.xcom_pull(task_ids='trigger_task_run') }}",
    )
    task_run_sensor = ArizeAxTaskRunSensor(
        task_id="task_run_sensor",
        run_id="{{ ti.xcom_pull(task_ids='trigger_task_run') }}",
        poke_interval=10,
        timeout=180,
        mode="poke",
        fail_on_error=False,
        soft_fail=True,
    )
    # Guard against cancel running with a placeholder ("None" string) run_id when
    # trigger_task_run was skipped (e.g. AI-integration Variable not set yet).
    gate_cancel_run = ShortCircuitOperator(
        task_id="gate_cancel_run",
        python_callable=_make_xcom_present_gate(task_ids="trigger_task_run"),
        trigger_rule="all_done",  # evaluate even if sensor failed/skipped
        ignore_downstream_trigger_rules=False,
    )
    cancel_task_run = ArizeAxCancelTaskRunOperator(
        task_id="cancel_task_run",
        run_id="{{ ti.xcom_pull(task_ids='trigger_task_run') }}",
    )

    def _log_synthetic_spans(**ctx) -> int:
        """Push a tiny set of OpenInference-shaped spans into the fresh
        project so that ``trigger_task_run`` (span-scoped template_evaluation)
        has data to evaluate.

        Without this, the freshly-created project has 0 spans and the eval
        trigger fires ``skip_on_no_data=True`` → cascade-skip on
        ``task_run_sensor`` / ``get_task_run`` / ``cancel_task_run``.

        Uses ``hook.log_spans`` (which hits ``client.spans.log``) with a
        minimal OpenInference parent-span shape.
        """
        import time
        from datetime import datetime, timezone

        import pandas as pd

        ti = ctx["ti"]
        project_name = "e2e-proj-" + ctx["ts_nodash"]
        space_id = (
            ti.xcom_pull(task_ids="list_spaces", key="first_id")
            or ctx["dag_run"].conf.get("space_id")
        )
        # Try the Variable first (matches the DAG's SPACE_ID resolution).
        try:
            from airflow.models import Variable
            space_var = Variable.get("arize_ax_space_id", default_var=None)
            if space_var:
                space_id = space_var
        except Exception:
            pass

        # Build 3 spans with the minimum OpenInference column set.
        now_ns = int(time.time() * 1e9)
        rows = []
        for i in range(3):
            rows.append({
                "context.span_id": f"e2e-span-{ctx['ts_nodash']}-{i:03d}",
                "context.trace_id": f"e2e-trace-{ctx['ts_nodash']}-{i:03d}",
                "name": "smoke-eval-target",
                "span_kind": "LLM",
                "attributes.input.value": f"echo: hello world {i}",
                "attributes.output.value": f"hello world {i}",
                "attributes.openinference.span.kind": "LLM",
                "start_time": datetime.fromtimestamp(
                    (now_ns - 60_000_000_000 + i * 1_000_000_000) / 1e9,
                    tz=timezone.utc,
                ),
                "end_time": datetime.fromtimestamp(
                    (now_ns - 59_000_000_000 + i * 1_000_000_000) / 1e9,
                    tz=timezone.utc,
                ),
                "status_code": "OK",
            })
        df = pd.DataFrame(rows)

        hook = ArizeAxHook()
        resp = hook.log_spans(
            space_id=space_id,
            project_name=project_name,
            dataframe=df,
        )
        print(f"[log_synthetic_spans] logged {len(rows)} spans to '{project_name}': {resp}")
        return len(rows)

    log_synthetic_spans = PythonOperator(
        task_id="log_synthetic_spans",
        python_callable=_log_synthetic_spans,
    )

    # log_synthetic_spans returns 200 OK immediately, but Arize's ingestion
    # pipeline is async (Kafka → backend → queryable store) and takes
    # tens of seconds to minutes to make the spans visible to ``spans.list``.
    # Best-effort: if ingestion hasn't completed in 5 minutes, soft_fail so
    # the downstream span-scoped eval trigger skips gracefully (its own
    # ``skip_on_no_data=True`` handles the no-data path) rather than
    # blocking the entire DAG run on observability latency.
    wait_for_synthetic_spans = ArizeAxSpanCountSensor(
        task_id="wait_for_synthetic_spans",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        min_count=1,
        poke_interval=10,
        timeout=300,  # up to 5 minutes for ingestion
        mode="poke",
        soft_fail=True,
    )

    # Run-experiment task variant.
    # Wire in the freshly-created AI integration so the run_experiment task
    # references one that exists in the DAG's own space — the legacy
    # path-of-reading-a-Variable can leak a cross-space integration ID.
    build_run_config = PythonOperator(
        task_id="build_run_config",
        python_callable=_build_run_configuration,
        op_kwargs={
            "integration_id": "{{ ti.xcom_pull(task_ids='create_ai_integration') }}",
        },
    )
    create_run_experiment_task = ArizeAxCreateRunExperimentTaskOperator(
        task_id="create_run_experiment_task",
        name="e2e-run-exp-task-{{ ts_nodash }}",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        run_configuration="{{ ti.xcom_pull(task_ids='build_run_config') }}",
        space_id=SPACE_ID,
        if_exists="skip",
    )

    # Demonstrate server-side experiment → evaluation chaining (SDK 8.27+).
    # Arize triggers the run_experiment task, and once it completes the listed
    # evaluation tasks run automatically against the new spans — no extra
    # Airflow task needed to wait + fan out. Reuses ``create_task`` (a
    # project-scoped template_evaluation task built earlier in this DAG) so
    # the chain has a real evaluator on the other side.
    #
    # Uses the ``arize_ax_chained_experiment_eval`` TaskGroup helper so DAG
    # authors get trigger + (optional) sensor + (optional) get_result wired
    # up in one call, instead of hand-rolling three operators. Here we keep
    # the default ``wait_for_completion=False`` — the rest of the e2e DAG
    # is fire-and-forget at this step.
    trigger_chained_experiment_run = arize_ax_chained_experiment_eval(
        group_id="trigger_chained_experiment_run",
        task_id_param="{{ ti.xcom_pull(task_ids='create_run_experiment_task') }}",
        space_id=SPACE_ID,
        experiment_name="e2e-chained-{{ ts_nodash }}",
        max_examples=2,
        evaluation_task_ids=[
            "{{ ti.xcom_pull(task_ids='create_task') }}",
        ],
    )

    # Phase 8 -- Annotations lifecycle
    create_annotation_config = ArizeAxCreateAnnotationConfigOperator(
        task_id="create_annotation_config",
        space=SPACE_ID,
        name="e2e-annot-cfg-{{ ts_nodash }}",
        config_type="categorical",
        values=[
            {"label": "correct", "score": 1},
            {"label": "wrong", "score": 0},
        ],
    )

    # Annotation queue sub-phase (gated by arize_annotator_email)
    gate_annotation_queue = ShortCircuitOperator(
        task_id="gate_annotation_queue",
        python_callable=_has_annotator_email_var,
        ignore_downstream_trigger_rules=False,
    )

    def _build_annotator_emails(**_ctx) -> list[str]:
        return [_get_first_var("arize_annotator_email", "ARIZE_ANNOTATOR_EMAIL")]

    build_annotator_emails = PythonOperator(
        task_id="build_annotator_emails",
        python_callable=_build_annotator_emails,
    )
    create_annotation_queue = ArizeAxCreateAnnotationQueueOperator(
        task_id="create_annotation_queue",
        name="e2e-annot-q-{{ ts_nodash }}",
        space=SPACE_ID,
        annotation_config_ids=[
            "{{ ti.xcom_pull(task_ids='create_annotation_config') }}",
        ],
        annotator_emails="{{ ti.xcom_pull(task_ids='build_annotator_emails') }}",
        instructions="e2e annotation queue — safe to delete.",
    )
    get_annotation_queue = ArizeAxGetAnnotationQueueOperator(
        task_id="get_annotation_queue",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        space=SPACE_ID,
    )
    update_annotation_queue = ArizeAxUpdateAnnotationQueueOperator(
        task_id="update_annotation_queue",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        space=SPACE_ID,
        instructions="updated by e2e DAG",
    )
    # Manually inject a record source so AddAnnotationQueueRecords has work to do.
    add_annotation_queue_records = ArizeAxAddAnnotationQueueRecordsOperator(
        task_id="add_annotation_queue_records",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        space=SPACE_ID,
        record_sources=[
            # SDK shape: AnnotationQueueExampleRecordInput uses key "record_type"
            # (discriminator) with value "example" — not "type"/"dataset".
            {
                "record_type": "example",
                "dataset_id": "{{ ti.xcom_pull(task_ids='create_dataset') }}",
            },
        ],
    )

    def _wait_for_annotation_queue_records(**ctx) -> int:
        """Poll the queue until at least one record materializes.

        ``add_annotation_queue_records`` returns immediately, but the backend
        scans the dataset and adds each example as a queue record
        asynchronously. Without this wait, ``list_annotation_queue_records``
        races and returns an empty list, causing the entire annotate /
        assign / delete branch to skip.
        """
        import time
        ti = ctx["ti"]
        queue_id = ti.xcom_pull(task_ids="create_annotation_queue")
        hook = ArizeAxHook()
        for attempt in range(20):  # 20 × 3s = up to 60s
            try:
                result = hook.list_annotation_queue_records(
                    annotation_queue=queue_id, limit=10,
                )
                items = (result or {}).get("items") if isinstance(result, dict) else None
                count = len(items or [])
                if count > 0:
                    print(f"[wait] queue has {count} record(s) after {attempt * 3}s")
                    return count
            except Exception as exc:  # noqa: BLE001
                print(f"[wait] attempt {attempt} probe error: {exc}")
            time.sleep(3)
        print("[wait] timed out after 60s — queue still empty, downstream will skip")
        return 0

    wait_for_annotation_queue_records = PythonOperator(
        task_id="wait_for_annotation_queue_records",
        python_callable=_wait_for_annotation_queue_records,
    )

    list_annotation_queue_records = ArizeAxListAnnotationQueueRecordsOperator(
        task_id="list_annotation_queue_records",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        space=SPACE_ID,
        limit=5,
    )

    annotation_queue_sensor = ArizeAxAnnotationQueueSensor(
        task_id="annotation_queue_sensor",
        space_id=SPACE_ID,
        min_count=1,
        poke_interval=5,
        timeout=30,
        mode="poke",
        soft_fail=True,
    )

    def _annotate_first_record(**ctx) -> dict[str, Any]:
        """Pick the first record from the queue (if any) and submit an
        annotation + assignment via the hook directly so we can chain to the
        AnnotateQueueRecord and AssignQueueRecord operators without forcing
        Jinja into ambiguous list-of-dicts payloads.
        """
        ti = ctx["ti"]
        recs = ti.xcom_pull(task_ids="list_annotation_queue_records")
        items = (recs or {}).get("items") if isinstance(recs, dict) else None
        if not items:
            print("[annotate_first_record] queue has no records — nothing to push")
            return {"record_id": None}
        first_id = str(items[0].get("id") or "")
        return {"record_id": first_id}

    pick_first_queue_record = PythonOperator(
        task_id="pick_first_queue_record",
        python_callable=_annotate_first_record,
    )

    def _has_real_queue_record(**ctx) -> bool:
        """Skip downstream queue-record ops if pick_first_queue_record didn't
        find any real records (freshly created queue has none until the
        backend processes its record_sources)."""
        ti = ctx["ti"]
        rec = ti.xcom_pull(task_ids="pick_first_queue_record")
        rid = (rec or {}).get("record_id") if isinstance(rec, dict) else None
        ok = bool(rid)
        if not ok:
            print("[gate] queue has no records yet — skipping annotate/assign/delete")
        return ok

    gate_queue_records = ShortCircuitOperator(
        task_id="gate_queue_records",
        python_callable=_has_real_queue_record,
        ignore_downstream_trigger_rules=False,
    )

    annotate_queue_record = ArizeAxAnnotateQueueRecordOperator(
        task_id="annotate_queue_record",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        record_id=(
            "{{ ti.xcom_pull(task_ids='pick_first_queue_record')['record_id'] }}"
        ),
        annotations=[
            {"name": "e2e-annot-cfg-{{ ts_nodash }}", "label": "correct"},
        ],
        space=SPACE_ID,
    )
    assign_queue_record = ArizeAxAssignQueueRecordOperator(
        task_id="assign_queue_record",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        record_id=(
            "{{ ti.xcom_pull(task_ids='pick_first_queue_record')['record_id'] }}"
        ),
        assigned_user_emails="{{ ti.xcom_pull(task_ids='build_annotator_emails') }}",
        space=SPACE_ID,
    )
    delete_annotation_queue_records = ArizeAxDeleteAnnotationQueueRecordsOperator(
        task_id="delete_annotation_queue_records",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        record_ids=[
            "{{ ti.xcom_pull(task_ids='pick_first_queue_record')['record_id'] }}",
        ],
        space_id=SPACE_ID,
        ignore_if_missing=True,
    )

    # Phase 9 -- Prompts
    create_prompt = ArizeAxCreatePromptOperator(
        task_id="create_prompt",
        space_id=SPACE_ID,
        name="e2e-test-prompt-{{ ts_nodash }}",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": "{query}"},
        ],
        provider="open_ai",
        input_variable_format="f_string",
        model="gpt-4o-mini",
        commit_message="Initial version from e2e run {{ ts_nodash }}",
        if_exists="skip",
    )
    get_prompt = ArizeAxGetPromptOperator(
        task_id="get_prompt",
        prompt_id="{{ ti.xcom_pull(task_ids='create_prompt') }}",
    )
    list_prompts = ArizeAxListPromptsOperator(
        task_id="list_prompts",
        space_id=SPACE_ID,
        limit=10,
    )
    promote_prompt = ArizeAxPromotePromptOperator(
        task_id="promote_prompt",
        prompt_name="e2e-test-prompt-{{ ts_nodash }}",
        label="staging",
        space_id=SPACE_ID,
    )

    # ComparePrompts only runs when AI integration is available
    # (it actually invokes the LLM via the prompt).
    gate_compare_prompts = ShortCircuitOperator(
        task_id="gate_compare_prompts",
        python_callable=_has_ai_integration_var,
        ignore_downstream_trigger_rules=False,
    )
    compare_prompts = ArizeAxComparePromptsOperator(
        task_id="compare_prompts",
        prompt_names=["e2e-test-prompt-{{ ts_nodash }}"],
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        task=_echo_task,
        evaluators=[_exact_match_evaluator],
        experiment_name_prefix="e2e-prompt-ab-{{ ts_nodash }}",
        space_id=SPACE_ID,
        concurrency=1,
    )

    # Phase 10 -- Spans (alpha, non-critical)
    list_spans = ArizeAxListSpansOperator(
        task_id="list_spans",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        limit=5,
    )

    # Phase 11 -- Verification + inspection window
    verify = PythonOperator(
        task_id="verify_results",
        python_callable=_verify_results,
        trigger_rule="all_done",
    )
    inspection_window = PythonOperator(
        task_id="inspection_window",
        python_callable=_inspection_pause,
        trigger_rule="all_done",
    )

    # Phase 12 -- Cleanup (always runs, deletion ordered by dependency)
    delete_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_delete_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
        space_id=SPACE_ID,
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_run_experiment_task = ArizeAxDeleteTaskOperator(
        task_id="cleanup_delete_run_experiment_task",
        task_id_param="{{ ti.xcom_pull(task_ids='create_run_experiment_task') }}",
        space_id=SPACE_ID,
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_delete_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_experiment = ArizeAxDeleteExperimentOperator(
        task_id="cleanup_delete_experiment",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_experiment_v2 = ArizeAxDeleteExperimentOperator(
        task_id="cleanup_delete_experiment_v2",
        experiment_id="{{ ti.xcom_pull(task_ids='run_experiment_v2') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_annotation_queue = ArizeAxDeleteAnnotationQueueOperator(
        task_id="cleanup_delete_annotation_queue",
        annotation_queue="{{ ti.xcom_pull(task_ids='create_annotation_queue') }}",
        space=SPACE_ID,
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_annotation_config = ArizeAxDeleteAnnotationConfigOperator(
        task_id="cleanup_delete_annotation_config",
        annotation_config="{{ ti.xcom_pull(task_ids='create_annotation_config') }}",
        space=SPACE_ID,
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_prompt = ArizeAxDeletePromptOperator(
        task_id="cleanup_delete_prompt",
        prompt_id="{{ ti.xcom_pull(task_ids='create_prompt') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_dataset = ArizeAxDeleteDatasetOperator(
        task_id="cleanup_delete_dataset",
        dataset_id="{{ ti.xcom_pull(task_ids='create_dataset') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_project = ArizeAxDeleteProjectOperator(
        task_id="cleanup_delete_project",
        project_id="{{ ti.xcom_pull(task_ids='create_project') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_api_key = ArizeAxDeleteAPIKeyOperator(
        task_id="cleanup_delete_api_key",
        api_key_id="{{ ti.xcom_pull(task_ids='create_api_key') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )
    delete_ai_integration = ArizeAxDeleteAIIntegrationOperator(
        task_id="cleanup_delete_ai_integration",
        integration_id="{{ ti.xcom_pull(task_ids='create_ai_integration') }}",
        space_id=SPACE_ID,
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    # Wiring

    # Phase 1: independent listings (only space_id-dependent ones chain off list_spaces).
    list_spaces >> [
        list_ai_integrations,
        list_annotation_configs,
        list_annotation_queues,
        list_api_keys,
        list_evaluators,
        list_tasks,
        list_projects,
        list_datasets,
    ]

    # Phase 2: AI integration CRUD
    list_spaces >> create_ai_integration >> get_ai_integration >> update_ai_integration

    # Phase 3: API key CRUD
    list_spaces >> create_api_key >> refresh_api_key

    # Phase 4: project + dataset setup + enrichment
    list_spaces >> create_project >> [get_project, span_count_sensor, span_ingestion_sensor]
    list_spaces >> create_dataset >> [get_dataset, append_examples]
    append_examples >> list_dataset_examples >> export_dataset_examples_to_file
    list_dataset_examples >> dataset_ready
    [create_project, create_dataset] >> eval_dataset_health
    [create_project, create_dataset] >> smart_dataset_refresh

    # Phase 5: evaluator lifecycle (gated)
    list_spaces >> gate_evaluator >> build_evaluator_config >> create_evaluator
    create_evaluator >> [get_evaluator, update_evaluator, add_evaluator_version]
    # list_evaluator_versions must wait for add_evaluator_version so the list
    # finds at least 2 versions (Arize's evaluators.list_versions does not appear
    # to return the initial create_evaluator version synchronously when polled
    # immediately after create — wait for v2 to guarantee non-empty results).
    [create_evaluator, add_evaluator_version] >> list_evaluator_versions >> gate_get_evaluator_version >> get_evaluator_version

    # Phase 6: experiment lifecycle
    dataset_ready >> [run_experiment, run_experiment_v2]
    run_experiment >> [
        experiment_complete,
        experiment_run_count_sensor,
        evaluation_score_sensor,
        get_experiment,
        list_experiment_runs,
        get_experiment_score,
        export_experiment_runs_to_file,
    ]
    create_dataset >> list_experiments
    [run_experiment, run_experiment_v2] >> compare_experiments
    [run_experiment, run_experiment_v2] >> detect_eval_drift
    [run_experiment, run_experiment_v2] >> behavioral_regression
    create_project >> eval_budget_allocator

    # Dataset-level HITL annotation (needs example IDs).
    # Both annotate ops reference the annotation config by name, so the config
    # must exist before they fire. They also depend on real first_id XComs from
    # the upstream list operators — if those are empty/missing, the API
    # returns 400 or even 500 instead of a clean validation error. Gate to skip.
    gate_annotate_ds = ShortCircuitOperator(
        task_id="gate_annotate_dataset_examples",
        python_callable=_make_xcom_present_gate(
            task_ids="list_dataset_examples", key="first_id",
        ),
        ignore_downstream_trigger_rules=False,
    )
    gate_annotate_exp = ShortCircuitOperator(
        task_id="gate_annotate_experiment_runs",
        python_callable=_make_xcom_present_gate(
            task_ids="list_experiment_runs", key="first_id",
        ),
        ignore_downstream_trigger_rules=False,
    )
    [list_dataset_examples, create_annotation_config] >> gate_annotate_ds
    gate_annotate_ds >> annotate_dataset_examples
    [list_experiment_runs, create_annotation_config] >> gate_annotate_exp
    gate_annotate_exp >> annotate_experiment_runs
    # Evaluator calibration is LLM-gated and depends on annotations being present.
    annotate_experiment_runs >> gate_calibration >> evaluator_calibration

    # Phase 7: tasks (gated, depend on evaluator + project)
    [create_evaluator, create_project] >> gate_tasks
    gate_tasks >> create_task >> [get_task, list_task_runs, update_task]
    # Span-scoped eval trigger needs spans in the project AND those spans
    # need to be ingested into the queryable path before the trigger fires.
    create_project >> log_synthetic_spans >> wait_for_synthetic_spans
    [create_task, wait_for_synthetic_spans] >> trigger_task_run >> [get_task_run, task_run_sensor]
    [trigger_task_run, task_run_sensor] >> gate_cancel_run >> cancel_task_run
    # build_run_config now consumes create_ai_integration's XCom, so we need
    # that upstream too (in addition to gate_tasks).
    [gate_tasks, create_ai_integration] >> build_run_config >> create_run_experiment_task
    create_dataset >> create_run_experiment_task
    # Chained trigger needs the eval task to be created first.
    [create_task, create_run_experiment_task] >> trigger_chained_experiment_run

    # Phase 8: annotations
    list_spaces >> create_annotation_config
    create_annotation_config >> gate_annotation_queue >> build_annotator_emails
    build_annotator_emails >> create_annotation_queue
    create_annotation_queue >> [
        get_annotation_queue,
        update_annotation_queue,
        add_annotation_queue_records,
        annotation_queue_sensor,
    ]
    # Wait for the queue records to actually materialize before listing.
    # ``add_annotation_queue_records`` is fire-and-forget on the backend.
    add_annotation_queue_records >> wait_for_annotation_queue_records
    wait_for_annotation_queue_records >> list_annotation_queue_records
    list_annotation_queue_records >> pick_first_queue_record >> gate_queue_records
    gate_queue_records >> [annotate_queue_record, assign_queue_record]
    [annotate_queue_record, assign_queue_record] >> delete_annotation_queue_records

    # Phase 9: prompts
    list_spaces >> create_prompt >> [get_prompt, list_prompts, promote_prompt]
    [create_prompt, create_dataset] >> gate_compare_prompts >> compare_prompts

    # Phase 10: spans
    create_project >> list_spans

    # Phase 11: verification feeds off "most everything", then inspection window
    [
        # Phase 1
        list_ai_integrations,
        list_annotation_configs,
        list_annotation_queues,
        list_api_keys,
        list_evaluators,
        list_tasks,
        list_projects,
        list_datasets,
        # Phase 2-3
        get_ai_integration,
        update_ai_integration,
        refresh_api_key,
        # Phase 4
        get_project,
        get_dataset,
        list_dataset_examples,
        export_dataset_examples_to_file,
        eval_dataset_health,
        smart_dataset_refresh,
        dataset_ready,
        span_count_sensor,
        span_ingestion_sensor,
        # Phase 5
        get_evaluator,
        update_evaluator,
        add_evaluator_version,
        list_evaluator_versions,
        get_evaluator_version,
        # Phase 6
        experiment_complete,
        experiment_run_count_sensor,
        evaluation_score_sensor,
        get_experiment,
        list_experiments,
        list_experiment_runs,
        get_experiment_score,
        compare_experiments,
        detect_eval_drift,
        behavioral_regression,
        eval_budget_allocator,
        export_experiment_runs_to_file,
        annotate_dataset_examples,
        annotate_experiment_runs,
        evaluator_calibration,
        # Phase 7
        get_task,
        list_task_runs,
        update_task,
        task_run_sensor,
        cancel_task_run,
        create_run_experiment_task,
        # Phase 8
        get_annotation_queue,
        update_annotation_queue,
        list_annotation_queue_records,
        add_annotation_queue_records,
        annotation_queue_sensor,
        annotate_queue_record,
        assign_queue_record,
        delete_annotation_queue_records,
        # Phase 9
        get_prompt,
        list_prompts,
        promote_prompt,
        compare_prompts,
        # Phase 10
        list_spans,
    ] >> verify

    verify >> inspection_window

    # Phase 12: cleanup (correct deletion order — children before parents)
    inspection_window >> [
        delete_task,
        delete_run_experiment_task,
        delete_annotation_queue,
    ] >> delete_evaluator
    inspection_window >> [
        delete_experiment,
        delete_experiment_v2,
        delete_prompt,
    ]
    [delete_evaluator, delete_experiment, delete_experiment_v2] >> delete_dataset
    [delete_dataset, delete_prompt] >> delete_project
    delete_annotation_queue >> delete_annotation_config
    inspection_window >> [delete_api_key, delete_ai_integration]
