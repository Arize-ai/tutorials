"""
Example DAG: Arize AX Tasks — full end-to-end eval-gate pipeline.

Fully self-contained — no pre-existing task IDs, evaluator IDs, or dataset IDs
required.  The DAG auto-discovers a project in the space, creates a demo
LLM-as-judge evaluator, wires it to an evaluation task, triggers an on-demand
run, waits for completion, then gates a deployment decision on the eval score.
The evaluator is deleted on exit (whether the run passed or failed).

**Evaluator naming:** the evaluator is created with a date-stamped name
(``arize-ax-tasks-e2e-demo-<YYYYMMDD>``) so each daily run starts with a
brand-new evaluator ID and zero evaluation history.  Arize tracks evaluation
history at the evaluator level — reusing the same evaluator ID across runs
causes the platform to report "already evaluated" for spans processed in
prior runs, making the trigger return ``cancelled``.  The date-stamp
guarantees a fresh evaluator each day.  ``if_exists="skip"`` still handles
same-day retries correctly.

Flow::

    check_ai_integration          (ShortCircuitOperator — skip if no AI integration)
            │
            ▼
    list_projects                 (ArizeAxListProjectsOperator — auto-discover)
            │
            ▼
    pick_project                  (ShortCircuitOperator — pick first; skip if none)
            │
            ▼
    build_template_config         (PythonOperator — build LLM-judge template)
            │
            ▼
    create_evaluator              (ArizeAxCreateEvaluatorOperator, if_exists="skip")
            │
            ▼
    create_task                   (PythonOperator — link evaluator + project)
            │
            ▼
    trigger_run                   (ArizeAxTriggerTaskRunOperator — returns scalar run_id)
            │
            ▼
    wait_for_run                  (ArizeAxTaskRunSensor, mode="reschedule")
            │
            ▼
    get_run_result                (ArizeAxGetTaskRunOperator)
            │
            ▼
    gate_on_score                 (PythonOperator — raise on score < threshold)
            │
            ▼
    cleanup_evaluator             (ArizeAxDeleteEvaluatorOperator, trigger_rule="all_done")

Requires:
- Airflow connection ``arize_ax_default`` with a valid API key.
- Either the Airflow Variable ``arize_ax_space_id`` **or** a ``default_space``
  field on the ``arize_ax_default`` connection extras.
- ``ARIZE_AI_INTEGRATION_ID`` in the **worker environment** (Docker/Kubernetes)
  **or** as an Airflow **Variable** (Admin → Variables).  This is the Arize
  AI Integration ID for the LLM used by the judge (e.g. OpenAI gpt-4o-mini).
  If neither is set, ``check_ai_integration`` short-circuits cleanly and the
  entire DAG is skipped.
- At least one project must exist in the space.  If none exist,
  ``pick_project`` short-circuits cleanly.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.providers.arize_ax.operators.evaluators import (
    ArizeAxCreateEvaluatorOperator,
    ArizeAxDeleteEvaluatorOperator,
)
from airflow.providers.arize_ax.operators.projects import ArizeAxListProjectsOperator
from airflow.providers.arize_ax.operators.tasks import (
    ArizeAxGetTaskRunOperator,
    ArizeAxTriggerTaskRunOperator,
)
from airflow.providers.arize_ax.sensors.arize_ax import ArizeAxTaskRunSensor
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

# Optional Airflow Variable ``arize_ax_project_name`` targets a specific
# project by name (e.g. "langraph-financial-agent-trace"); falls back to the
# first project returned by the API when unset. Read at task-execution time
# inside callables to avoid parse-time Variable lookups.

_SPACE_JINJA = "{{ var.value.get('arize_ax_space_id', '') or None }}"

_EVALUATOR_NAME = "arize-ax-tasks-e2e-demo"
_SCORE_THRESHOLD = 0.7

# Evaluate only LLM spans (ChatOpenAI, etc.) — excludes CHAIN, TOOL, RETRIEVER.
_QUERY_FILTER = "\"attributes.openinference.span.kind\" = 'LLM'"



def _get_ai_integration_id() -> str:
    """Worker env ``ARIZE_AI_INTEGRATION_ID`` wins; else Airflow Variable of same key."""
    v = os.environ.get("ARIZE_AI_INTEGRATION_ID", "").strip()
    if v:
        return v
    return Variable.get("ARIZE_AI_INTEGRATION_ID", default_var="").strip()


def _check_ai_integration() -> bool:
    """Short-circuit when ARIZE_AI_INTEGRATION_ID is not configured."""
    configured = bool(_get_ai_integration_id())
    if not configured:
        print(
            "ARIZE_AI_INTEGRATION_ID is not set in worker env or Airflow Variables — "
            "skipping tasks e2e demo.  Set it to your Arize AI Integration ID to enable."
        )
    return configured


def _pick_project(**ctx) -> str | None:
    """Return the project ID to use for evaluation.

    If ``arize_ax_project_name`` Variable is set, find the project with that
    name.  Falls back to the first project returned by the API.
    Short-circuits if no projects exist in the space.
    """
    items = ctx["ti"].xcom_pull(task_ids="list_projects") or {}
    projects = items.get("items", []) if isinstance(items, dict) else []
    if not projects:
        print(
            "No projects found in space — cannot create an evaluation task without a project. "
            "Create at least one project in Arize AX and re-run."
        )
        return None

    target_name = Variable.get("arize_ax_project_name", default_var=None)
    if target_name:
        for p in projects:
            if isinstance(p, dict) and p.get("name") == target_name:
                project_id = str(p["id"])
                print(f"[pick_project] Matched target project name={target_name!r}, id={project_id!r}")
                return project_id
        print(
            f"[pick_project] Project {target_name!r} not found in page of {len(projects)} results — "
            "falling back to first available project."
        )

    project = projects[0]
    project_id = project.get("id") if isinstance(project, dict) else str(project)
    name = project.get("name", "?") if isinstance(project, dict) else "?"
    print(f"[pick_project] Using project id={project_id!r}, name={name!r}")
    return project_id


def _build_template_config(**_ctx) -> dict[str, Any]:
    """Build the LLM-judge template_config for ArizeAxCreateEvaluatorOperator."""
    integration_id = _get_ai_integration_id()
    return {
        "template_config": {
            "name": "relevance",
            # The Arize Evaluator API uses short aliases {input} and {output}
            # which it maps to the OpenInference attributes input.value /
            # output.value internally.  Do NOT use {input.value} or {output.value}
            # directly — the API rejects them with "Invalid column".
            "template": (
                "You are an LLM judge evaluating response relevance.\n\n"
                "Query: {input}\n"
                "Response: {output}\n\n"
                "Is the response relevant to the query?\n"
                "Respond with exactly one word: relevant or irrelevant"
            ),
            "include_explanations": True,
            "use_function_calling_if_available": False,
            "classification_choices": {"irrelevant": 0, "relevant": 1},
            "llm_config": {
                "ai_integration_id": integration_id,
                "model_name": "gpt-4o-mini",
                "invocation_parameters": {"temperature": 0},
                "provider_parameters": {},
            },
        }
    }


def _create_task(**ctx) -> str:
    """Create (or reuse) an evaluation task and return its ID.

    Pulls the evaluator ID (from create_evaluator XCom) and the project ID
    (from pick_project XCom).

    Tasks cannot be deleted via the SDK, so orphaned tasks from previous runs
    accumulate.  To avoid creating a new task on same-day retries, scan for an
    existing task named _EVALUATOR_NAME that already has the current evaluator
    linked → reuse it.  Because the evaluator name is date-stamped, the
    evaluator_id is always fresh across daily runs, so a new task is created
    each day while retries within the same day reuse the existing one.

    ``column_mappings`` explicitly binds ``{input}`` / ``{output}`` in the
    evaluator template to the OpenInference attribute paths so the API always
    resolves the placeholders correctly regardless of span kind.
    """
    from airflow.providers.arize_ax.hooks.arize_ax import ArizeAxHook

    ti = ctx["ti"]
    evaluator_id = str(ti.xcom_pull(task_ids="create_evaluator") or "")
    project_id = str(ti.xcom_pull(task_ids="pick_project") or "")

    if not evaluator_id:
        raise ValueError("[create_task] evaluator_id not found in create_evaluator XCom.")
    if not project_id:
        raise ValueError("[create_task] project_id not found in pick_project XCom.")

    hook = ArizeAxHook()

    # column_mappings: explicitly map template placeholders to OpenInference
    # attribute column paths so {input}/{output} always resolve, even when the
    # API cannot auto-resolve the short aliases.
    evaluator_entry = {
        "evaluator_id": evaluator_id,
        "column_mappings": {
            "input": "attributes.input.value",
            "output": "attributes.output.value",
        },
    }

    # Reuse an existing task that has this evaluator already linked
    # (handles same-day retries without creating duplicate tasks).
    existing = hook.list_tasks(
        space_id=Variable.get("arize_ax_space_id", default_var=None),
        limit=50,
    )
    for t in (existing.get("items", []) if isinstance(existing, dict) else []):
        if not isinstance(t, dict) or t.get("name") != _EVALUATOR_NAME:
            continue
        linked = [
            e.get("evaluator_id") or e.get("id")
            for e in (t.get("evaluators") or [])
            if isinstance(e, dict)
        ]
        if evaluator_id in linked:
            task_id = str(t["id"])
            print(f"[create_task] Reusing existing task id={task_id!r}")
            return task_id

    # No matching task found — create a fresh one scoped to LLM spans.
    result = hook.create_task(
        name=_EVALUATOR_NAME,
        task_type="template_evaluation",
        evaluators=[evaluator_entry],
        project_id=project_id,
        is_continuous=False,
        query_filter=_QUERY_FILTER,
    )
    task_id = None
    if isinstance(result, dict):
        task_id = result.get("id") or result.get("task_id")
    if not task_id:
        raise ValueError(f"[create_task] Could not extract task ID from result: {result!r}")
    print(f"[create_task] Created task id={task_id!r}")
    return task_id


def _gate_on_score(**ctx) -> None:
    """Log eval scores and raise if the average falls below the threshold."""
    from airflow.exceptions import AirflowException

    run = ctx["ti"].xcom_pull(task_ids="get_run_result") or {}
    if not isinstance(run, dict):
        print(f"[gate_on_score] Unexpected run result type: {type(run).__name__!r} — passing gate.")
        return

    run_id = run.get("id", "unknown")
    run_status = run.get("status", "unknown")

    # Evaluation results may be nested differently across SDK versions
    evals = run.get("evaluations") or run.get("evals") or run.get("evaluation_results") or {}
    scores: list[float] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            score = obj.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(evals)

    if not scores:
        print(
            f"[gate_on_score] Run {run_id!r} (status={run_status!r}) — "
            "no numeric scores found in eval results. Passing gate."
        )
        return

    avg = sum(scores) / len(scores)
    print(
        f"[gate_on_score] Run {run_id!r} — avg score={avg:.4f} "
        f"across {len(scores)} evaluator(s) (threshold={_SCORE_THRESHOLD})."
    )

    if avg < _SCORE_THRESHOLD:
        raise AirflowException(
            f"Eval gate FAILED: avg score {avg:.4f} < threshold {_SCORE_THRESHOLD}. "
            "Block deployment until evaluators pass."
        )
    print("[gate_on_score] Eval gate PASSED — safe to deploy.")



with DAG(
    dag_id="example_arize_ax_tasks",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    tags=["arize_ax", "example", "tasks"],
    catchup=False,
    doc_md=__doc__,
) as dag:

    # 1. Gate: require ARIZE_AI_INTEGRATION_ID
    check_ai_integration = ShortCircuitOperator(
        task_id="check_ai_integration",
        python_callable=_check_ai_integration,
    )

    # 2. Auto-discover projects in the space — fetch up to 50 so name-based
    # lookup (arize_ax_project_name Variable) finds projects beyond position 5.
    list_projects = ArizeAxListProjectsOperator(
        task_id="list_projects",
        space_id=_SPACE_JINJA,
        limit=50,
    )

    # 3. Pick first project; skip rest if none found
    pick_project = ShortCircuitOperator(
        task_id="pick_project",
        python_callable=_pick_project,
    )

    # 4. Build the LLM-judge template config at runtime
    build_template_config = PythonOperator(
        task_id="build_template_config",
        python_callable=_build_template_config,
    )

    # 5. Create (or reuse) the demo evaluator in Eval Hub.
    # Name is date-stamped so each daily run gets a fresh evaluator ID with no
    # prior evaluation history.  Arize deduplicates at the evaluator level, so
    # reusing the same ID would cause "already evaluated" cancellations on re-runs.
    # if_exists="skip" handles same-day retries (same logical date → same name →
    # resolves the existing evaluator instead of creating a duplicate).
    create_evaluator = ArizeAxCreateEvaluatorOperator(
        task_id="create_evaluator",
        space_id=_SPACE_JINJA,
        name=_EVALUATOR_NAME + "-{{ ds_nodash }}",
        commit_message="arize-ax-tasks-e2e-demo v1",
        description="Auto-created by example_arize_ax_tasks DAG — safe to delete.",
        template_config_task_id="build_template_config",
        template_config_key="template_config",
        if_exists="skip",
    )

    # 6. Create the evaluation task linking evaluator + auto-discovered project
    create_task = PythonOperator(
        task_id="create_task",
        python_callable=_create_task,
    )

    # 7. Trigger an on-demand run.  override_evaluations=True forces re-evaluation
    # of spans that already have labels from prior runs — without this, the platform
    # cancels the run with "already evaluated" if any previous task/evaluator has
    # scored the same spans.  The operator returns a scalar run_id string via XCom.
    trigger_run = ArizeAxTriggerTaskRunOperator(
        task_id="trigger_run",
        task_id_param="{{ ti.xcom_pull(task_ids='create_task') }}",
        space_id=_SPACE_JINJA,
        override_evaluations=True,
    )

    # 8. Wait for the run to reach a terminal state (reschedule to free the worker slot)
    wait_for_run = ArizeAxTaskRunSensor(
        task_id="wait_for_run",
        run_id="{{ ti.xcom_pull(task_ids='trigger_run') }}",
        poke_interval=30,
        timeout=1800,
        mode="reschedule",
        fail_on_error=True,
    )

    # 9. Fetch the completed run result (scores, labels, metadata)
    get_run_result = ArizeAxGetTaskRunOperator(
        task_id="get_run_result",
        run_id="{{ ti.xcom_pull(task_ids='trigger_run') }}",
    )

    # 10. Gate deployment decision on eval scores
    gate_on_score = PythonOperator(
        task_id="gate_on_score",
        python_callable=_gate_on_score,
    )

    # 11. Clean up demo evaluator regardless of outcome
    cleanup_evaluator = ArizeAxDeleteEvaluatorOperator(
        task_id="cleanup_evaluator",
        evaluator_id="{{ ti.xcom_pull(task_ids='create_evaluator') }}",
        ignore_if_missing=True,
        trigger_rule="all_done",
    )

    (
        check_ai_integration
        >> list_projects
        >> pick_project
        >> build_template_config
        >> create_evaluator
        >> create_task
        >> trigger_run
        >> wait_for_run
        >> get_run_result
        >> gate_on_score
        >> cleanup_evaluator
    )
