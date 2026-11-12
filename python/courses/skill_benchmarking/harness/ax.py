"""Small helpers around the Arize AX SDK: find datasets and experiments, read run results,
and run the LLM judge."""

import time
from datetime import datetime, timedelta

import pandas as pd

from harness.config import Settings

JUDGE_COLUMN = "judge"
JUDGE_RESULT = f"trace_eval.{JUDGE_COLUMN}"


# --- Datasets and experiments ---------------------------------------------------------------

def find_dataset(client, s: Settings, name: str):
    for dataset in client.datasets.list(name=name, space=s.arize_space_id).datasets or []:
        if dataset.name == name:
            return dataset
    return None


def require_dataset(client, s: Settings, name: str):
    dataset = find_dataset(client, s, name)
    if dataset is None:
        raise SystemExit(f"No dataset named {name!r}. Create it with `uv run create_dataset.py`.")
    return dataset


def find_experiment(client, s: Settings, dataset_id: str, name: str):
    for experiment in client.experiments.list(dataset=dataset_id, space=s.arize_space_id, limit=100).experiments or []:
        if experiment.name == name:
            return experiment
    return None


def traces_project(dataset_id: str) -> str:
    """The project where Arize AX keeps the traces for every experiment on a dataset.

    The harness sends each agent session there, so the experiment can link every run to its
    trace with the See Experiment Trace icon.
    """
    return f"experiment_traces_for_dataset_{dataset_id}"


def traces_project_id(client, s: Settings, dataset_id: str) -> str | None:
    """The ID of that project, taken from any experiment on the dataset. Experiment projects
    can't be looked up by name, and an evaluation task needs the ID."""
    for experiment in client.experiments.list(dataset=dataset_id, space=s.arize_space_id, limit=1).experiments or []:
        return experiment.experiment_traces_project_id
    return None


def eval_names(rows: list[dict]) -> list[str]:
    names = {key.split(".")[1] for row in rows for key in row if key.startswith("eval.") and key.endswith(".score")}
    return sorted(names, key=lambda n: (n == JUDGE_COLUMN, n))


# --- The LLM judge --------------------------------------------------------------------------

def find_judge(client, s: Settings):
    for evaluator in client.evaluators.list(name=s.judge_name, space=s.arize_space_id).evaluators or []:
        if evaluator.name == s.judge_name:
            return evaluator
    return None


def _root_spans(client, s: Settings, project: str, start: datetime, end: datetime):
    """The root span of every agent trace in a time window, with any trace evals on it."""
    df = client.spans.export_to_df(space_id=s.arize_space_id, project_name=project,
                                   start_time=start, end_time=end)
    if df.empty or "parent_id" not in df.columns:
        return df
    return df[df["parent_id"].isna()]


def wait_for_traces(client, s: Settings, project: str, session_ids: set[str], start: datetime, end: datetime,
                    column: str | None = None, timeout: int = 600):
    """Wait until every session has a root span in Arize AX, and, with `column`, a value in it."""
    deadline = time.time() + timeout
    while True:
        roots = _root_spans(client, s, project, start, end)
        if not roots.empty:
            roots = roots[roots["attributes.session.id"].isin(session_ids)]
            if column:
                roots = roots[roots[column].notna()] if column in roots.columns else roots.iloc[0:0]
        found = set(roots["attributes.session.id"]) if not roots.empty else set()
        if found >= session_ids or time.time() > deadline:
            return roots, session_ids - found
        time.sleep(10)


def _epoch_ms(value) -> int:
    ts = pd.Timestamp(value)
    return int((ts.tz_localize("UTC") if ts.tzinfo is None else ts).timestamp() * 1000)


def agent_traces(client, s: Settings, project: str, session_ids: set[str], start: datetime, end: datetime) -> dict:
    """Wait for each agent session's trace, and return {session_id: {"id", "timestamp"}}: the trace
    ID as Arize AX stores it, and the trace's start time in milliseconds."""
    print("Waiting for the agent traces to arrive in Arize AX...")
    # Every experiment on the dataset traces to the same project, so the window starts just before
    # this run's sessions, or the judge would also score the end of the previous experiment.
    start, end = start - timedelta(seconds=5), end + timedelta(minutes=1)
    roots, missing = wait_for_traces(client, s, project, session_ids, start, end)
    if missing:
        print(f"  {len(missing)} traces didn't arrive, so those runs have no trace link or judge result")
    return {
        row["attributes.session.id"]: {
            "id": row["context.trace_id"],
            "timestamp": _epoch_ms(row["start_time"]),
        }
        for _, row in roots.iterrows()
    }


def run_judge(client, s: Settings, project: str, project_id: str, session_ids: set[str],
              start: datetime, end: datetime) -> dict:
    """Run the LLM judge on the agent traces from one experiment, and return its result for each
    session: {session_id: {"score", "label", "explanation"}}.

    The judge is a trace-level evaluator on the experiment traces project. For each trace, Arize AX fills
    {input} and {output} from the root span (the request and the agent's answer) and
    {tool_results} from every tool span, which is where the campaign list is if the agent read it.
    """
    from arize._generated.api_client.models.task_query_mapping_input import TaskQueryMappingInput
    from arize._generated.api_client.models.trace_or_session_evaluator_input import TraceOrSessionEvaluatorInput
    from arize.tasks.types import TaskEvaluatorInput, TaskQueryFilter, TaskQueryFilters, TaskType

    judge = find_judge(client, s)
    if judge is None:
        print(f"No {s.judge_name!r} evaluator in your space yet, so only the code evaluators ran.")
        return {}
    start, end = start - timedelta(seconds=5), end + timedelta(minutes=1)

    evaluator = TaskEvaluatorInput(TraceOrSessionEvaluatorInput(evaluator_id=judge.id, query_mappings=[
        TaskQueryMappingInput(variable_name="input", query_ids=["A"], attribute_path="attributes.input.value"),
        TaskQueryMappingInput(variable_name="output", query_ids=["A"], attribute_path="attributes.output.value"),
        TaskQueryMappingInput(variable_name="tool_results", query_ids=["B"], attribute_path="attributes.output.value"),
    ]))
    task = client.tasks.create_evaluation_task(
        name=f"{s.judge_name} {int(time.time())}",
        task_type=TaskType.TEMPLATE_EVALUATION,
        evaluators=[evaluator],
        project=project_id,
        space=s.arize_space_id,
        query_filters=TaskQueryFilters(filters=[
            TaskQueryFilter(id="A", filter="parent_id IS NULL"),
            TaskQueryFilter(id="B", filter="attributes.openinference.span.kind = 'TOOL'"),
        ], expression="A OR B"),
        is_continuous=False,
    )
    print("Running the LLM judge on the agent traces in Arize AX...")
    run = client.tasks.trigger_run(task=task.id, space=s.arize_space_id, data_start_time=start, data_end_time=end)
    finished = client.tasks.wait_for_run(run_id=run.id, poll_interval=5, timeout=900)
    print(f"Judge finished: {finished.status} ({finished.num_successes} scored, {finished.num_errors} errors)")

    roots, missing = wait_for_traces(client, s, project, session_ids, start, end, column=f"{JUDGE_RESULT}.label", timeout=180)
    if missing:
        print(f"  {len(missing)} traces have no judge result")
    return {
        row["attributes.session.id"]: {
            "score": row.get(f"{JUDGE_RESULT}.score"),
            "label": row.get(f"{JUDGE_RESULT}.label"),
            "explanation": row.get(f"{JUDGE_RESULT}.explanation"),
        }
        for _, row in roots.iterrows()
    }
