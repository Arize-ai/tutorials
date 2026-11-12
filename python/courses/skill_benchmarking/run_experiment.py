"""Run the skill against every request in the dataset as an Arize AX experiment (Lessons 2-5).

    uv run run_experiment.py v1-first                        # just check the harness runs
    uv run run_experiment.py v1-code --evaluators            # score with your code evaluators
    uv run run_experiment.py v1 --evaluators --judge         # code evaluators and the LLM judge
    uv run run_experiment.py v2 --evaluators --judge --skill solutions/skills/v2/generate-utm-link
    uv run run_experiment.py v1 --evaluators solutions.evaluators --judge

For each dataset example, the harness runs headless Claude Code in a fresh sandbox with only
the skill under test, and traces the session to the dataset's experiment traces project in
Arize AX. With --evaluators, your code evaluators score each response on your machine. With
--judge, the LLM judge scores each agent trace in Arize AX. Then the script creates the
experiment: one run per example, with the agent's response as the output, a column for every
evaluator, the agent's latency, tokens, and cost, and the agent's trace attached, so the See
Experiment Trace icon opens it.
"""

import argparse
import asyncio
import importlib
import json
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from harness.agent import run_agent
from harness.ax import (
    JUDGE_COLUMN,
    agent_traces,
    eval_names,
    find_experiment,
    require_dataset,
    run_judge,
    traces_project,
    traces_project_id,
)
from harness.config import COURSE_DIR, arize_client, settings
from harness.links import add_session_footer


def load_evaluators(module_name: str) -> list:
    module = importlib.import_module(module_name)
    return list(module.EVALUATORS)


def _append_log(log_path: Path, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


async def run_all(examples, skill_dir: Path, project: str, log_path: Path, concurrency: int) -> list[dict]:
    """Run the agent once per example, a few at a time, and save each session to log_path."""
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def one(example):
        nonlocal done
        row = example.additional_properties
        async with limit:
            run = await run_agent(row["request"], skill_dir, project)
        done += 1
        print(f"  {done}/{len(examples)} {row.get('request_id')} rep {row.get('rep')}"
              + (f": {run.error}" if run.error else ""))
        await asyncio.to_thread(_append_log, log_path, {
            "request_id": row.get("request_id"), "rep": row.get("rep"), "session_id": run.session_id,
            "tools": run.tools, "cost_usd": run.cost_usd, "error": run.error, "response": run.response})
        # A harness error (timeout, crash) isn't a skill failure, so it's counted separately and
        # not scored.
        # latency, token_count, and total_cost are the columns Arize AX's own experiment runner
        # writes, so the experiment shows them as Avg latency, Avg tokens, and Total cost.
        return {"example_id": example.id, "row": row, "session_id": run.session_id, "error": run.error,
                "output": add_session_footer(run.response, run.session_id),
                "latency": run.duration_ms, "token_count": run.tokens, "total_cost": run.cost_usd}

    return await asyncio.gather(*(one(e) for e in examples))


def score(runs: list[dict], evaluators: list, judge: dict, traces: dict) -> None:
    """Add each run's trace and its eval.<name>.score, .label, and .explanation."""
    for run in runs:
        # The same columns Arize AX's own experiment runner writes. They attach the agent's
        # trace to the run.
        trace = traces.get(run["session_id"], {})
        run["result.trace.id"] = trace.get("id")
        run["result.trace.timestamp"] = trace.get("timestamp")
        if run["error"]:
            continue
        for evaluator in evaluators:
            result = evaluator.evaluate(output=run["output"], dataset_row=run["row"])
            run[f"eval.{evaluator.name}.score"] = result.score
            run[f"eval.{evaluator.name}.label"] = result.label
            run[f"eval.{evaluator.name}.explanation"] = result.explanation
        for field, value in judge.get(run["session_id"], {}).items():
            run[f"eval.{JUDGE_COLUMN}.{field}"] = value


def create_experiment(client, s, dataset_id: str, name: str, runs: list[dict]):
    from arize.experiments.evaluators.types import EvaluationResultFieldNames
    from arize.experiments.types import ExperimentTaskFieldNames

    df = pd.DataFrame([{k: v for k, v in run.items() if k != "row"} for run in runs])
    df["error"] = df["error"].fillna("")
    df["result.trace.timestamp"] = df["result.trace.timestamp"].astype("Int64")
    df["token_count"] = df["token_count"].astype("Int64")
    df["latency"] = pd.to_numeric(df["latency"])
    df["total_cost"] = pd.to_numeric(df["total_cost"])
    for n in eval_names(runs):
        # An ignored run has no score. AX needs the column to be numeric, so it's stored as NaN.
        df[f"eval.{n}.score"] = pd.to_numeric(df[f"eval.{n}.score"])
    evaluator_columns = {
        n: EvaluationResultFieldNames(score=f"eval.{n}.score", label=f"eval.{n}.label",
                                      explanation=f"eval.{n}.explanation")
        for n in eval_names(runs)
    }
    # Upload from a Parquet file, which goes over Arrow Flight. The REST upload sends an ignored
    # run's missing score as null, and AX rejects that column as not numeric.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs.parquet"
        df.to_parquet(path, index=False)
        return client.experiments.create(
            name=name,
            dataset=dataset_id,
            space=s.arize_space_id,
            experiment_runs=path,
            task_fields=ExperimentTaskFieldNames(example_id="example_id", output="output"),
            evaluator_columns=evaluator_columns or None,
        )


def print_summary(rows: list[dict]) -> None:
    errors = [r for r in rows if r.get("error")]
    scored = [r for r in rows if not r.get("error")]
    print(f"\n{len(rows)} runs, {len(errors)} harness errors")
    for name in eval_names(rows):
        values = [r.get(f"eval.{name}.score") for r in scored]
        scores = [float(x) for x in values if x is not None and pd.notna(x)]
        # Runs an evaluator skips (no score, label not_applicable) are ignored, not failed.
        ignored = len(values) - len(scores)
        if scores:
            line = f"  {name:14s} {sum(scores):5.0f}/{len(scores):<3d} passed ({sum(scores) / len(scores):.0%})"
            print(line + (f", {ignored} ignored" if ignored else ""))
    failing = Counter()
    for r in scored:
        failed = [n for n in eval_names(rows) if r.get(f"eval.{n}.score") == 0]
        if failed:
            failing[tuple(failed)] += 1
    if failing:
        print("\nFailing checks (runs):")
        for checks, count in failing.most_common():
            print(f"  {', '.join(checks)}: {count}")


def main() -> None:
    s = settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Experiment name, for example v1 (must be new for this dataset)")
    parser.add_argument("--skill", type=Path, default=COURSE_DIR / "skill" / "generate-utm-link")
    parser.add_argument("--dataset", default=s.dataset_name)
    parser.add_argument("--evaluators", nargs="?", const="evaluators", metavar="MODULE",
                        help="Score runs with the code evaluators in MODULE (default: evaluators)")
    parser.add_argument("--judge", action="store_true", help="Score the agent traces with the LLM judge in Arize AX")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    s.require_anthropic_key()
    client = arize_client()
    dataset = require_dataset(client, s, args.dataset)
    if find_experiment(client, s, dataset.id, args.name):
        raise SystemExit(f"Experiment {args.name!r} already exists on {args.dataset!r}. Pick a new name.")
    project = traces_project(dataset.id)
    project_id = traces_project_id(client, s, dataset.id)
    if args.judge and project_id is None:
        raise SystemExit(f"{args.dataset!r} has no experiments yet, so the judge has no traces project to run in. "
                         "Run one experiment without --judge first.")
    evaluators = load_evaluators(args.evaluators) if args.evaluators else []
    examples = client.datasets.list_examples(dataset=dataset.id, space=s.arize_space_id, all=True).examples or []
    log_path = COURSE_DIR / "runs" / f"{args.name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text("")

    print(f"Running {args.skill.name} from {args.skill} on {args.dataset!r} with {s.agent_model}")
    started = time.time()
    start = datetime.now(UTC)
    runs = asyncio.run(run_all(examples, args.skill.resolve(), project, log_path, args.concurrency))
    end = datetime.now(UTC)
    print(f"Agent runs finished in {time.time() - started:.0f}s")

    sessions = {r["session_id"] for r in runs if not r["error"]}
    traces = agent_traces(client, s, project, sessions, start, end)
    judge = run_judge(client, s, project, project_id, sessions, start, end) if args.judge else {}
    score(runs, evaluators, judge, traces)

    experiment = create_experiment(client, s, dataset.id, args.name, runs)
    print(f"Created experiment {args.name!r} ({experiment.id})")
    print_summary(runs)
    cost = sum(json.loads(line).get("cost_usd") or 0 for line in log_path.read_text().splitlines())
    print(f"\nAgent cost: ${cost:.3f}. Session details: {log_path.relative_to(COURSE_DIR)}")


if __name__ == "__main__":
    main()
