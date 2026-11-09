#!/usr/bin/env python3
"""End-to-end check: send one request, then look its trace id up in Arize AX and Dynatrace.

    scripts/verify.py collector          # Python app -> local collector -> both backends
    scripts/verify.py collector-nochain  # same, app does not tag its request span
    scripts/verify.py collector-nested   # same, plus an agent nested in an agent
    scripts/verify.py incode             # Python app -> both backends directly
    scripts/verify.py incode-nested      # same, plus an agent nested in an agent
    scripts/verify.py incode-ts          # Node app (example-ts) -> both backends directly
    scripts/verify.py all                # every scenario above (the default)

Reads settings from .env at the kit root (see .env.example). Needs:
  - the `ax` CLI, logged in with a key that can read ARIZE_SPACE_ID. Set AX_PROFILE to
    switch profiles for the run; the previously active profile is restored on exit.
  - a Dynatrace platform token that can run Grail queries (DYNATRACE_QUERY_TOKEN)
  - for the collector scenarios, a collector binary in bin/ (scripts/run-collector.sh
    downloads one). COLLECTOR_BIN picks one by file name.

Writes out/results-<scenario>.json and prints a summary. Exit code 1 if a check fails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

APM_NAMES = ("SELECT", "SELECT customers", "GET")
PYTHON_AI = {"support_agent", "ChatCompletion", "get_order_status", "execute_tool get_weather"}

# app: how to run the example. ai: span name prefixes that must reach both backends.
# apm: must reach Dynatrace only. root: expected Arize root. parents: child -> parent in Arize.
SCENARIOS = {
    "collector": {
        "mode": "collector", "app": {"AI_ROOT": "server"},
        "ai": PYTHON_AI, "apm": {"SELECT", "GET"}, "logs": 6, "root": "POST /ask",
        "parents": {"support_agent": "POST /ask"},
    },
    "collector-nochain": {
        "mode": "collector", "app": {"AI_ROOT": "none"},
        "ai": PYTHON_AI, "apm": {"SELECT", "GET"}, "logs": 6, "root": "POST /ask",
        "parents": {"support_agent": "POST /ask"},
    },
    "collector-nested": {
        "mode": "collector", "app": {"AI_ROOT": "none", "NESTED_AGENTS": "1"},
        "ai": PYTHON_AI | {"triage_agent"}, "apm": {"SELECT", "GET"}, "logs": 6, "root": "POST /ask",
        "parents": {"triage_agent": "POST /ask", "support_agent": "triage_agent"},
    },
    "incode": {
        "mode": "incode", "app": {"AI_ROOT": "none"},
        "ai": PYTHON_AI, "apm": {"SELECT", "GET"}, "logs": 6, "root": "support_agent",
        "parents": {},
    },
    "incode-nested": {
        "mode": "incode", "app": {"AI_ROOT": "none", "NESTED_AGENTS": "1"},
        "ai": PYTHON_AI | {"triage_agent"}, "apm": {"SELECT", "GET"}, "logs": 6, "root": "triage_agent",
        "parents": {"support_agent": "triage_agent"},
    },
    "incode-ts": {
        "mode": "incode-ts", "app": {},
        "ai": {"invoke_agent", "chat ", "execute_tool get_weather"}, "apm": {"SELECT customers", "GET"}, "logs": 3,
        "root": "invoke_agent",
        "parents": {},
    },
}


def covers(names: list[str], prefixes: set[str]) -> bool:
    return all(any(name.startswith(prefix) for name in names) for prefix in prefixes)


def load_env() -> dict[str, str]:
    # .env wins over the shell, so a stale exported key can't shadow the one in .env.
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$", line)
            if match and match.group(2).strip():
                env[match.group(1)] = match.group(2).strip().strip("'\"")
    return env


ENV = load_env()
PROJECTS = {
    "collector": ENV.get("ARIZE_COLLECTOR_PROJECT", "ax-dt-collector-example"),
    "incode": ENV.get("ARIZE_INCODE_PROJECT", "ax-dt-incode-example"),
}


def project_for(mode: str) -> str:
    return PROJECTS["collector" if mode == "collector" else "incode"]


def seconds(duration: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m)", duration.strip())
    if not match:
        return 30.0
    value, unit = float(match.group(1)), match.group(2)
    return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value


# ---------------------------------------------------------------- collector

def collector_metrics() -> dict[str, float]:
    try:
        text = urllib.request.urlopen("http://localhost:8888/metrics", timeout=3).read().decode()
    except OSError:
        return {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r'^(otelcol_(?:exporter_sent|exporter_send_failed)_\w+)\{exporter="([^"]+)"[^}]*\}\s+([\d.e+]+)$', line)
        if match:
            values[f"{match.group(1)}[{match.group(2)}]"] = float(match.group(3))
    return values


@contextmanager
def collector():
    if healthy():
        raise SystemExit("something already listens on :13133; stop it so the run uses collector/config.yaml")
    names = [ENV["COLLECTOR_BIN"]] if ENV.get("COLLECTOR_BIN") else ["dynatrace-otel-collector", "otelcol-contrib"]
    binary = next((ROOT / "bin" / name for name in names if (ROOT / "bin" / name).exists()), None)
    if binary is None:
        raise SystemExit("no collector binary in bin/ (run scripts/run-collector.sh once)")
    OUT.mkdir(exist_ok=True)
    log = open(OUT / "collector.log", "w")
    env = {**ENV, "ARIZE_PROJECT_NAME": PROJECTS["collector"]}
    proc = subprocess.Popen([str(binary), "--config", str(ROOT / "collector/config.yaml")], env=env, stdout=log, stderr=log)
    for _ in range(50):
        if healthy():
            break
        time.sleep(0.2)
    else:
        proc.terminate()
        raise SystemExit("collector did not become healthy; see out/collector.log")
    print(f"collector: started {binary.name} (pid {proc.pid})")
    try:
        yield binary.name
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def healthy() -> bool:
    try:
        return urllib.request.urlopen("http://localhost:13133/", timeout=1).status == 200
    except OSError:
        return False


# ---------------------------------------------------------------- app

def run_app(scenario: dict) -> dict:
    mode = scenario["mode"]
    env = {**ENV, **scenario["app"]}
    if mode == "incode-ts":
        env["ARIZE_PROJECT_NAME"] = project_for(mode)
        proc = subprocess.run(["npx", "tsx", "index.ts"], cwd=ROOT / "example-ts", env=env, capture_output=True, text=True, timeout=300)
        errors_pattern = r"error|failed"
    else:
        if mode == "incode":
            env["ARIZE_PROJECT_NAME"] = project_for(mode)
        else:
            env.pop("ARIZE_PROJECT_NAME", None)  # the collector adds it
        python = ROOT / ".venv/bin/python"
        proc = subprocess.run([str(python), str(ROOT / "example/app.py"), "--mode", mode], env=env, capture_output=True, text=True, timeout=300)
        errors_pattern = r"Failed to export|error|Traceback"
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"app exited {proc.returncode}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["export_errors"] = [line for line in proc.stderr.splitlines() if re.search(errors_pattern, line, re.I)]
    return result


# ---------------------------------------------------------------- Arize

def ax(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ax", *args], capture_output=True, text=True, timeout=180)


@contextmanager
def ax_profile(name: str | None):
    if not name:
        yield
        return
    listing = ax("profiles", "list")
    # ax renders the table on stderr when stdout is not a terminal.
    active = next((m.group(1) for m in re.finditer(r"(\S+) \u25cf", listing.stdout + listing.stderr)), None)
    if active is None:
        raise SystemExit("could not read the active ax profile; refusing to switch profiles")
    if active != name:
        ax("profiles", "use", name)
    try:
        yield
    finally:
        if active != name:
            ax("profiles", "use", active)
            print(f"ax: restored active profile {active}")


def check_arize(project: str, trace_id: str, scenario: dict) -> dict:
    space = ENV["ARIZE_SPACE_ID"]
    spans: list = []
    for _ in range(12):
        proc = ax("spans", "export", project, "--space", space, "--trace-id", trace_id, "--days", "1", "--stdout")
        try:
            spans = json.loads(proc.stdout)
        except json.JSONDecodeError:
            spans = []
        names = [span["name"] for span in spans]
        if covers(names, scenario["ai"] | {scenario["root"]}) and any(not s.get("parent_id") for s in spans):
            break
        time.sleep(10)
    names = sorted(span["name"] for span in spans)
    by_id = {span["context"]["span_id"]: span["name"] for span in spans}
    roots = [span["name"] for span in spans if not span.get("parent_id")]
    dangling = [span["name"] for span in spans if span.get("parent_id") and span["parent_id"] not in by_id]
    parents = {span["name"]: by_id.get(span.get("parent_id")) for span in spans}
    listed = False
    for _ in range(6):
        proc = ax("traces", "list", project, "--space", space, "--limit", "50", "--output", "json")
        try:
            listed = any(t.get("trace_id") == trace_id for t in json.loads(proc.stdout).get("traces", []))
        except json.JSONDecodeError:
            pass
        if listed:
            break
        time.sleep(10)
    return {
        "project": project,
        "span_count": len(spans),
        "spans": names,
        "kinds": {span["name"]: span.get("kind") for span in spans},
        "roots": roots,
        "parents": parents,
        "dangling_parents": dangling,
        "listed_in_traces": listed,
        "apm_spans": [n for n in names if n in APM_NAMES],
        "content_present": any("input.value" in span.get("attributes", {}) for span in spans),
    }


# ---------------------------------------------------------------- Dynatrace

def grail(query: str) -> list:
    apps = ENV.get("DYNATRACE_APPS_URL") or ENV["DYNATRACE_ENV_URL"].rstrip("/").replace(".live.dynatrace.com", ".apps.dynatrace.com")
    token = ENV.get("DYNATRACE_QUERY_TOKEN") or ENV["DYNATRACE_API_TOKEN"]
    scheme = "Api-Token" if token.startswith("dt0c") else "Bearer"

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{apps}/platform/storage/query/v1/{path}",
            data=json.dumps(payload).encode(),
            headers={"authorization": f"{scheme} {token}", "content-type": "application/json"},
        )
        try:
            return json.loads(urllib.request.urlopen(request, timeout=60).read() or b"{}")
        except urllib.error.HTTPError as error:
            return {"state": "FAILED", "error": {"message": f"HTTP {error.code}: {error.read()[:300].decode(errors='replace')}"}}

    body = post("query:execute", {"query": query, "requestTimeoutMilliseconds": 10000, "maxResultRecords": 200})
    for _ in range(20):
        if body.get("state") in (None, "SUCCEEDED", "FAILED") or not body.get("requestToken"):
            break
        time.sleep(1)
        body = post("query:poll", {"requestToken": body["requestToken"], "requestTimeoutMilliseconds": 10000})
    if body.get("state") == "FAILED":
        raise RuntimeError(body.get("error", {}).get("message", "query failed"))
    return body.get("result", {}).get("records", [])


def check_dynatrace(trace_id: str, scenario: dict, service: str) -> dict:
    logs: list = []
    for _ in range(12):
        logs = grail(
            f'fetch logs, from: now()-30m | filter trace_id == "{trace_id}" '
            "| sort timestamp asc | fields timestamp, loglevel, content, span_id, service.name"
        )
        if len(logs) >= scenario["logs"]:
            break
        time.sleep(10)
    metrics = {}
    names = ("example.requests",) if service.endswith("-ts") else ("example.requests", "example.llm.tokens", "example.agent.duration")
    for name in names:
        aggregate = "sum" if name != "example.agent.duration" else "max"
        try:
            rows = grail(f"timeseries v = {aggregate}({name}), from: now()-30m, filter: service.name == \"{service}\"")
            series = rows[0]["v"] if rows else []
            metrics[name] = sum(x for x in series if x is not None) if aggregate == "sum" else max((x for x in series if x is not None), default=None)
        except (RuntimeError, KeyError, IndexError, TypeError) as error:
            metrics[name] = f"error: {error}"
    spans: list = []
    roots: list = []
    for _ in range(9):
        rows = grail(f'fetch spans, from: now()-30m | filter trace.id == toUid("{trace_id}") | fields span.name, span.parent_id')
        spans = sorted(row.get("span.name") for row in rows)
        roots = [row.get("span.name") for row in rows if not row.get("span.parent_id")]
        if covers(spans, scenario["apm"] | scenario["ai"] | {"POST /ask"}):
            break
        time.sleep(10)
    rows = grail(
        f'fetch spans, from: now()-30m | filter trace.id == toUid("{trace_id}") '
        "| filter isNotNull(input.value) or isNotNull(output.value) or isNotNull(gen_ai.tool.call.arguments) "
        "| summarize c = count()"
    )
    return {
        "log_count": len(logs),
        "logs": [f'{row.get("loglevel")} {row.get("content")}' for row in logs],
        "logs_with_span_id": sum(1 for row in logs if row.get("span_id")),
        "metrics_last_30m": metrics,
        "spans": spans,
        "roots": roots,
        "spans_with_ai_content": int(rows[0]["c"]) if rows else 0,
    }


# ---------------------------------------------------------------- main

def verify(name: str) -> bool:
    scenario = SCENARIOS[name]
    mode = scenario["mode"]
    print(f"\n=== {name}")
    sent: dict = {}
    distribution = None
    if mode == "collector":
        with collector() as distribution:
            before = collector_metrics()
            app = run_app(scenario)
            # The root-span branch decides after ARIZE_ROOT_DECISION_WAIT; wait for it.
            time.sleep(seconds(ENV.get("ARIZE_ROOT_DECISION_WAIT", "30s")) + 5)
            after = collector_metrics()
        sent = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
    else:
        app = run_app(scenario)
    trace_id = app["trace_id"]
    print(f"trace id: {trace_id}")

    with ax_profile(ENV.get("AX_PROFILE")):
        arize = check_arize(project_for(mode), trace_id, scenario)
    dynatrace = check_dynatrace(trace_id, scenario, "ax-dt-example-ts" if mode == "incode-ts" else "ax-dt-example")

    checks = {
        "app exported without errors": not app["export_errors"],
        "arize: all AI spans present": covers(arize["spans"], scenario["ai"]),
        "arize: no database or HTTP client spans": not arize["apm_spans"],
        f"arize: one root ({scenario['root']}), no dangling parents": len(arize["roots"]) == 1
        and arize["roots"][0].startswith(scenario["root"]) and not arize["dangling_parents"],
        "arize: trace listed in Traces": arize["listed_in_traces"],
        "arize: prompt and response content present": arize["content_present"],
        "dynatrace: full trace (APM and AI spans)": covers(dynatrace["spans"], scenario["apm"] | scenario["ai"] | {"POST /ask"}),
        "dynatrace: one root (POST /ask)": dynatrace["roots"] == ["POST /ask"],
        "dynatrace: prompt and response content removed": dynatrace["spans_with_ai_content"] == 0,
        "dynatrace: correlated logs found by trace_id": dynatrace["log_count"] >= scenario["logs"],
        "dynatrace: metrics present": isinstance(dynatrace["metrics_last_30m"].get("example.requests"), (int, float))
        and dynatrace["metrics_last_30m"]["example.requests"] > 0,
    }
    for child, parent in scenario["parents"].items():
        checks[f"arize: {child} is a child of {parent}"] = (arize["parents"].get(child) or "").startswith(parent)
    if mode == "collector":
        checks["collector: spans sent to Dynatrace, none failed"] = (
            sent.get("otelcol_exporter_sent_spans[otlp_http/dynatrace]", 0) >= 9
            and not any(k.startswith("otelcol_exporter_send_failed") and v for k, v in sent.items())
        )
        checks["collector: Arize got exactly the spans it stored"] = sent.get("otelcol_exporter_sent_spans[otlp_grpc/arize]", 0) == arize["span_count"]

    result = {"scenario": name, "mode": mode, "collector": distribution, "trace_id": trace_id, "answer": app["answer"],
              "checks": checks, "collector_self_metrics_delta": sent, "arize": arize, "dynatrace": dynatrace}
    OUT.mkdir(exist_ok=True)
    (OUT / f"results-{name}.json").write_text(json.dumps(result, indent=2))
    for check, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    passed = sum(checks.values())
    print(f"  {passed}/{len(checks)} checks passed" + (f" ({distribution})" if distribution else ""))
    print(f"  arize spans: {arize['spans']} roots={arize['roots']}")
    print(f"  dynatrace spans: {dynatrace['spans']} roots={dynatrace['roots']}")
    print(f"  dynatrace logs: {dynatrace['log_count']}, metrics: {dynatrace['metrics_last_30m']}")
    if sent:
        print(f"  collector self-metrics: {sent}")
    print(f'  DQL: fetch spans | filter trace.id == toUid("{trace_id}")')
    return passed == len(checks)


def main() -> None:
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario {unknown}; choose from {', '.join(SCENARIOS)} or all")
    results = {name: verify(name) for name in names}
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
