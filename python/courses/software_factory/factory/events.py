"""Factory event log: append-only JSONL in runs/events.jsonl.

Every station calls emit() when it starts, finishes, fails or waits, and each agent
tool call is logged as an info event. `tail -f runs/events.jsonl` follows a run live.
Events are deliberately tiny and flat so they can also be joined to Arize traces
later via trace_id.

Schema:
    ts        ISO-8601 UTC timestamp
    run       run id, e.g. "issue-12"
    issue     GitHub issue number
    title     issue title (sent on the first event of a run; optional after)
    station   issue | spec | build | ci | babysit | review_correctness |
              review_security | review_tests | router | automerge | human |
              release | done
    status    start | ok | fail | wait | info
    detail    short human-readable text
    trace_id  optional Arize trace id for deep links
    pr        optional PR number
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

EVENTS_PATH = Path(
    os.environ.get("FACTORY_EVENTS", Path(__file__).resolve().parent.parent / "runs" / "events.jsonl")
)
_lock = threading.Lock()

STATIONS = [
    "issue",
    "spec",
    "build",
    "ci",
    "babysit",
    "review_correctness",
    "review_security",
    "review_tests",
    "router",
    "automerge",
    "human",
    "release",
    "done",
]
STATUSES = {"start", "ok", "fail", "wait", "info"}


def emit(run: str, station: str, status: str, detail: str = "", **extra) -> dict:
    if station not in STATIONS:
        raise ValueError(f"unknown station {station!r}")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    event = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "run": run,
        "station": station,
        "status": status,
        "detail": detail,
        **{k: v for k, v in extra.items() if v is not None},
    }
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event
