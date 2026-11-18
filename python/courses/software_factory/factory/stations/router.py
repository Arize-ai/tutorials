"""Router station: auto-merge or the review gate? An agent decides, over a rules floor.

The rules are a hard floor: if any rule fires, the PR goes to the review gate whatever
the agent says. Recording both lets you measure how often the agent alone would have
auto-merged something the rules (or later, you) caught.
"""

from __future__ import annotations

import json

from .. import config, gh
from . import _agent

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["automerge", "human"]},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["route", "risk", "reasons"],
}


def evidence(ctx) -> dict:
    files = gh.changed_files(ctx.wt)
    return {
        "spec_risk": ctx.spec["risk"],
        "spec_risk_reason": ctx.spec.get("risk_reason", ""),
        "files_changed": files,
        "diff_lines": gh.diff_lines(ctx.wt),
        "tests_delta": ctx.tests_delta,
        "ci_runs": ctx.ci_runs,
        "ci_fixes": ctx.ci_fixes,
        "reviews": {
            k: {
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "summary": r["summary"],
                "findings": [f"{f['severity']}: {f['title']}" for f in r["findings"]],
            }
            for k, r in ctx.reviews.items()
        },
    }


def rules(ev: dict) -> list[str]:
    hits = []
    flagged = [k for k, r in ev["reviews"].items() if r["verdict"] != "approve"]
    if flagged:
        hits.append(f"reviewer requested changes: {', '.join(flagged)}")
    unsure = [k for k, r in ev["reviews"].items() if r["confidence"] < 0.6]
    if unsure:
        hits.append(f"low reviewer confidence: {', '.join(unsure)}")
    if ev["spec_risk"] == "high":
        hits.append("spec marked high risk")
    if ev["tests_delta"] < 0:
        hits.append(f"{-ev['tests_delta']} test(s) removed")
    if ev["ci_fixes"] >= 2:
        hits.append(f"needed {ev['ci_fixes']} CI fixes")
    if ev["diff_lines"] > config.BIG_DIFF_LINES:
        hits.append(f"large diff ({ev['diff_lines']} lines)")
    touchy = [f for f in ev["files_changed"] if any(f.startswith(p) for p in config.SENSITIVE_PATHS)]
    if touchy:
        hits.append(f"sensitive files: {', '.join(touchy)}")
    return hits


async def run(ctx, ev: dict) -> dict:
    res = await _agent.run(
        ctx,
        "router",
        "router",
        prompt=(
            f"PR #{ctx.pr['number']}: {ctx.pr.get('title') or ctx.issue['title']}\n\n"
            f"Evidence:\n```json\n{json.dumps(ev, indent=2)}\n```"
        ),
        cwd=ctx.wt,
        model=config.BUILD_MODEL,
        sandbox="read_only",
        schema=SCHEMA,
        max_tokens=config.ROUTER_MAX_TOKENS,
    )
    agent = (
        res.structured
        if isinstance(res.structured, dict)
        else {"route": "human", "risk": "high", "reasons": [f"router agent failed ({_agent.why(res)})"]}
    )
    floor = rules(ev)
    route = "human" if floor or agent["route"] == "human" else "automerge"
    return {
        "route": route,
        "agent_route": agent["route"],
        "agent_risk": agent["risk"],
        "agent_reasons": agent["reasons"],
        "rule_hits": floor,
        "tokens": res.tokens,
    }
