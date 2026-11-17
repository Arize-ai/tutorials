"""Review station: three reviewers (correctness, security, tests) in parallel.

Each returns a structured verdict. These verdicts are what the course evaluates: later,
the harvester compares them to your verdict at the review gate on the same PR.
"""

from __future__ import annotations

from .. import config, prompts
from . import _agent

KINDS = ["correctness", "security", "tests"]

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "request_changes"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["blocker", "major", "minor", "nit"]},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["severity", "file", "title", "body"],
            },
        },
    },
    "required": ["verdict", "confidence", "summary", "findings"],
}

ICONS = {"blocker": "🛑", "major": "🔴", "minor": "🟡", "nit": "⚪"}


def system_prompt(kind: str) -> str:
    """Local fallback text; the live prompt comes from Prompt Hub (factory.prompts)."""
    return prompts.local_text(f"review_{kind}")


async def run_one(ctx, kind: str, diff: str) -> dict:
    station = f"review_{kind}"
    res = await _agent.run(
        ctx,
        station,
        station,
        prompt=(
            f"Review PR #{ctx.pr['number']} ({ctx.pr.get('title') or ctx.issue['title']}) for "
            f"issue #{ctx.number}. Head commit {ctx.pr['head_sha'][:7]}. CI is green.\n\n"
            f"## Spec\n\n{ctx.spec['markdown']}\n\n## Diff against origin/main\n\n"
            f"```diff\n{diff}\n```"
        ),
        cwd=ctx.wt,
        model=config.REVIEW_MODEL,
        sandbox="read_only",
        schema=SCHEMA,
        max_tokens=config.REVIEW_MAX_TOKENS,
    )
    if not isinstance(res.structured, dict):
        raise TypeError(f"{kind} reviewer returned no verdict ({_agent.why(res)})")
    out = dict(res.structured)
    # The verdict rule is mechanical, so enforce it rather than trust it.
    serious = [f for f in out["findings"] if f["severity"] in ("blocker", "major")]
    out["model_verdict"] = out["verdict"]
    out["verdict"] = "request_changes" if serious else "approve"
    out["kind"] = kind
    out["tokens"] = res.tokens
    out["thread_id"] = res.thread_id
    out["head_sha"] = ctx.pr["head_sha"]
    return out


def render(r: dict) -> str:
    head = "✅ approve" if r["verdict"] == "approve" else "❌ request changes"
    meta = f"<sub>confidence {r['confidence']:.2f} · {config.REVIEW_MODEL} · commit {r['head_sha'][:7]}</sub>"
    lines = [f"### 🤖 {r['kind'].title()} reviewer: {head}", meta, "", r["summary"], ""]
    for f in r["findings"]:
        loc = f"`{f['file']}" + (f":{f['line']}`" if f.get("line") else "`")
        lines += [
            f"- {ICONS.get(f['severity'], '')} **{f['severity']}** {loc}: **{f['title']}**",
            f"  {f['body'].strip()}",
        ]
    if not r["findings"]:
        lines.append("_No findings._")
    return "\n".join(lines)
