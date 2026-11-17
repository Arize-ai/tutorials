"""Spec station: issue -> acceptance criteria, posted back to the issue."""

from __future__ import annotations

from .. import config, gh
from . import _agent

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "out_of_scope": {"type": "array", "items": {"type": "string"}},
        "files_likely_touched": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "risk_reason": {"type": "string"},
    },
    "required": ["summary", "acceptance_criteria", "risk", "risk_reason"],
}


def render(spec: dict) -> str:
    def section(title: str, items: list[str] | None) -> list[str]:
        return [f"**{title}**", *[f"- {i}" for i in items], ""] if items else []

    lines = ["### 🏭 Factory spec", "", spec["summary"], ""]
    lines += section("Acceptance criteria", [f"[ ] {c}" for c in spec["acceptance_criteria"]])
    lines += section("Assumptions", spec.get("assumptions"))
    lines += section("Out of scope", spec.get("out_of_scope"))
    lines += section("Likely files", [f"`{f}`" for f in spec.get("files_likely_touched", [])])
    lines += section("Questions for a human", spec.get("questions"))
    lines += [f"**Risk:** {spec['risk']} ({spec['risk_reason']})"]
    return "\n".join(lines)


async def run(ctx) -> dict:
    clone = gh.ensure_clone()
    issue = ctx.issue
    res = await _agent.run(
        ctx,
        "spec",
        "spec",
        prompt=(
            f"Issue #{issue['number']}: {issue['title']}\n\n{issue.get('body') or '(no body)'}"
            "\n\nWrite the spec for this issue."
        ),
        cwd=clone,
        model=config.BUILD_MODEL,
        sandbox="read_only",
        schema=SCHEMA,
        max_tokens=config.SPEC_MAX_TOKENS,
    )
    if not isinstance(res.structured, dict):
        raise TypeError(f"spec agent returned no structured spec ({_agent.why(res)})")
    spec = res.structured
    spec["markdown"] = render(spec)
    spec["tokens"] = res.tokens
    gh.comment_issue(issue["number"], spec["markdown"])
    return spec
