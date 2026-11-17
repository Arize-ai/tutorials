"""Release station: smoke-test the merged commit, pick a version, publish a release.

The agent tests and writes notes; the factory computes the version and creates the
GitHub release, so tags are never invented by a model.
"""

from __future__ import annotations

import re

from .. import config, gh
from . import _agent

SCHEMA = {
    "type": "object",
    "properties": {
        "smoke_passed": {"type": "boolean"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["command", "passed"],
            },
        },
        "bump": {"type": "string", "enum": ["major", "minor", "patch"]},
        "release_notes": {"type": "string"},
    },
    "required": ["smoke_passed", "checks", "bump", "release_notes"],
}


def next_version(current: str | None, bump: str) -> str:
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", current or "")
    if not m:
        return "v0.1.0"
    major, minor, patch = map(int, m.groups())
    if bump == "major":
        return f"v{major + 1}.0.0"
    if bump == "minor":
        return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"


async def run(ctx, merge_sha: str) -> dict:
    gh.ensure_clone()
    wt = gh.detached_worktree(f"release-{ctx.number}", merge_sha or "origin/main")
    gh.sh("uv", "sync", "-q", cwd=wt)
    out_png = wt / "smoke-card.png"
    res = await _agent.run(
        ctx,
        "release",
        "release",
        prompt=(
            f'Merged: PR #{ctx.pr["number"]} "{ctx.pr.get("title") or ctx.issue["title"]}" '
            f"(commit {merge_sha[:7] if merge_sha else 'HEAD'}), closing issue #{ctx.number}.\n\n"
            f"PR description:\n{(ctx.pr.get('body') or '')[:4000]}\n\n"
            f"Smoke-test PR URL: {config.SMOKE_PR_URL}\nOutput path for the card: {out_png}"
        ),
        cwd=wt,
        model=config.BUILD_MODEL,
        sandbox="workspace_write",
        network=True,  # the smoke test fetches a public PR from GitHub
        schema=SCHEMA,
        max_tokens=config.RELEASE_MAX_TOKENS,
    )
    if not isinstance(res.structured, dict):
        raise TypeError(f"release agent returned no result ({_agent.why(res)})")
    out = dict(res.structured)
    # Trust, but verify the one check we can see from here.
    if out["smoke_passed"] and not out_png.exists():
        out["smoke_passed"] = False
        out["checks"].append(
            {
                "command": "factory: card PNG exists",
                "passed": False,
                "note": f"{out_png.name} was not written",
            }
        )
    out["tokens"] = res.tokens
    if not out["smoke_passed"]:
        return out
    current = gh.latest_release_tag()
    version = next_version(current, out["bump"])
    notes = (
        f"{out['release_notes'].strip()}\n\n---\nShipped by the software factory from "
        f"#{ctx.pr['number']} (closes #{ctx.number})."
    )
    out["version"] = version
    out["previous"] = current
    out["release_url"] = gh.create_release(version, merge_sha, version, notes)
    return out
