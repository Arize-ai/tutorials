"""Build station: spec -> code + tests in a worktree -> pushed branch -> PR."""

from __future__ import annotations

from .. import config, gh
from . import _agent

SCHEMA = {
    "type": "object",
    "properties": {
        "commit_message": {"type": "string"},
        "pr_title": {"type": "string"},
        "pr_body": {"type": "string"},
        "local_ci_passed": {"type": "boolean"},
        "tests_added": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["commit_message", "pr_title", "pr_body", "local_ci_passed"],
}

TRAILER = "Co-Authored-By: Codex <noreply@openai.com>"


async def run(ctx) -> dict:
    issue, spec = ctx.issue, ctx.spec
    branch = f"factory/issue-{issue['number']}"
    wt = gh.worktree(branch)
    gh.sh("uv", "sync", "-q", cwd=wt)

    res = await _agent.run(
        ctx,
        "build",
        "build",
        prompt=(
            f"Implement this spec for issue #{issue['number']} ({issue['title']}).\n\n"
            f"{spec['markdown']}\n\nOriginal issue text:\n{issue.get('body') or '(none)'}"
        ),
        cwd=wt,
        model=config.BUILD_MODEL,
        sandbox="workspace_write",
        network=True,  # `uv add` needs PyPI
        schema=SCHEMA,
        max_tokens=config.BUILD_MAX_TOKENS,
    )
    out = res.structured
    if not isinstance(out, dict):
        raise TypeError(f"builder returned no structured result ({_agent.why(res)})")

    stat = gh.diff_stat(wt)
    if not stat:
        raise RuntimeError("builder made no changes")

    subject_body = "\n".join(
        line
        for line in out["commit_message"].strip().splitlines()
        if not line.lower().startswith("co-authored-by:")
    ).strip()
    message = f"{subject_body}\n\nRefs #{issue['number']}\n\n{TRAILER}\n"
    sha = gh.commit_and_push(wt, branch, message)

    body = (
        f"{out['pr_body'].strip()}\n\nCloses #{issue['number']}\n\n"
        f"<details><summary>Factory spec</summary>\n\n{spec['markdown']}\n\n</details>\n\n"
        f"<details><summary>Builder notes</summary>\n\n{out.get('notes') or '(none)'}\n\n"
        f"Local CI passed: {out['local_ci_passed']}\n\n</details>\n\n"
        f"<!-- factory run={ctx.run_id} trace_id={ctx.trace_id} -->\n"
        "🏭 Built by the software factory."
    )
    pr = gh.open_pr(branch, out["pr_title"], body, labels=["factory"])
    return {
        **out,
        **pr,
        "head_sha": sha,
        "diff_stat": stat,
        "branch": branch,
        "tokens": res.tokens,
        "thread_id": res.thread_id,
    }
