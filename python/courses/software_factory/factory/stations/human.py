"""Review gate: you review the PR on GitHub. Deterministic, no agent.

The factory acts on GitHub as you, and GitHub won't let you approve or request
changes on your own PR. So your verdict is a comment (or COMMENT review) that
starts with a command:

    /ship                    merge it
    /changes <what to fix>   send it back to the babysitter (inline review comments
                             posted since the gate opened are included too)

Merging or closing the PR by hand also counts.
"""

from __future__ import annotations

import functools

from .. import gh

INSTRUCTIONS = """### 🧑‍💻 Over to you

The risk router sent this PR to the review gate.

{reasons}

Reply with a comment starting with **`/ship`** to merge and release, or **`/changes`** followed
by what to fix to send it back to the factory. Inline review comments count too. Merging or
closing the PR by hand also works."""


@functools.cache
def me() -> str:
    return gh.sh("gh", "api", "user", "--jq", ".login")


def verdict(pr_number: int, since: str) -> tuple[str | None, str]:
    """(None|'ship'|'changes'|'merged'|'closed', feedback text)."""
    state = gh.pr(pr_number)["state"]
    if state == "MERGED":
        return "merged", "merged by hand"
    if state == "CLOSED":
        return "closed", "closed by hand"
    convo = [c for c in gh.pr_conversation(pr_number) if c["at"] >= since and c["author"] == me()]
    for i, c in enumerate(convo):
        body = c["body"].strip()
        if body.startswith("/ship"):
            return "ship", body[len("/ship") :].strip()
        if body.startswith("/changes"):
            notes = [body[len("/changes") :].strip()]
            notes += [x["body"] for x in convo[:i] if x["kind"] == "line"]
            return "changes", "\n\n".join(n for n in notes if n) or "(no details given)"
    return None, ""
