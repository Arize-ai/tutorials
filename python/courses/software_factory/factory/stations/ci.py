"""CI station: wait for GitHub Actions on the PR head. Deterministic, no agent.

CI is the cheapest, most objective gate in the factory: its verdict on each head SHA
is a free label on the builder's (or babysitter's) work.
"""

from __future__ import annotations

import asyncio
import time

from .. import config, gh


class Conflicting(Exception):
    """The PR conflicts with main, so GitHub will never run CI on it."""


async def wait(sha: str, timeout_s: int | None = None, pr: int | None = None) -> tuple[bool, str]:
    """Return (passed, summary) once every check run on sha has completed.

    Raises Conflicting if no checks have started after a minute and the PR conflicts
    with main (main moved on while this PR waited)."""
    start = time.monotonic()
    deadline = start + (timeout_s or config.CI_TIMEOUT_S)
    seen_any = checked = False
    while time.monotonic() < deadline:
        runs = await asyncio.to_thread(gh.check_runs, sha)
        if not runs and pr and not checked and time.monotonic() - start > 60:
            checked = True
            if await asyncio.to_thread(gh.mergeable, pr) == "CONFLICTING":
                raise Conflicting(f"PR #{pr} conflicts with main")
        if runs:
            seen_any = True
            if all(r["status"] == "completed" for r in runs):
                bad = [r for r in runs if r["conclusion"] not in ("success", "skipped", "neutral")]
                if bad:
                    return False, ", ".join(f"{r['name']}: {r['conclusion']}" for r in bad)
                return True, f"{len(runs)} check{'s' if len(runs) != 1 else ''} green"
        await asyncio.sleep(10 if seen_any else 5)
    raise TimeoutError(f"CI did not finish on {sha[:7]} within {timeout_s or config.CI_TIMEOUT_S}s")
