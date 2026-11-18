"""Run the software factory on GitHub issues.

    python -m factory.run 7                  # one issue, end to end
    python -m factory.run 7 12 3 -p 3        # several, up to 3 at once
    python -m factory.run 1 --from ci        # adopt an existing factory PR (skip spec+build)
    python -m factory.run --watch            # resume runs parked at the review gate
    python -m factory.run --release 7        # re-run only the release step
    python -m factory.run --retry 7          # resume a failed run from where it stopped

Stations: spec → build → CI (⟲ babysit) → 3 reviewers → router → auto-merge | review gate → release.
One issue = one factory trace in Arize AX, plus one Codex trace per agent turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time

from . import config, gh, tracing
from .context import Ctx, utcnow
from .stations import babysit, build, ci, human, release, review, router, spec

log = logging.getLogger("factory")
_release_lock = asyncio.Lock()  # versions are computed from the latest tag: one at a time


# ---------- stations ----------


async def do_spec(ctx: Ctx) -> None:
    with ctx.station(
        "spec", detail="writing acceptance criteria", model=config.BUILD_MODEL, input=ctx.issue["title"]
    ) as s:
        ctx.spec = await spec.run(ctx)
        tracing.set_output(s, ctx.spec["markdown"])
        s.set_attribute("factory.risk", ctx.spec["risk"])
    ctx.emit("spec", "ok", f"{len(ctx.spec['acceptance_criteria'])} criteria, risk {ctx.spec['risk']}")


async def do_build(ctx: Ctx) -> None:
    with ctx.station(
        "build",
        detail="implementing in a worktree",
        model=config.BUILD_MODEL,
        input=ctx.spec["markdown"],
        spec_risk=ctx.spec["risk"],
    ) as s:
        try:
            pr = await build.run(ctx)
        except build.Blocked as e:
            # A handoff, not an error: record it on the span, then raise it once the span
            # has closed so the station doesn't log it as a failure.
            s.set_attribute("factory.blocked", True)
            tracing.set_output(s, {"blocked": e.reasons})
            blocked = e
        else:
            blocked = None
            ctx.pr = {
                "number": pr["number"],
                "url": pr["url"],
                "branch": pr["branch"],
                "head_sha": pr["head_sha"],
                "title": pr["pr_title"],
                "body": pr["pr_body"],
            }
            ctx.worktree = str(gh.worktree_for(pr["branch"]))
            tracing.set_output(
                s, {k: pr[k] for k in ("url", "pr_title", "diff_stat", "local_ci_passed", "notes") if k in pr}
            )
            s.set_attribute("factory.pr", pr["number"])
            s.set_attribute("factory.head_sha", pr["head_sha"])
    if blocked:
        raise blocked
    ctx.emit("build", "ok", f"PR #{pr['number']} opened")


def adopt(ctx: Ctx) -> None:
    """--from ci: pick up the existing factory PR and spec for this issue."""
    md = gh.latest_spec_comment(ctx.number)
    if not md:
        raise RuntimeError(f"no factory spec comment on issue #{ctx.number}")
    risk = (re.search(r"\*\*Risk:\*\* (low|medium|high)", md) or [None, "medium"])[1]
    ctx.spec = {"markdown": md, "risk": risk, "acceptance_criteria": [], "tokens": 0}
    branch = f"factory/issue-{ctx.number}"
    found = json.loads(gh.gh("pr", "list", "--head", branch, "--state", "open", "--json", "number"))
    if not found:
        raise RuntimeError(f"no open PR for {branch}")
    p = gh.pr(found[0]["number"])
    ctx.pr = {
        "number": p["number"],
        "url": p["url"],
        "branch": branch,
        "head_sha": p["headRefOid"],
        "title": p["title"],
        "body": p["body"],
    }
    ctx.worktree = str(gh.worktree_for(branch))
    ctx.emit("spec", "ok", f"reused spec (risk {risk})")
    ctx.emit("build", "ok", f"adopted PR #{p['number']}")


async def ci_loop(ctx: Ctx) -> bool:
    """CI, and on red, babysit and retry. True when green."""
    while True:
        if not await resolve_conflicts(ctx):
            return False
        ctx.ci_runs += 1
        sha = ctx.pr["head_sha"]
        with ctx.station(
            "ci", detail=f"run {ctx.ci_runs} on {sha[:7]}", head_sha=sha, ci_run=ctx.ci_runs
        ) as s:
            try:
                passed, summary = await ci.wait(sha, pr=ctx.pr["number"])
            except ci.Conflicting as e:
                tracing.set_output(s, str(e))
                ctx.emit("ci", "info", "main moved on: merging main first")
                continue
            s.set_attribute("factory.ci_passed", passed)
            tracing.set_output(s, summary)
        ctx.emit("ci", "ok" if passed else "fail", summary, ci_run=ctx.ci_runs)
        if passed:
            return True
        if ctx.ci_fixes >= config.MAX_CI_FIXES:
            return False
        ctx.ci_fixes += 1
        log_text = await asyncio.to_thread(gh.failed_ci_log, sha)
        if not await do_babysit(ctx, "ci", log_text):
            return False


async def resolve_conflicts(ctx: Ctx) -> bool:
    """If main has moved on and the PR conflicts, merge main (babysitter resolves). True if OK."""
    if await asyncio.to_thread(gh.mergeable, ctx.pr["number"]) != "CONFLICTING":
        return True
    return await do_babysit(ctx, "conflict", "PR conflicts with main")


async def do_babysit(ctx: Ctx, reason: str, evidence: str) -> bool:
    detail = {"ci": "reading CI logs", "human": "addressing your review", "conflict": "merging main"}[reason]
    with ctx.station(
        "babysit", detail=detail, model=config.BUILD_MODEL, input=evidence[-4000:], reason=reason
    ) as s:
        out = await babysit.run(ctx, reason=reason, evidence=evidence)
        tracing.set_output(s, {k: out.get(k) for k in ("fixed", "pushed", "root_cause", "notes")})
        s.set_attribute("factory.fixed", bool(out.get("pushed")))
    if out.get("pushed"):
        ctx.emit("babysit", "ok", f"pushed {out['head_sha'][:7]}: {out['root_cause'][:100]}")
        return True
    ctx.emit("babysit", "fail", f"couldn't fix: {out.get('root_cause', '')[:120]}")
    return False


async def do_reviews(ctx: Ctx) -> None:
    diff = gh.pr_diff(ctx.wt)
    # Compare with where the branch left main, not current main: tests that other PRs
    # merged since then are not "removed" by this PR.
    base = gh.sh("git", "merge-base", "HEAD", "origin/main", cwd=ctx.wt)
    ctx.tests_delta = gh.test_count(ctx.wt, "HEAD") - gh.test_count(ctx.wt, base)

    async def one(kind: str) -> dict:
        station = f"review_{kind}"
        with ctx.station(
            station, detail=f"{kind} review", model=config.REVIEW_MODEL, input=diff[:8000], reviewer=kind
        ) as s:
            r = await review.run_one(ctx, kind, diff)
            tracing.set_output(s, {k: r[k] for k in ("verdict", "confidence", "summary", "findings")})
            s.set_attribute("factory.verdict", r["verdict"])
            s.set_attribute("factory.confidence", float(r["confidence"]))
            s.set_attribute("factory.findings", len(r["findings"]))
        n = len(r["findings"])
        ctx.emit(
            station,
            "ok" if r["verdict"] == "approve" else "fail",
            f"{'approve' if r['verdict'] == 'approve' else 'request changes'}"
            f" · {n} finding{'s' if n != 1 else ''}",
            verdict=r["verdict"],
        )
        return r

    with ctx.station("review", emit_start=False, input=diff[:8000], tests_delta=ctx.tests_delta) as s:
        results = await asyncio.gather(*(one(k) for k in review.KINDS))
        ctx.reviews = {r["kind"]: r for r in results}
        tracing.set_output(s, {k: r["verdict"] for k, r in ctx.reviews.items()})
    for r in results:
        gh.review_comment(ctx.pr["number"], review.render(r))


async def do_router(ctx: Ctx) -> str:
    ev = router.evidence(ctx)
    with ctx.station("router", detail="weighing the evidence", model=config.BUILD_MODEL, input=ev) as s:
        r = await router.run(ctx, ev)
        tracing.set_output(s, r)
        s.set_attribute("factory.route", r["route"])
        s.set_attribute("factory.agent_route", r["agent_route"])
    ctx.route = r
    why = (r["rule_hits"] or r["agent_reasons"] or ["all clear"])[0]
    ctx.emit(
        "router",
        "ok",
        f"→ {'auto-merge' if r['route'] == 'automerge' else 'review gate'}: {why[:110]}",
        route=r["route"],
    )
    return r["route"]


async def do_merge(ctx: Ctx, station: str = "automerge") -> str | None:
    # Another PR may have merged while this one waited: bring it up to date and re-run CI.
    conflicting = await asyncio.to_thread(gh.mergeable, ctx.pr["number"]) == "CONFLICTING"
    if conflicting and not await ci_loop(ctx):
        return None
    with ctx.station("automerge", detail="squash-merging", emit_start=station == "automerge") as s:
        try:
            sha = gh.merge_pr(ctx.pr["number"])
        except RuntimeError as e:
            s.set_attribute("factory.merged", False)
            tracing.set_output(s, str(e)[:500])
            if station == "automerge":
                ctx.emit("automerge", "fail", "merge failed, sending to the review gate")
            return None
        s.set_attribute("factory.merged", True)
        s.set_attribute("factory.merge_sha", sha)
    if station == "automerge":
        ctx.emit("automerge", "ok", f"merged as {sha[:7]}")
    ctx.status = "merged"
    return sha


async def do_release(ctx: Ctx, merge_sha: str) -> None:
    async with _release_lock:
        with ctx.station(
            "release", detail="smoke test + release notes", model=config.BUILD_MODEL, merge_sha=merge_sha
        ) as s:
            out = await release.run(ctx, merge_sha)
            tracing.set_output(
                s, {k: out.get(k) for k in ("smoke_passed", "checks", "bump", "version", "release_url")}
            )
            s.set_attribute("factory.smoke_passed", out["smoke_passed"])
    if out["smoke_passed"]:
        ctx.status = "released"
        ctx.emit(
            "release",
            "ok",
            f"{out['version']} published",
            version=out["version"],
            release_url=out.get("release_url"),
        )
        finish(ctx, f"merged + released {out['version']}")
    else:
        failed_checks = [c for c in out["checks"] if not c["passed"]]
        failed = [c["command"] for c in failed_checks]
        lines = [
            f"- `{c['command']}`" + (f": {c['note'].strip()}" if c.get("note") else "") for c in failed_checks
        ]
        gh.comment_pr(
            ctx.pr["number"],
            "### 🚨 Release smoke test failed after merge\n\n"
            + "\n".join(lines or ["- (the release agent reported a failure but named no check)"])
            + "\n\nNo release was created. Rerun it with `python -m factory.run --release "
            + f"{ctx.number}`.",
        )
        ctx.emit("release", "fail", f"smoke failed: {', '.join(failed)[:120]}")
        finish(ctx, "merged, smoke test failed", status="failed")


async def open_human_gate(ctx: Ctx, reasons: list[str]) -> None:
    with ctx.station("human", emit_start=False, reasons=" | ".join(reasons)):
        gh.label_pr(ctx.pr["number"], add=["needs-human"])
        bullet = "\n".join(f"- {r}" for r in reasons) or "- (no reasons given)"
        gh.comment_pr(ctx.pr["number"], human.INSTRUCTIONS.format(reasons=bullet))
    ctx.status = "awaiting_human"
    ctx.gate_opened_at = utcnow()
    ctx.emit("human", "wait", f"waiting at the review gate: comment /ship or /changes on {ctx.pr['url']}")
    ctx.save()


BLOCKED_COMMENT = """### 🏭 The factory stopped before building this

The builder made no changes. Its reasons:

{reasons}

A human needs to decide before this can be built. Update the issue with the decision,
remove the `needs-human` label, and run the factory on it again.
"""


def block_on_human(ctx: Ctx, e: build.Blocked) -> None:
    """The builder declined: hand the issue to a human instead of failing the run."""
    ctx.tokens += e.tokens
    gh.comment_issue(ctx.number, BLOCKED_COMMENT.format(reasons=e.reasons))
    gh.label_issue(ctx.number, add=["needs-human"])
    ctx.emit("build", "wait", f"blocked, needs a human: {e.reasons[:140]}", reasons=e.reasons)
    finish(ctx, "blocked: the builder left a decision for a human", status="blocked")


def finish(ctx: Ctx, outcome: str, status: str | None = None) -> None:
    ctx.outcome = outcome
    if status:
        ctx.status = status
    ctx.emit("done", "info", outcome)
    ctx.save()


# ---------- pipeline ----------


async def after_build(ctx: Ctx) -> None:
    if not await ci_loop(ctx):
        gh.label_pr(ctx.pr["number"], add=["factory-stuck"])
        return finish(ctx, f"CI still red after {ctx.ci_fixes} fixes", status="failed")
    await do_reviews(ctx)
    if await do_router(ctx) == "automerge":
        sha = await do_merge(ctx)
        if sha is not None:
            return await do_release(ctx, sha)
        return await open_human_gate(ctx, ["auto-merge failed (likely a conflict with main)"])
    reasons = ctx.route["rule_hits"] + [f"router: {r}" for r in ctx.route["agent_reasons"]]
    await open_human_gate(ctx, reasons)
    if config.HUMAN_WAIT_S:
        await watch_one(ctx, deadline=time.monotonic() + config.HUMAN_WAIT_S)


async def run_issue(number: int, attempt: int = 1, start: str = "spec") -> Ctx:
    ctx = Ctx(issue=gh.issue(number), attempt=attempt)
    with tracing.span(
        "factory.run",
        session_id=ctx.session_id,
        metadata=ctx.base("run"),
        input=f"#{number} {ctx.issue['title']}\n\n{ctx.issue.get('body') or ''}",
    ) as root:
        ctx.trace_id = tracing.trace_id_of(root)
        ctx.root_span_id = tracing.span_id_of(root)
        now_ms = int(time.time() * 1000)
        ctx.trace_url = tracing.trace_url(ctx.trace_id, now_ms - 3_600_000, now_ms + 24 * 3_600_000)
        ctx.emit(
            "issue",
            "ok",
            f"#{number} picked up (trace {ctx.trace_id})",
            title=ctx.issue["title"],
            trace_url=ctx.trace_url,
        )
        try:
            if start == "spec":
                await do_spec(ctx)
                await do_build(ctx)
            else:
                adopt(ctx)
            await after_build(ctx)
        except build.Blocked as e:
            block_on_human(ctx, e)
        except Exception as e:
            finish(ctx, f"failed: {str(e)[:160]}", status="failed")
            raise
        finally:
            stamp(root, ctx)
    return ctx


def stamp(span, ctx: Ctx) -> None:
    if ctx.pr:
        span.set_attribute("factory.pr", ctx.pr["number"])
        span.set_attribute("factory.head_sha", ctx.pr["head_sha"])
    span.set_attribute("factory.tokens", ctx.tokens)
    span.set_attribute("factory.duration_s", round(time.time() - ctx.started, 1))
    span.set_attribute("factory.status", ctx.status)
    span.set_attribute("factory.ci_runs", ctx.ci_runs)
    span.set_attribute("factory.ci_fixes", ctx.ci_fixes)
    if ctx.route:
        span.set_attribute("factory.route", ctx.route["route"])
    for k, r in ctx.reviews.items():
        span.set_attribute(f"factory.review_{k}", r["verdict"])
    tracing.set_output(span, {"status": ctx.status, "outcome": ctx.outcome, "pr": (ctx.pr or {}).get("url")})


# ---------- review gate resume ----------


async def watch_one(ctx: Ctx, deadline: float | None = None) -> bool:
    """Poll one parked run until the gate decides (or the deadline). True if it moved on."""
    while True:
        v, feedback = await asyncio.to_thread(human.verdict, ctx.pr["number"], ctx.gate_opened_at)
        if v:
            await resume(ctx, v, feedback)
            return True
        if deadline is None or time.monotonic() >= deadline:
            return False
        await asyncio.sleep(config.POLL_S)


async def resume(ctx: Ctx, v: str, feedback: str) -> None:
    ctx.status = "running"
    with tracing.span(
        "factory.resume",
        session_id=ctx.session_id,
        metadata=ctx.base("resume"),
        input=f"{v}: {feedback}",
        parent=(ctx.trace_id, ctx.root_span_id),
    ) as root:
        try:
            with ctx.station("human", emit_start=False, human_verdict=v) as s:
                tracing.set_output(s, feedback)
                s.set_attribute("factory.human_verdict", v)
            gh.label_pr(ctx.pr["number"], remove=["needs-human"])
            if v in ("ship", "merged"):
                ctx.emit("human", "ok", "merged by hand" if v == "merged" else "/ship")
                if v == "merged":
                    sha = (gh.pr(ctx.pr["number"]).get("mergeCommit") or {}).get("oid")
                else:
                    sha = await do_merge(ctx, station="human")
                if not sha:
                    return finish(ctx, "approved but merge failed", status="failed")
                ctx.status = "merged"
                await do_release(ctx, sha)
            elif v == "closed":
                ctx.emit("human", "fail", "closed by hand")
                finish(ctx, "closed at the review gate", status="stopped")
            else:
                ctx.emit("human", "fail", f"/changes: {feedback[:120]}")
                if ctx.reworks >= config.MAX_REWORKS:
                    return finish(ctx, f"stopped after {ctx.reworks} reworks", status="stopped")
                ctx.reworks += 1
                if not await do_babysit(ctx, "human", feedback):
                    return finish(ctx, "rework failed", status="failed")
                if not await ci_loop(ctx):
                    return finish(ctx, "CI red after rework", status="failed")
                await open_human_gate(ctx, [f"rework {ctx.reworks} after your /changes"])
        except Exception as e:
            finish(ctx, f"failed: {str(e)[:160]}", status="failed")
            raise
        finally:
            stamp(root, ctx)
            ctx.save()


async def rerelease(number: int) -> None:
    """Re-run only the release step for a run that merged but didn't release."""
    ctx = Ctx.load(number)
    sha = (gh.pr(ctx.pr["number"]).get("mergeCommit") or {}).get("oid")
    if not sha:
        raise SystemExit(f"issue #{number}: PR #{ctx.pr['number']} is not merged")
    ctx.status = "merged"
    with tracing.span(
        "factory.resume",
        session_id=ctx.session_id,
        metadata=ctx.base("release"),
        input=f"re-release {sha[:7]}",
        parent=(ctx.trace_id, ctx.root_span_id),
    ) as root:
        try:
            await do_release(ctx, sha)
        finally:
            stamp(root, ctx)
            ctx.save()


async def retry(number: int) -> None:
    """Resume a failed run from the step it was on, in the same trace. For runs that died
    partway, for example when the agents ran out of credits."""
    ctx = Ctx.load(number)
    if ctx.status != "failed":
        raise SystemExit(f"issue #{number} is {ctx.status}, not failed: nothing to retry")
    if not ctx.pr:
        raise SystemExit(f"issue #{number} failed before it had a PR: run it again from the start")
    ctx.status = "running"
    with tracing.span(
        "factory.resume",
        session_id=ctx.session_id,
        metadata=ctx.base("retry"),
        input=f"retry after: {ctx.outcome}",
        parent=(ctx.trace_id, ctx.root_span_id),
    ) as root:
        try:
            verdict = None
            if ctx.gate_opened_at:
                verdict, _ = await asyncio.to_thread(human.verdict, ctx.pr["number"], ctx.gate_opened_at)
            if verdict == "ship":
                # You already said /ship: merge (resolving any conflict with main) and release.
                sha = await do_merge(ctx, station="human")
                if not sha:
                    return finish(ctx, "approved but merge failed", status="failed")
                ctx.status = "merged"
                await do_release(ctx, sha)
            elif ctx.reworks:
                # A /changes rework was pushed: finish CI and hand the PR back to you.
                if not await ci_loop(ctx):
                    return finish(ctx, "CI red after rework", status="failed")
                await open_human_gate(ctx, [f"rework {ctx.reworks} after your /changes"])
            else:
                await after_build(ctx)
        except Exception as e:
            finish(ctx, f"failed: {str(e)[:160]}", status="failed")
            raise
        finally:
            stamp(root, ctx)
            ctx.save()


async def watch(poll_s: int, once: bool) -> None:
    log.info("watching runs parked at the review gate (Ctrl-C to stop)")
    while True:
        for n in Ctx.parked():
            ctx = Ctx.load(n)
            try:
                if await watch_one(ctx):
                    log.info("issue #%s resumed -> %s", n, ctx.status)
            except Exception:
                log.exception("resume of issue #%s failed", n)
        if once:
            return
        await asyncio.sleep(poll_s)


# ---------- CLI ----------


async def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("issues", type=int, nargs="*")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("-p", "--parallel", type=int, default=1, help="issues in flight at once")
    p.add_argument("--from", dest="start", choices=["spec", "ci"], default="spec")
    p.add_argument("--watch", action="store_true", help="resume runs parked at the review gate")
    p.add_argument("--once", action="store_true", help="with --watch: check once and exit")
    p.add_argument("--release", type=int, metavar="ISSUE", help="re-run only the release step")
    p.add_argument("--retry", type=int, metavar="ISSUE", help="resume a failed run from where it stopped")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config.require_target()
    if args.release or args.retry:
        tracing.init()
        try:
            if args.release:
                await rerelease(args.release)
            else:
                await retry(args.retry)
        finally:
            tracing.shutdown()
        return 0
    if not args.issues and not args.watch:
        p.error("give issue numbers, or --watch")
    tracing.init()
    failures = 0
    sem = asyncio.Semaphore(max(1, args.parallel))

    async def one(n: int, i: int) -> None:
        nonlocal failures
        await asyncio.sleep(i * 3 if args.parallel > 1 else 0)
        async with sem:
            try:
                ctx = await run_issue(n, args.attempt, args.start)
                log.info(
                    "issue #%s -> %s (%s, %s tokens)",
                    n,
                    ctx.status,
                    (ctx.pr or {}).get("url"),
                    f"{ctx.tokens:,}",
                )
            except Exception:
                failures += 1
                log.exception("issue #%s failed", n)

    try:
        if args.issues:
            await asyncio.gather(*(one(n, i) for i, n in enumerate(args.issues)))
        if args.watch:
            await watch(config.POLL_S, args.once)
    finally:
        tracing.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        sys.exit(130)
