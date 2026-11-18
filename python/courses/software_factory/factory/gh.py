"""Thin wrappers over git and the gh CLI. Deterministic plumbing, no agents here.

The agents never push, open PRs or comment: the orchestrator does that, so every
write to GitHub is a known, logged step.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from . import config


def sh(*args: str, cwd: Path | None = None, check: bool = True, input: str | None = None) -> str:
    out = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True, input=input)
    if check and out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({out.returncode}): {out.stderr.strip()}")
    return out.stdout.strip()


def gh(*args: str, input: str | None = None) -> str:
    return sh("gh", *args, "-R", config.TARGET_REPO, input=input)


def issue(number: int) -> dict:
    return json.loads(gh("issue", "view", str(number), "--json", "number,title,body,labels,url"))


def comment_issue(number: int, body: str) -> None:
    gh("issue", "comment", str(number), "--body-file", "-", input=body)


def ensure_clone() -> Path:
    """A factory-owned clone of the target repo (never your own working copy)."""
    config.WORKDIR.mkdir(parents=True, exist_ok=True)
    if not (config.CLONE_DIR / ".git").exists():
        sh("gh", "repo", "clone", config.TARGET_REPO, str(config.CLONE_DIR))
    sh("git", "fetch", "--prune", "origin", cwd=config.CLONE_DIR)
    sh("git", "checkout", "--detach", "origin/main", cwd=config.CLONE_DIR)
    return config.CLONE_DIR


def worktree(branch: str) -> Path:
    """Fresh worktree on a new branch from origin/main."""
    clone = ensure_clone()
    path = config.WORKTREES / branch.replace("/", "-")
    if path.exists():
        sh("git", "worktree", "remove", "--force", str(path), cwd=clone)
    sh("git", "branch", "-D", branch, cwd=clone, check=False)
    sh("git", "worktree", "add", "-b", branch, str(path), "origin/main", cwd=clone)
    return path


def diff_stat(path: Path) -> str:
    sh("git", "add", "-A", cwd=path)
    return sh("git", "diff", "--cached", "--stat", "origin/main", cwd=path)


def commit_and_push(path: Path, branch: str, message: str) -> str:
    sh("git", "add", "-A", cwd=path)
    sh("git", "commit", "-q", "-F", "-", cwd=path, input=message)
    sh("git", "push", "-q", "--force", "-u", "origin", branch, cwd=path)
    return sh("git", "rev-parse", "HEAD", cwd=path)


def open_pr(branch: str, title: str, body: str, labels: list[str]) -> dict:
    existing = json.loads(gh("pr", "list", "--head", branch, "--json", "number,url"))
    if existing:
        gh("pr", "edit", str(existing[0]["number"]), "--title", title, "--body-file", "-", input=body)
        return existing[0]
    ensure_labels(labels)
    args = ["pr", "create", "--head", branch, "--base", "main", "--title", title, "--body-file", "-"]
    for label in labels:
        args += ["--label", label]
    url = gh(*args, input=body)
    number = int(url.rstrip("/").split("/")[-1])
    return {"number": number, "url": url}


# ---------- worktrees for existing branches (babysit, rework, resume) ----------


def worktree_for(branch: str) -> Path:
    """Worktree on an existing remote branch, reused if already present."""
    clone = ensure_clone()
    path = config.WORKTREES / branch.replace("/", "-")
    if (path / ".git").exists():
        # A run that died partway (out of credits, a crash) can leave a merge half done.
        # The worktree is reset to the remote branch below, so drop any local state first.
        sh("git", "merge", "--abort", cwd=path, check=False)
        sh("git", "reset", "-q", "--hard", cwd=path)
        sh("git", "fetch", "-q", "origin", branch, cwd=path)
        sh("git", "checkout", "-q", branch, cwd=path)
        sh("git", "reset", "-q", "--hard", f"origin/{branch}", cwd=path)
        return path
    sh("git", "branch", "-D", branch, cwd=clone, check=False)
    sh("git", "fetch", "-q", "origin", f"{branch}:refs/remotes/origin/{branch}", cwd=clone)
    sh("git", "worktree", "add", "-B", branch, str(path), f"origin/{branch}", cwd=clone)
    return path


def detached_worktree(name: str, ref: str = "origin/main") -> Path:
    """Throwaway detached worktree (release smoke tests)."""
    clone = ensure_clone()
    path = config.WORKTREES / name
    if path.exists():
        sh("git", "worktree", "remove", "--force", str(path), cwd=clone)
    sh("git", "worktree", "add", "--detach", str(path), ref, cwd=clone)
    return path


def has_changes(path: Path) -> bool:
    return bool(sh("git", "status", "--porcelain", cwd=path))


def pr_diff(path: Path, limit: int = 60_000) -> str:
    sh("git", "fetch", "-q", "origin", "main", cwd=path)
    d = sh("git", "diff", "origin/main...HEAD", cwd=path)
    return d if len(d) <= limit else d[:limit] + "\n\n[diff truncated]"


def changed_files(path: Path) -> list[str]:
    out = sh("git", "diff", "--name-only", "origin/main...HEAD", cwd=path)
    return [f for f in out.splitlines() if f]


def diff_lines(path: Path) -> int:
    out = sh("git", "diff", "--numstat", "origin/main...HEAD", cwd=path)
    total = 0
    for line in out.splitlines():
        a, d, *_ = line.split("\t")
        total += (int(a) if a.isdigit() else 0) + (int(d) if d.isdigit() else 0)
    return total


def test_count(path: Path, ref: str) -> int:
    """Number of `def test_` functions under tests/ at a git ref (tripwire for deleted tests)."""
    files = sh("git", "ls-tree", "-r", "--name-only", ref, "tests", cwd=path, check=False)
    n = 0
    for f in files.splitlines():
        if f.endswith(".py"):
            src = sh("git", "show", f"{ref}:{f}", cwd=path, check=False)
            n += sum(
                1 for line in src.splitlines() if line.lstrip().startswith(("def test_", "async def test_"))
            )
    return n


def push(path: Path, branch: str, message: str) -> str:
    return commit_and_push(path, branch, message)


def push_head(path: Path, branch: str) -> str:
    """Push commits already made in the worktree (e.g. a clean merge of main)."""
    sh("git", "push", "-q", "-u", "origin", branch, cwd=path)
    return sh("git", "rev-parse", "HEAD", cwd=path)


def mergeable(number: int) -> str:
    """MERGEABLE | CONFLICTING | UNKNOWN. GitHub computes this lazily, so poll briefly.

    GitHub runs no pull_request workflows on a conflicting PR, so a CI wait on one
    just times out. The factory checks this first and merges main when it conflicts."""
    m = "UNKNOWN"
    for _ in range(8):
        m = json.loads(gh("pr", "view", str(number), "--json", "mergeable"))["mergeable"]
        if m != "UNKNOWN":
            return m
        time.sleep(4)
    return m


def merge_main(path: Path) -> list[str]:
    """Merge origin/main into the worktree's branch. Returns [] if it merged cleanly
    (the merge commit is made), else the conflicted files, with the merge left in
    progress for the babysitter to resolve."""
    sh("git", "fetch", "-q", "origin", "main", cwd=path)
    out = subprocess.run(
        ["git", "merge", "--no-edit", "origin/main"], cwd=path, check=False, capture_output=True, text=True
    )
    if out.returncode == 0:
        return []
    files = sh("git", "diff", "--name-only", "--diff-filter=U", cwd=path).splitlines()
    if not files:
        sh("git", "merge", "--abort", cwd=path, check=False)
        raise RuntimeError(f"git merge origin/main failed: {out.stderr.strip()[:300]}")
    return files


# ---------- CI ----------


def check_runs(sha: str) -> list[dict]:
    data = json.loads(sh("gh", "api", f"repos/{config.TARGET_REPO}/commits/{sha}/check-runs"))
    return data.get("check_runs", [])


def failed_ci_log(sha: str, limit: int = 15_000) -> str:
    runs = json.loads(gh("run", "list", "--commit", sha, "--json", "databaseId,conclusion,name"))
    logs = []
    for r in runs:
        if r.get("conclusion") == "failure":
            logs.append(
                sh(
                    "gh",
                    "run",
                    "view",
                    str(r["databaseId"]),
                    "--log-failed",
                    "-R",
                    config.TARGET_REPO,
                    check=False,
                )
            )
    text = "\n".join(logs) or "(no failed-step log available)"
    return text if len(text) <= limit else "[log truncated]\n" + text[-limit:]


# ---------- PRs ----------


def pr(number: int) -> dict:
    return json.loads(
        gh(
            "pr",
            "view",
            str(number),
            "--json",
            "number,url,state,headRefName,headRefOid,title,body,labels,mergeCommit",
        )
    )


def comment_pr(number: int, body: str) -> None:
    gh("pr", "comment", str(number), "--body-file", "-", input=body)


def review_comment(number: int, body: str) -> None:
    """A COMMENT review. The factory acts as the repo owner, who can't approve or
    request changes on their own PR, so verdicts live in the review body."""
    try:
        gh("pr", "review", str(number), "--comment", "--body-file", "-", input=body)
    except RuntimeError:
        comment_pr(number, body)


def ensure_labels(names) -> None:
    """Create any labels the repo doesn't have yet. A new repo has none of the factory's."""
    for name in names:
        sh("gh", "label", "create", name, "-R", config.TARGET_REPO, "--force", check=False)


def label_issue(number: int, add: list[str] = ()) -> None:
    ensure_labels(add)
    args = ["issue", "edit", str(number)]
    for name in add:
        args += ["--add-label", name]
    if len(args) > 3:
        gh(*args)


def label_pr(number: int, add: list[str] = (), remove: list[str] = ()) -> None:
    ensure_labels(add)
    args = ["pr", "edit", str(number)]
    for name in add:
        args += ["--add-label", name]
    for name in remove:
        args += ["--remove-label", name]
    if len(args) > 3:
        gh(*args)


def merge_pr(number: int) -> str:
    gh("pr", "merge", str(number), "--squash", "--delete-branch")
    return (pr(number).get("mergeCommit") or {}).get("oid", "")


def pr_conversation(number: int) -> list[dict]:
    """Issue comments and review bodies on a PR, oldest first: {id, at, author, body, kind}."""
    repo = config.TARGET_REPO
    items = []
    for c in json.loads(sh("gh", "api", "--paginate", f"repos/{repo}/issues/{number}/comments")):
        items.append(
            {
                "id": f"c{c['id']}",
                "at": c["created_at"],
                "author": c["user"]["login"],
                "body": c.get("body") or "",
                "kind": "comment",
            }
        )
    for r in json.loads(sh("gh", "api", "--paginate", f"repos/{repo}/pulls/{number}/reviews")):
        items.append(
            {
                "id": f"r{r['id']}",
                "at": r.get("submitted_at") or "",
                "author": r["user"]["login"],
                "body": r.get("body") or "",
                "kind": "review",
            }
        )
    for c in json.loads(sh("gh", "api", "--paginate", f"repos/{repo}/pulls/{number}/comments")):
        items.append(
            {
                "id": f"l{c['id']}",
                "at": c["created_at"],
                "author": c["user"]["login"],
                "body": f"{c.get('path')}:{c.get('line') or c.get('original_line')}: {c.get('body') or ''}",
                "kind": "line",
            }
        )
    return sorted(items, key=lambda i: i["at"])


def latest_spec_comment(issue_number: int) -> str | None:
    comments = json.loads(
        sh("gh", "api", "--paginate", f"repos/{config.TARGET_REPO}/issues/{issue_number}/comments")
    )
    specs = [c["body"] for c in comments if (c.get("body") or "").startswith("### 🏭 Factory spec")]
    return specs[-1] if specs else None


# ---------- releases ----------


def latest_release_tag() -> str | None:
    """Highest semver tag. `gh release list` orders by date, which can disagree with
    version order (a release re-run later for an older merge), so sort explicitly."""
    tags = gh("release", "list", "--limit", "200", "--json", "tagName", "--jq", ".[].tagName").split()

    def key(t):
        m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", t)
        return tuple(map(int, m.groups())) if m else (-1,)

    tags = [t for t in tags if key(t) != (-1,)]
    return max(tags, key=key) if tags else None


def create_release(tag: str, target_sha: str, title: str, notes: str) -> str:
    return gh(
        "release", "create", tag, "--target", target_sha, "--title", title, "--notes-file", "-", input=notes
    )
