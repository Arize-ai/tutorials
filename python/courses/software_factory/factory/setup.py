"""Create your own helpdesk repo on GitHub from the template in app/.

    python -m factory.setup you/helpdesk            # a private repo
    python -m factory.setup you/helpdesk --public

The factory needs a repo of its own to work on: one where it can open PRs, run GitHub
Actions and create releases. This copies app/ into a new git repo, pushes it to GitHub
with the GitHub CLI, and prints the FACTORY_TARGET_REPO line for your .env.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config

TEMPLATE = config.ROOT / "app"
IGNORE = shutil.ignore_patterns(
    ".venv", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", "*.db", ".env"
)


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)


def create(repo: str, *, public: bool) -> None:
    if "/" not in repo:
        sys.exit("Name the repo as owner/name, for example you/helpdesk.")
    if run("gh", "repo", "view", repo).returncode == 0:
        sys.exit(f"{repo} already exists. Pick another name, or set FACTORY_TARGET_REPO={repo}.")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "helpdesk"
        shutil.copytree(TEMPLATE, src, ignore=IGNORE)
        for step in (
            ("git", "init", "-q", "-b", "main"),
            ("git", "add", "-A"),
            ("git", "commit", "-q", "-m", "Helpdesk, from the Arize AX software factory course"),
        ):
            out = run(*step, cwd=src)
            if out.returncode != 0:
                sys.exit(f"{' '.join(step)} failed: {out.stderr.strip()}")
        out = run(
            "gh",
            "repo",
            "create",
            repo,
            "--public" if public else "--private",
            "--description",
            "A multi-tenant helpdesk, built and released by a software factory",
            "--source",
            str(src),
            "--push",
        )
        if out.returncode != 0:
            sys.exit(f"gh repo create failed: {out.stderr.strip()}")
    print(f"Created https://github.com/{repo}\n\nAdd this to your .env:\n\nFACTORY_TARGET_REPO={repo}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m factory.setup", description=__doc__.split("\n")[0])
    parser.add_argument("repo", help="owner/name for the new repo, for example you/helpdesk")
    parser.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    args = parser.parse_args(argv)
    create(args.repo, public=args.public)


if __name__ == "__main__":
    main()
