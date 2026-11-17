"""Factory configuration, read from .env (which wins over the shell)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", "")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "")
# The factory's own trace and every agent's Codex trace land in this project. Its own
# variable, not ARIZE_PROJECT_NAME: a traced coding-agent session exports that one, and
# running the factory from inside it would otherwise send the run to that session's project.
ARIZE_PROJECT_NAME = os.getenv("FACTORY_ARIZE_PROJECT") or "software-factory"
# Prompt Hub names are <prefix>-spec, <prefix>-build, ...
PROMPT_PREFIX = os.getenv("FACTORY_PROMPT_PREFIX") or "software-factory"
# Only used to build UI deep links in the console output.
ARIZE_ORG_ID = os.getenv("ARIZE_ORG_ID", "")
ARIZE_PROJECT_ID = os.getenv("ARIZE_PROJECT_ID", "")

TARGET_REPO = os.getenv("FACTORY_TARGET_REPO", "")
WORKDIR = (ROOT / os.getenv("FACTORY_WORKDIR", "work")).resolve()
CLONE_DIR = WORKDIR / (TARGET_REPO.split("/")[-1] or "target")
WORKTREES = WORKDIR / "worktrees"

# Codex models: the workhorse builds, the frontier model reviews.
BUILD_MODEL = os.getenv("FACTORY_BUILD_MODEL", "gpt-6.1-sol")
REVIEW_MODEL = os.getenv("FACTORY_REVIEW_MODEL", "gpt-6-astra")

# Hard stops per station so a confused agent can't run up a bill. Budgets count the
# tokens a turn actually pays full price for: uncached input plus output.
SPEC_MAX_TOKENS = int(os.getenv("FACTORY_SPEC_MAX_TOKENS", "300000"))
BUILD_MAX_TOKENS = int(os.getenv("FACTORY_BUILD_MAX_TOKENS", "2000000"))
BABYSIT_MAX_TOKENS = int(os.getenv("FACTORY_BABYSIT_MAX_TOKENS", "1000000"))
REVIEW_MAX_TOKENS = int(os.getenv("FACTORY_REVIEW_MAX_TOKENS", "600000"))  # per reviewer
ROUTER_MAX_TOKENS = int(os.getenv("FACTORY_ROUTER_MAX_TOKENS", "100000"))
RELEASE_MAX_TOKENS = int(os.getenv("FACTORY_RELEASE_MAX_TOKENS", "400000"))
# Wall-clock limit for any one agent turn.
AGENT_TIMEOUT_S = int(os.getenv("FACTORY_AGENT_TIMEOUT_S", "1800"))

# Loop limits and gates.
MAX_CI_FIXES = int(os.getenv("FACTORY_MAX_CI_FIXES", "3"))
MAX_REWORKS = int(os.getenv("FACTORY_MAX_REWORKS", "2"))
CI_TIMEOUT_S = int(os.getenv("FACTORY_CI_TIMEOUT_S", "900"))
# How long a run waits in-process for your /ship or /changes before handing off to
# `python -m factory.run --watch`. 0 = hand off immediately.
HUMAN_WAIT_S = int(os.getenv("FACTORY_HUMAN_WAIT_S", "0"))
POLL_S = int(os.getenv("FACTORY_POLL_S", "15"))
# Rules floor for the router: these always go to the human gate.
BIG_DIFF_LINES = int(os.getenv("FACTORY_BIG_DIFF_LINES", "400"))
# Auth, secrets, the data model and the dependency/CI config. In the helpdesk app, a
# mistake in any of these can leak one workspace's data to another.
DEFAULT_SENSITIVE_PATHS = (
    ".github/",
    "pyproject.toml",
    "uv.lock",
    "helpdesk/deps.py",
    "helpdesk/security.py",
    "helpdesk/models.py",
    "helpdesk/config.py",
    "helpdesk/app.py",
)
SENSITIVE_PATHS = [
    p for p in os.getenv("FACTORY_SENSITIVE_PATHS", ",".join(DEFAULT_SENSITIVE_PATHS)).split(",") if p
]
STATE_DIR = ROOT / "runs" / "state"

PROMPTS = ROOT / "factory" / "prompts"


def require_target() -> str:
    if not TARGET_REPO:
        raise SystemExit("Set FACTORY_TARGET_REPO in .env to your fork, e.g. you/helpdesk")
    return TARGET_REPO


def prompt(name: str) -> str:
    """Local prompt file. Stations use factory.prompts (Prompt Hub, falling back to these)."""
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
