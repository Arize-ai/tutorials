"""Settings for the course harness, read from the environment and an optional .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

COURSE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(COURSE_DIR / ".env")
# The Arize SDK uses gRPC, which logs a harmless warning each time the harness starts a
# Claude Code process. This turns that warning off.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set. Add it to {COURSE_DIR / '.env'} (see .env.example).")
    return value


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    arize_api_key: str
    arize_space_id: str
    agent_model: str
    dataset_name: str
    judge_name: str

    def require_anthropic_key(self) -> str:
        """The Anthropic API key, which the agent runs need."""
        if not self.anthropic_api_key:
            _required("ANTHROPIC_API_KEY")
        return self.anthropic_api_key


def settings() -> Settings:
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        arize_api_key=_required("ARIZE_API_KEY"),
        arize_space_id=_required("ARIZE_SPACE_ID"),
        agent_model=os.environ.get("AGENT_MODEL", "claude-haiku-5-5"),
        dataset_name=os.environ.get("DATASET_NAME", "brindlewick-utm-requests"),
        judge_name=os.environ.get("JUDGE_NAME", "brindlewick-utm-judge"),
    )


def arize_client():
    from arize import ArizeClient

    return ArizeClient(api_key=settings().arize_api_key)
