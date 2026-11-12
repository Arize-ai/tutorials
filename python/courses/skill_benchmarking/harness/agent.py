"""Run one request through headless Claude Code with only the skill under test installed."""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import settings
from harness.sandbox import create_sandbox

# Claude Code can load the skill and read files. Everything that writes, runs commands,
# or reaches the network is blocked. (Claude Code's --tools flag would be simpler, but it
# stops coding-harness-tracing from recording tool and LLM spans.)
ALLOWED_TOOLS = "Skill,Read,Glob,Grep"
BLOCKED_TOOLS = "Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Agent"
TIMEOUT_SECONDS = 300


@dataclass
class AgentRun:
    response: str
    session_id: str
    tools: list[dict] = field(default_factory=list)
    cost_usd: float | None = None
    duration_ms: float | None = None
    tokens: int | None = None
    num_turns: int | None = None
    error: str | None = None


def _agent_env(config_dir: Path, project: str) -> dict[str, str]:
    """Environment for the Claude Code process.

    Starts from your environment minus anything Claude-, Arize-, or OpenTelemetry-related,
    then adds only what this run needs.
    """
    s = settings()
    prefixes = ("CLAUDE", "ANTHROPIC", "ARIZE", "OTEL", "PHOENIX")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env.update(
        ANTHROPIC_API_KEY=s.require_anthropic_key(),
        CLAUDE_CONFIG_DIR=str(config_dir),
        # Read by the coding-harness-tracing hooks, which send the session to Arize AX.
        ARIZE_API_KEY=s.arize_api_key,
        ARIZE_SPACE_ID=s.arize_space_id,
        ARIZE_PROJECT_NAME=project,
        ARIZE_TRACE_ENABLED="true",
        ARIZE_USER_ID="skill-benchmark",
        # The judge reads the files the agent opened from its tool spans.
        ARIZE_LOG_TOOL_CONTENT="true",
    )
    return env


def _total_tokens(usage: dict) -> int | None:
    """Every token the session's model calls used, including the cached prompt."""
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return sum(usage.get(k) or 0 for k in keys) if usage else None


def _parse_stream(stdout: str) -> tuple[str, list[dict], dict]:
    response, tools, result = "", [], {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tools.append({"name": block.get("name"), "input": block.get("input")})
        elif event.get("type") == "result":
            result = event
            response = event.get("result") or ""
    return response, tools, result


async def run_agent(request: str, skill_dir: Path, project: str) -> AgentRun:
    """Run headless Claude Code on one request and trace the session to `project` in Arize AX."""
    s = settings()
    sandbox = create_sandbox(skill_dir)
    session_id = str(uuid.uuid4())
    cmd = [
        "claude", "-p", request,
        "--model", s.agent_model,
        "--session-id", session_id,
        "--allowedTools", ALLOWED_TOOLS,
        "--disallowedTools", BLOCKED_TOOLS,
        "--strict-mcp-config",
        "--max-turns", "10",
        "--output-format", "stream-json",
        "--verbose",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=sandbox.workspace,
            env=_agent_env(sandbox.config_dir, project),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            return AgentRun("", session_id, error=f"Timed out after {TIMEOUT_SECONDS}s")
        response, tools, result = _parse_stream(stdout.decode())
        error = None
        if proc.returncode != 0 or result.get("is_error"):
            error = (stderr.decode()[-500:] or result.get("subtype") or "claude exited with an error").strip()
        return AgentRun(
            response=response,
            session_id=session_id,
            tools=tools,
            cost_usd=result.get("total_cost_usd"),
            duration_ms=result.get("duration_ms"),
            tokens=_total_tokens(result.get("usage") or {}),
            num_turns=result.get("num_turns"),
            error=error,
        )
    finally:
        sandbox.cleanup()
