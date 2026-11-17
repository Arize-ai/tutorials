"""One helper to run a Codex agent for a station.

Each call starts a Codex runtime (the Codex SDK drives `codex app-server`), opens one
thread and runs one turn:

- the station's system prompt goes in as developer instructions, on top of Codex's
  own base instructions;
- the sandbox is the permission model: read-only stations can't write anywhere, and
  workspace-write stations can only write inside their worktree. Escalation requests
  are denied, so nothing ever waits for an approval;
- the output schema is converted to OpenAI's strict form, so the final message is JSON
  the station can parse;
- a token budget and a wall-clock limit interrupt a turn that runs away.

Tracing happens outside this process. The Codex notify hook traces the turn when it
completes, and the env from tracing.agent_env() stamps its spans with the run they
belong to.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionThreadItem,
    FileChangeThreadItem,
    ItemCompletedNotification,
    ItemStartedNotification,
    McpToolCallThreadItem,
    MessagePhase,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    WebSearchThreadItem,
)

from . import tracing

log = logging.getLogger(__name__)

SANDBOXES = {"read_only": Sandbox.read_only, "workspace_write": Sandbox.workspace_write}


@dataclass
class AgentResult:
    text: str
    structured: Any = None
    thread_id: str = ""
    status: str = ""
    stopped: str = ""  # why the factory interrupted the turn, if it did
    tokens: int = 0  # uncached input + output tokens
    usage: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status != "completed" or bool(self.stopped)


async def run_agent(
    *,
    prompt: str,
    instructions: str,
    cwd: Path,
    model: str,
    sandbox: str,
    trace_attrs: dict,
    output_schema: dict | None = None,
    max_tokens: int = 500_000,
    timeout_s: int = 1800,
    network: bool = False,
    on_tool: Callable[[str], None] | None = None,
) -> AgentResult:
    """Run one agent turn to completion. on_tool gets a one-line summary of each tool
    call, e.g. "$ uv run pytest" or "edit shipit/stats.py"."""
    overrides = ("sandbox_workspace_write.network_access=true",) if network else ()
    codex_config = CodexConfig(cwd=str(cwd), env=tracing.agent_env(trace_attrs), config_overrides=overrides)
    result = AgentResult(text="")
    async with AsyncCodex(codex_config) as codex:
        thread = await codex.thread_start(
            cwd=str(cwd),
            model=model,
            sandbox=SANDBOXES[sandbox],
            approval_mode=ApprovalMode.deny_all,
            developer_instructions=instructions,
        )
        result.thread_id = thread.id
        turn = await thread.turn(
            prompt, output_schema=strict_schema(output_schema) if output_schema else None
        )
        final: str | None = None
        last: str | None = None

        async def consume() -> None:
            nonlocal final, last
            async for event in turn.stream():
                p = event.payload
                if isinstance(p, ItemStartedNotification) and on_tool:
                    label = describe_item(p.item, cwd)
                    if label:
                        on_tool(label)
                elif isinstance(p, ItemCompletedNotification):
                    item = getattr(p.item, "root", p.item)
                    if isinstance(item, AgentMessageThreadItem):
                        last = item.text
                        if item.phase == MessagePhase.final_answer:
                            final = item.text
                elif isinstance(p, ThreadTokenUsageUpdatedNotification):
                    total = p.token_usage.total
                    result.usage = total.model_dump(mode="json")
                    result.tokens = total.input_tokens - total.cached_input_tokens + total.output_tokens
                    if result.tokens > max_tokens and not result.stopped:
                        result.stopped = f"token budget ({max_tokens}) exceeded"
                        log.warning("%s: interrupting turn: %s", thread.id, result.stopped)
                        await turn.interrupt()
                elif isinstance(p, TurnCompletedNotification):
                    result.status = p.turn.status.value
                    if p.turn.error and p.turn.error.message:
                        result.text = p.turn.error.message

        try:
            await asyncio.wait_for(consume(), timeout=timeout_s)
        except TimeoutError:
            result.stopped = f"timed out after {timeout_s}s"
            log.warning("%s: %s", thread.id, result.stopped)
            try:
                await turn.interrupt()
            except Exception as e:  # noqa: BLE001 - the runtime is torn down on exit anyway
                log.warning("%s: interrupt failed: %s", thread.id, e)

    answer = final if final is not None else last
    if answer:
        result.text = answer
    if output_schema and answer:
        try:
            result.structured = drop_nulls(json.loads(answer))
        except json.JSONDecodeError:
            result.structured = None
    return result


# ---------- schemas ----------


def strict_schema(schema: dict) -> dict:
    """Convert a JSON schema to OpenAI's strict structured-output form.

    Strict mode needs every object to list all its properties as required and to set
    additionalProperties to false. A property that was optional becomes nullable
    instead, and drop_nulls() removes the nulls again so stations can keep using
    .get() for optional fields.
    """
    s = copy.deepcopy(schema)

    def walk(node: dict) -> None:
        if node.get("type") == "object" and "properties" in node:
            required = set(node.get("required", []))
            for name, prop in node["properties"].items():
                walk(prop)
                if name not in required:
                    _make_nullable(prop)
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        elif node.get("type") == "array" and isinstance(node.get("items"), dict):
            walk(node["items"])

    walk(s)
    return s


def _make_nullable(prop: dict) -> None:
    t = prop.get("type")
    if isinstance(t, str):
        prop["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        prop["type"] = [*t, "null"]
    if "enum" in prop and None not in prop["enum"]:
        prop["enum"] = [*prop["enum"], None]


def drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nulls(v) for v in value]
    return value


# ---------- tool ticker ----------


def describe_item(item: Any, cwd: Path) -> str:
    """Short, human-readable label for a tool call, or "" for messages and reasoning."""
    item = getattr(item, "root", item)

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(Path(cwd).resolve()))
        except (ValueError, OSError):
            return Path(p).name

    if isinstance(item, CommandExecutionThreadItem):
        label = "$ " + " ".join(item.command.split())
    elif isinstance(item, FileChangeThreadItem):
        label = "edit " + ", ".join(rel(c.path) for c in item.changes)
    elif isinstance(item, WebSearchThreadItem):
        label = f"search {item.query}"
    elif isinstance(item, McpToolCallThreadItem):
        label = f"{item.server}.{item.tool}"
    else:
        return ""
    return label if len(label) <= 80 else label[:77] + "..."
