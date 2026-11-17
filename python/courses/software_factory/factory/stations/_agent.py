"""The one place a station runs its agent.

Resolves the station's production prompt from Prompt Hub, runs a Codex turn with the
attributes that tie its trace to this run, and records the result on the station span.
"""

from __future__ import annotations

from pathlib import Path

from opentelemetry import trace

from .. import config, prompts, tracing
from ..agent import AgentResult, run_agent


async def run(
    ctx,
    station: str,
    key: str,
    *,
    prompt: str,
    cwd: Path,
    model: str,
    sandbox: str,
    schema: dict,
    max_tokens: int,
    network: bool = False,
) -> AgentResult:
    """station is the span/event name (e.g. "review_tests"); key is the prompt catalog key."""
    p = prompts.use(key)
    res = await run_agent(
        prompt=prompt,
        instructions=p.text,
        cwd=cwd,
        model=model,
        sandbox=sandbox,
        trace_attrs=ctx.agent_attrs(station, model, p.meta),
        output_schema=schema,
        max_tokens=max_tokens,
        timeout_s=config.AGENT_TIMEOUT_S,
        network=network,
        on_tool=ctx.ticker(station),
    )
    tracing.record_agent(trace.get_current_span(), res)
    ctx.tokens += res.tokens
    return res


def why(res: AgentResult) -> str:
    """Why an agent produced no usable result, for error messages."""
    reason = res.stopped or res.status or "no result"
    return f"{reason}: {res.text[:300]}" if res.text else reason
