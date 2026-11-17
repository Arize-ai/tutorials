"""Arize AX tracing for the factory.

Two kinds of trace land in the same Arize AX project:

1. The factory's run trace, one per GitHub issue, which this module sends:

    factory.run            CHAIN   root; input = issue, output = status + PR url
      station.spec         CHAIN   one per station; metadata.station = "spec"
      station.build        CHAIN
      station.ci           CHAIN   waits on GitHub Actions (no agent)
      station.babysit      CHAIN   only when CI fails
      station.review       CHAIN
        station.review_correctness / _security / _tests   CHAIN (parallel)
      station.router       CHAIN   agent + rules floor
      station.automerge | station.human
      station.release      CHAIN
    factory.resume         CHAIN   child of factory.run, when --watch picks up your verdict

   Every span carries session.id = "<repo>#<issue>", and station spans carry the
   metadata later lessons join on: station, issue, pr, attempt.

2. One trace per agent turn, sent by the Codex coding harness tracing (the notify hook
   installed with `install.sh codex`): an LLM span for the turn with a TOOL span per
   tool call. The factory can't nest those under its station spans, so it stamps them
   instead. agent_env() passes OTEL_RESOURCE_ATTRIBUTES to the Codex runtime, and the
   hook copies them onto every span it sends: factory.issue, factory.station,
   factory.attempt, factory.run_trace_id, and the Prompt Hub version that ran. Each
   station span records factory.codex_thread_id, which is the agent trace's session.id.

With no Arize credentials, init() still installs a tracer (so trace ids exist) but
nothing is exported.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from . import config

log = logging.getLogger(__name__)
_provider: Any = None
_tracer: Any = None


def init() -> None:
    global _provider, _tracer
    if _tracer is not None:
        return
    from openinference.instrumentation import OITracer, TraceConfig
    from opentelemetry.sdk.trace import TracerProvider

    if config.ARIZE_SPACE_ID and config.ARIZE_API_KEY:
        from arize.otel import HTTPSpanExporter
        from openinference.semconv.resource import ResourceAttributes
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        _provider = TracerProvider(
            resource=Resource.create({ResourceAttributes.PROJECT_NAME: config.ARIZE_PROJECT_NAME})
        )
        _provider.add_span_processor(
            BatchSpanProcessor(
                HTTPSpanExporter(
                    space_id=config.ARIZE_SPACE_ID,
                    api_key=config.ARIZE_API_KEY,
                    endpoint="https://otlp.arize.com/v1/traces",
                )
            )
        )
        log.info("Tracing to Arize project %s", config.ARIZE_PROJECT_NAME)
    else:
        _provider = TracerProvider()
        log.warning("ARIZE_SPACE_ID/ARIZE_API_KEY not set: running untraced")
    trace.set_tracer_provider(_provider)
    _tracer = OITracer(_provider.get_tracer("software-factory"), config=TraceConfig())


def shutdown() -> None:
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()


def _flat(meta: dict) -> str:
    return json.dumps({k: v for k, v in meta.items() if v is not None})


@contextmanager
def span(
    name: str, *, session_id: str, metadata: dict, input: Any = None, parent: tuple[str, str] | None = None
) -> Iterator[Any]:
    """A CHAIN span with session + metadata set the OpenInference way.

    parent=(trace_id, span_id) attaches the span to an earlier trace, so a run resumed
    after the human gate (in another process, maybe hours later) stays one trace.
    """
    attrs = {"session.id": session_id, "metadata": _flat(metadata)}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            attrs[f"factory.{k}"] = v
    if input is not None:
        attrs["input.value"] = input if isinstance(input, str) else json.dumps(input)
        attrs["input.mime_type"] = "text/plain" if isinstance(input, str) else "application/json"
    ctx = None
    if parent:
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        sc = SpanContext(
            trace_id=int(parent[0], 16),
            span_id=int(parent[1], 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(sc))
    with _tracer.start_as_current_span(
        name, context=ctx, attributes=attrs, openinference_span_kind="chain"
    ) as s:
        yield s


def set_output(s: Any, value: Any) -> None:
    if isinstance(value, str):
        s.set_attribute("output.value", value)
        s.set_attribute("output.mime_type", "text/plain")
    else:
        s.set_attribute("output.value", json.dumps(value))
        s.set_attribute("output.mime_type", "application/json")


def trace_id_of(s: Any) -> str:
    return format(s.get_span_context().trace_id, "032x")


def span_id_of(s: Any) -> str:
    return format(s.get_span_context().span_id, "016x")


# ---------- linking agent traces to the run ----------

_UNSAFE = re.compile(r"[,\r\n]+")


def agent_env(attrs: dict) -> dict[str, str]:
    """Environment for a station's Codex runtime, read by the Codex notify hook.

    ARIZE_PROJECT_NAME sends the agent's trace to the factory's project (overriding the
    project the hook was installed with). OTEL_RESOURCE_ATTRIBUTES is a "k=v,k=v" list
    the hook adds to every span. It splits pairs on commas and each pair on its first
    "=", so only commas (and newlines) in a value need replacing; base64 ids keep their "=".
    """
    pairs = []
    for k, v in attrs.items():
        if v is None or v == "":
            continue
        pairs.append(f"factory.{k}={_UNSAFE.sub('_', str(v))[:200]}")
    return {"ARIZE_PROJECT_NAME": config.ARIZE_PROJECT_NAME, "OTEL_RESOURCE_ATTRIBUTES": ",".join(pairs)}


def record_agent(s: Any, result: Any) -> None:
    """Stamp a station span with the agent run it made, so you can jump to its trace."""
    s.set_attribute("factory.codex_thread_id", result.thread_id)
    s.set_attribute("factory.agent_status", result.status)
    s.set_attribute("factory.tokens", result.tokens)
    if result.stopped:
        s.set_attribute("factory.agent_stopped", result.stopped)


def trace_url(trace_id: str, start_ms: int, end_ms: int) -> str | None:
    """Arize UI deep link for a trace, if ARIZE_ORG_ID and ARIZE_PROJECT_ID are set."""
    if not (config.ARIZE_ORG_ID and config.ARIZE_PROJECT_ID and config.ARIZE_SPACE_ID):
        return None
    return (
        f"https://app.arize.com/organizations/{config.ARIZE_ORG_ID}/spaces/{config.ARIZE_SPACE_ID}"
        f"/projects/{config.ARIZE_PROJECT_ID}?selectedTraceId={trace_id}&queryFilterA="
        f"&selectedTab=llmTracing&timeZoneA=UTC&startA={start_ms}"
        f"&endA={end_ms}&envA=tracing&modelType=generative_llm"
    )
