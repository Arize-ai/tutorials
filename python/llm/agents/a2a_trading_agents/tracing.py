"""Arize AX tracing for both agents.

The two agents need different instrumentation because they are built on different
frameworks:

  Bear (Pydantic AI) emits OpenTelemetry GenAI spans. OpenInferenceSpanProcessor
  translates those into OpenInference attributes so Arize AX reads them as LLM spans.

  Bull (Google ADK) is instrumented directly by GoogleADKInstrumentor.

Both write into one tracer provider, so a single trace spans the orchestrator, the A2A
hop, and each specialist's tool calls.

The processor is passed to register() through span_processors= rather than added
afterwards with add_span_processor(). That is deliberate: add_span_processor() on a
provider returned by register() shuts down and discards the default Arize exporter, and
since OpenInferenceSpanProcessor only translates spans and does not export them, adding
it that way would leave the process with no exporter and send nothing to Arize AX.
"""

import os

from arize.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from pydantic_ai import Agent, InstrumentationSettings

_tracer_provider = None


def setup_tracing(project_name: str | None = None):
    """Configure tracing once per process and return the tracer provider."""
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    project_name = project_name or os.environ.get(
        "ARIZE_PROJECT_NAME", "a2a-trading-agents"
    )

    _tracer_provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=project_name,
        span_processors=[OpenInferenceSpanProcessor()],
        set_global_tracer_provider=True,
    )

    GoogleADKInstrumentor().instrument(tracer_provider=_tracer_provider)

    # Pydantic AI 2.x sets instrumentation on the Agent class rather than per-agent
    # (Agent(instrument=True) was removed), so point every agent at this provider.
    Agent.instrument_all(InstrumentationSettings(tracer_provider=_tracer_provider))

    return _tracer_provider


def flush() -> None:
    """Force-flush pending spans. Arize AX ingests asynchronously, so call this before exit."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush()
