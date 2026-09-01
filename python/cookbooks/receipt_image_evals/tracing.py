"""Arize AX and OpenTelemetry setup for the receipt demo."""

from __future__ import annotations

import os

from arize.otel import register
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor


def initialize_tracing(instrumentation_scope: str) -> OITracer:
    """Configure mandatory AX tracing before any OpenAI client is created."""
    required = ("ARIZE_API_KEY", "ARIZE_SPACE_ID")
    if not all(os.environ.get(name) for name in required):
        raise RuntimeError("Tracing requires ARIZE_API_KEY and ARIZE_SPACE_ID before starting the receipt app.")
    provider = register(
        project_name=os.environ.get("ARIZE_PROJECT_NAME", "receipt-image-evals"),
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        # Export each span before the process moves on; batch mode is disabled
        # so the CLI batch command does not need an explicit flush.
        batch=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    # OITracer adds @chain, which captures the receipt function's input/output.
    return OITracer(provider.get_tracer(instrumentation_scope), config=TraceConfig())
