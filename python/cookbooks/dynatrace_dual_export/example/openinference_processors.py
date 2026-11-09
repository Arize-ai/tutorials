"""Span processors and exporters for dual export from one TracerProvider.

`OpenInferenceSpanProcessor` wraps the Arize processor: it forwards only AI spans
(OpenInference or OTel GenAI semantic conventions) and, optionally, detaches AI spans
whose parent is a non-AI span so that they show up as trace roots in Arize.

`ContentRedactingExporter` wraps the Dynatrace exporter: it forwards every span but
drops prompt and completion content, so Dynatrace keeps the full trace tree, timings,
models, and token counts without a second copy of the AI payload.

Both work on copies. The SDK hands the same span object to every processor, so
neither processor mutates it.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional, Sequence

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

SPAN_KIND = "openinference.span.kind"


def is_ai_span(span: ReadableSpan) -> bool:
    """True for OpenInference spans and for spans that follow the OTel GenAI conventions.

    Arize normalizes `gen_ai.*` spans into OpenInference at ingestion, so both are kept.
    """
    attributes = span.attributes or {}
    if SPAN_KIND in attributes:
        return True
    return any(key.startswith("gen_ai.") for key in attributes)


def _copy(span: ReadableSpan, *, parent=..., attributes=None) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent if parent is ... else parent,
        resource=span.resource,
        attributes=span.attributes if attributes is None else attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class OpenInferenceSpanProcessor(SpanProcessor):
    """Forwards AI spans to `delegate` and drops everything else.

    With `reparent_orphans=True`, an AI span whose local parent is a non-AI span (for
    example the HTTP server span) is exported without a parent. Arize lists traces by
    their root span, so without this the AI spans of that trace appear only in the
    Spans tab. Remote parents (W3C traceparent from another service) are kept as is,
    because the upstream service may export its own AI spans to Arize.

    The decision uses the parent's attributes at the child's start time, like
    `reparentOrphanedSpans` in @arizeai/openinference-vercel.
    """

    def __init__(
        self,
        delegate: SpanProcessor,
        *,
        span_filter: Callable[[ReadableSpan], bool] = is_ai_span,
        reparent_orphans: bool = True,
    ) -> None:
        self._delegate = delegate
        self._filter = span_filter
        self._reparent = reparent_orphans
        self._orphans: set[int] = set()

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        if self._reparent:
            parent = trace.get_current_span(parent_context)
            if isinstance(parent, ReadableSpan) and parent.context.is_valid and not self._filter(parent):
                self._orphans.add(span.context.span_id)
        self._delegate.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        orphan = span.context.span_id in self._orphans
        self._orphans.discard(span.context.span_id)
        if not self._filter(span):
            return
        self._delegate.on_end(_copy(span, parent=None) if orphan else span)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


# Prompt and completion content. Token counts, model names, and tool names are kept.
DEFAULT_CONTENT_PATTERNS = (
    r"^input\.value$",
    r"^output\.value$",
    r"^llm\.input_messages\.",
    r"^llm\.output_messages\.",
    r"^llm\.prompts?\.",
    r"^llm\.prompt_template\.variables$",
    r"^gen_ai\.input\.messages$",
    r"^gen_ai\.output\.messages$",
    r"^gen_ai\.system_instructions$",
    r"^gen_ai\.tool\.call\.(arguments|result)$",
)


class ContentRedactingExporter(SpanExporter):
    """Exports every span, minus attributes that hold prompt or completion content."""

    def __init__(self, delegate: SpanExporter, patterns: Iterable[str] = DEFAULT_CONTENT_PATTERNS) -> None:
        self._delegate = delegate
        self._patterns = re.compile("|".join(patterns))

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._delegate.export([self._redact(span) for span in spans])

    def _redact(self, span: ReadableSpan) -> ReadableSpan:
        attributes = span.attributes or {}
        if not any(self._patterns.match(key) for key in attributes):
            return span
        kept = {key: value for key, value in attributes.items() if not self._patterns.match(key)}
        return _copy(span, attributes=kept)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
