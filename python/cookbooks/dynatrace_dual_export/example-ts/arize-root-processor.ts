import { trace, type Context } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace-base";

/**
 * Wraps the Arize span processor. AI spans whose parent is a local non-AI span (for
 * example the HTTP server span) reach the inner processor without a parent, so they are
 * trace roots in Arize. Other processors on the same provider (Dynatrace) keep the real
 * parent, because only a view of the span is changed, never the shared span.
 *
 * Use this instead of `reparentOrphanedSpans: true` when AI spans also go to another
 * backend: that option clears the parent on the live span for every processor.
 */
export class ArizeRootProcessor implements SpanProcessor {
  private readonly orphans = new Set<string>();

  constructor(private readonly inner: SpanProcessor) {}

  onStart(span: Span, parentContext: Context): void {
    const parent = trace.getSpan(parentContext);
    if (parent && !parent.spanContext().isRemote && !isAiLike(Reflect.get(parent, "attributes"))) {
      this.orphans.add(span.spanContext().spanId);
    }
    this.inner.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    const orphan = this.orphans.delete(span.spanContext().spanId);
    this.inner.onEnd(orphan ? (Object.create(span, { parentSpanContext: { value: undefined } }) as ReadableSpan) : span);
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush();
  }
}

function isAiLike(attributes: unknown): boolean {
  if (!attributes || typeof attributes !== "object") return false;
  return Object.keys(attributes).some(
    (key) => key === "openinference.span.kind" || key.startsWith("gen_ai.") || key.startsWith("ai."),
  );
}
