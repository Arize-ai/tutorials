// In-code dual export from Node: Vercel AI SDK v7 + OpenInference, no collector.
//
//   one NodeTracerProvider
//     ├─ ArizeRootProcessor(OpenInferenceBatchSpanProcessor)
//     │                                   AI spans only, AI roots detached   -> Arize AX
//     └─ BatchSpanProcessor               every span, AI content redacted    -> Dynatrace
//   LoggerProvider -> Dynatrace     MeterProvider (delta) -> Dynatrace
//
// Run: npm install && npm start   (reads ../.env)

import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";
import { context, metrics, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { logs, SeverityNumber } from "@opentelemetry/api-logs";
import { OpenTelemetry } from "@ai-sdk/otel";
import { openai } from "@ai-sdk/openai";
import {
  enrichSpanWithOpenInference,
  isOpenInferenceSpan,
  OpenInferenceBatchSpanProcessor,
} from "@arizeai/openinference-vercel";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-proto";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-proto";
import { AggregationTemporalityPreference } from "@opentelemetry/exporter-metrics-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchLogRecordProcessor, LoggerProvider } from "@opentelemetry/sdk-logs";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { BatchSpanProcessor, type ReadableSpan, type SpanExporter } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { generateText, registerTelemetry, stepCountIs, tool } from "ai";
import { z } from "zod";
import { ArizeRootProcessor } from "./arize-root-processor.ts";

loadEnv(new URL("../.env", import.meta.url));

const env = (name: string) => {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set`);
  return value;
};

// ---------------------------------------------------------------- telemetry

const dtBase = `${env("DYNATRACE_ENV_URL").replace(/\/$/, "")}/api/v2/otlp`;
const dtToken = env("DYNATRACE_API_TOKEN");
const dtHeaders = { Authorization: `${dtToken.startsWith("dt0c") ? "Api-Token" : "Bearer"} ${dtToken}` };

const resource = resourceFromAttributes({
  "service.name": "ax-dt-example-ts",
  [SEMRESATTRS_PROJECT_NAME]: process.env.ARIZE_PROJECT_NAME ?? "ax-dt-incode-example",
});

registerTelemetry(new OpenTelemetry({ enrichSpan: enrichSpanWithOpenInference }));

const tracerProvider = new NodeTracerProvider({
  resource,
  spanProcessors: [
    new ArizeRootProcessor(
      new OpenInferenceBatchSpanProcessor({
        exporter: new OTLPTraceExporter({
          url: "https://otlp.arize.com/v1/traces",
          headers: { space_id: env("ARIZE_SPACE_ID"), api_key: env("ARIZE_API_KEY") },
        }),
        spanFilter: isOpenInferenceSpan,
        // Not reparentOrphanedSpans: it clears the parent on the shared live span, which
        // would also detach the AI subtree from the HTTP span in Dynatrace.
      }),
    ),
    new BatchSpanProcessor(redactAiContent(new OTLPTraceExporter({ url: `${dtBase}/v1/traces`, headers: dtHeaders }))),
  ],
});
tracerProvider.register();

const loggerProvider = new LoggerProvider({
  resource,
  processors: [new BatchLogRecordProcessor({ exporter: new OTLPLogExporter({ url: `${dtBase}/v1/logs`, headers: dtHeaders }) })],
});
logs.setGlobalLoggerProvider(loggerProvider);

const meterProvider = new MeterProvider({
  resource,
  readers: [
    new PeriodicExportingMetricReader({
      exporter: new OTLPMetricExporter({
        url: `${dtBase}/v1/metrics`,
        headers: dtHeaders,
        temporalityPreference: AggregationTemporalityPreference.DELTA,
      }),
      exportIntervalMillis: 10_000,
    }),
  ],
});
metrics.setGlobalMeterProvider(meterProvider);

// Every span still reaches Dynatrace; only prompt and completion text is removed.
function redactAiContent(inner: SpanExporter): SpanExporter {
  // Same list as the collector's transform/redact-ai-content.
  const content =
    /^(input\.value|output\.value|llm\.prompt_template\.variables|llm\.(input|output)_messages\..*|llm\.prompts?\..*|gen_ai\.(input|output)\.messages|gen_ai\.system_instructions|gen_ai\.tool\.call\.(arguments|result)|ai\.prompt.*|ai\.response\.text)$/;
  return {
    export: (spans, done) =>
      inner.export(
        spans.map((span) => {
          const kept = Object.fromEntries(Object.entries(span.attributes).filter(([key]) => !content.test(key)));
          // Same span, different attributes; the original object is shared with the Arize processor.
          return Object.create(span, { attributes: { value: kept } }) as ReadableSpan;
        }),
        done,
      ),
    shutdown: () => inner.shutdown(),
    forceFlush: () => inner.forceFlush?.() ?? Promise.resolve(),
  };
}

// ---------------------------------------------------------------- app

const tracer = trace.getTracer("ax-dt-example-ts");
const logger = logs.getLogger("ax-dt-example-ts");
const requests = metrics.getMeter("ax-dt-example-ts").createCounter("example.requests");
const db = new DatabaseSync(":memory:");
db.exec("CREATE TABLE customers (id TEXT, name TEXT); INSERT INTO customers VALUES ('c-42', 'Ada Lovelace');");

function log(body: string, attributes: Record<string, string | number> = {}) {
  // No trace_id attribute: the SDK stamps trace and span ids from the active context.
  logger.emit({ severityNumber: SeverityNumber.INFO, severityText: "INFO", body, attributes, context: context.active() });
}

async function ask(question: string) {
  return tracer.startActiveSpan(
    "POST /ask",
    { kind: SpanKind.SERVER, attributes: { "http.request.method": "POST", "http.route": "/ask" } },
    async (server) => {
      const traceId = server.spanContext().traceId;
      log("request received");
      const customer = tracer.startActiveSpan(
        "SELECT customers",
        { kind: SpanKind.CLIENT, attributes: { "db.system.name": "sqlite", "db.query.text": "SELECT name FROM customers WHERE id = ?" } },
        (span) => {
          const row = db.prepare("SELECT name FROM customers WHERE id = ?").get("c-42") as { name: string };
          span.end();
          return row;
        },
      );
      const result = await generateText({
        model: openai(process.env.OPENAI_MODEL ?? "gpt-4.1-mini"),
        system: `You are a concise support agent for ${customer.name}. Answer in one sentence.`,
        prompt: question,
        tools: {
          get_weather: tool({
            description: "Get the precipitation forecast for Paris.",
            inputSchema: z.object({ city: z.string() }),
            execute: async ({ city }) =>
              tracer.startActiveSpan("GET", { kind: SpanKind.CLIENT, attributes: { "http.request.method": "GET", "server.address": "api.open-meteo.com" } }, async (span) => {
                const response = await fetch("https://api.open-meteo.com/v1/forecast?latitude=48.85&longitude=2.35&daily=precipitation_probability_max&forecast_days=3");
                span.setAttribute("http.response.status_code", response.status);
                span.end();
                log(`weather lookup ${city} status=${response.status}`);
                return response.json();
              }),
          }),
        },
        stopWhen: stepCountIs(3),
        experimental_telemetry: { isEnabled: true, functionId: "support_agent" },
      });
      requests.add(1, { "http.route": "/ask", outcome: "ok" });
      log("request completed", { "gen_ai.usage.output_tokens": result.totalUsage.outputTokens ?? 0 });
      server.setStatus({ code: SpanStatusCode.OK });
      server.end();
      return { trace_id: traceId, answer: result.text };
    },
  );
}

try {
  const result = await ask("Will it rain in Paris in the next three days?");
  console.log(JSON.stringify({ ...result, mode: "incode-ts" }));
} finally {
  // Serverless: flush before the function returns, or the last batches are lost.
  await Promise.all([tracerProvider.shutdown(), loggerProvider.shutdown(), meterProvider.shutdown()]);
}

function loadEnv(file: URL) {
  try {
    for (const line of readFileSync(file, "utf8").split("\n")) {
      const match = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$/);
      if (match && process.env[match[1]] === undefined) process.env[match[1]] = match[2].trim();
    }
  } catch {
    // no .env; rely on the environment
  }
}
