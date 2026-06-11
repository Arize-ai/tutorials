// instrumentation.ts
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { BeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import * as beeaiFramework from "beeai-framework";


diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.ERROR);

const SERVICE_NAME = "beeai";
// Initialize Instrumentation Manually
const beeAIInstrumentation = new BeeAIInstrumentation();

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: SERVICE_NAME,
    // defaults to "default" in the Phoenix UI
    [SEMRESATTRS_PROJECT_NAME]: SERVICE_NAME,
  }),
  spanProcessors: [
    // BatchSpanProcessor will flush spans in batches after some time,
    // this is recommended in production. For development or testing purposes
    // you may try SimpleSpanProcessor for instant span flushing to the Phoenix UI.
    new BatchSpanProcessor(
      new OTLPTraceExporter({
        // PHOENIX_COLLECTOR_ENDPOINT is the workspace base URL, e.g.
        // https://app.phoenix.arize.com/s/<workspace> for Phoenix Cloud
        // or http://localhost:6006 for self-hosted Phoenix.
        url: `${process.env.PHOENIX_COLLECTOR_ENDPOINT ?? "http://localhost:6006"}/v1/traces`,
        headers: { authorization: `Bearer ${process.env.PHOENIX_API_KEY ?? ""}` },
      })
    ),
  ],
});

provider.register();


console.log("🔧 Manually instrumenting BeeAgent...");
beeAIInstrumentation.manuallyInstrument(beeaiFramework);
console.log("✅ BeeAgent manually instrumented.");

// eslint-disable-next-line no-console
console.log("👀 OpenInference initialized");
