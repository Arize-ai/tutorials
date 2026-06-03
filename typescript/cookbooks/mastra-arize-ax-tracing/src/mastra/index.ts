import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';
// Import orchestrator/worker agents - this is the only workflow pattern now
import { weatherOrchestratorAgent } from './agents/weather-orchestrator-agent';

import {
  isOpenInferenceSpan,
  OpenInferenceOTLPTraceExporter,
} from "@arizeai/openinference-mastra";

const ARIZE_SPACE_ID = process.env.ARIZE_SPACE_ID;
const ARIZE_API_KEY = process.env.ARIZE_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!ARIZE_SPACE_ID || !ARIZE_API_KEY || !OPENAI_API_KEY) {
  throw new Error(
    "Missing required env vars: ARIZE_SPACE_ID, ARIZE_API_KEY, and OPENAI_API_KEY must all be set"
  );
}

export const mastra = new Mastra({
  // No workflows - using pure orchestrator/worker agent pattern
  agents: {
    // Orchestrator agent that coordinates the entire workflow
    weatherOrchestratorAgent
  },
  storage: new LibSQLStore({
    // stores telemetry, evals, ... into memory storage, if it needs to persist, change to file:../mastra.db
    url: ":memory:",
  }),
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'info',
  }),
  telemetry: {
    enabled: true,
    serviceName: "mastra-orchestrator-workflow",
    export: {
      type: "custom",
      exporter: new OpenInferenceOTLPTraceExporter({
        url: "https://otlp.arize.com/v1/traces",
        headers: {
          "arize-space-id": ARIZE_SPACE_ID,
          "arize-api-key": ARIZE_API_KEY,
        },
        spanFilter: isOpenInferenceSpan,
      }),
    },
  },
});
