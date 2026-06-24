import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';
import { Observability } from '@mastra/observability';
import { ArizeExporter } from '@mastra/arize';
// Import orchestrator/worker agents - this is the only workflow pattern now
import { weatherOrchestratorAgent } from './agents/weather-orchestrator-agent';

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
    // File-backed so Mastra's internal workflow/scheduler tables (e.g.
    // mastra_workflow_snapshot) persist and are shared across connections.
    // ":memory:" gives each libsql connection its own DB, which breaks them.
    id: "mastra-storage",
    url: "file:../mastra.db",
  }),
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'info',
  }),
  // Mastra AI Tracing: export every agent and tool span to Arize AX via the
  // native @mastra/arize exporter (replaces the deprecated legacy telemetry path).
  observability: new Observability({
    configs: {
      arize: {
        serviceName: "mastra-orchestrator-workflow",
        exporters: [
          new ArizeExporter({
            spaceId: ARIZE_SPACE_ID,
            apiKey: ARIZE_API_KEY,
            projectName: "mastra-orchestrator-workflow",
          }),
        ],
      },
    },
  }),
});
