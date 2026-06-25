# Mastra Agent with Arize AX Tracing

Companion repo for the Arize AX guide [Align LLM Evals with Human Judgment](https://arize.com/docs/ax/cookbooks/evaluate/align-llm-evals-with-human-judgment). A Mastra orchestrator agent instrumented with OpenInference tracing that sends spans to Arize AX.

## Prerequisites

- [Node.js](https://nodejs.org) v22.13.0+
- An [Arize AX](https://app.arize.com) account (Space ID and API key)
- An [OpenAI](https://platform.openai.com) API key

## Setup

```bash
npm install
```

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ARIZE_API_KEY="your-arize-api-key"
export ARIZE_SPACE_ID="your-arize-space-id"
```

## Run

```bash
npm start
```

This starts the Mastra dev server. Open [Mastra Studio](https://mastra.ai/docs/studio/overview) at [http://localhost:4111](http://localhost:4111) in your browser (Mastra picks the next free port if 4111 is taken), select the **WeatherOrchestratorAgent**, and send it a request such as "What should I do in London today?" to generate traces — they will appear in your Arize AX space within a few seconds.

### Generate a batch of example traces

To populate Arize AX with a diverse set of traces at once — covering each tool path (weather lookup, activity planning, and the full analysis-and-planning chain) — run:

```bash
npm run generate-traces
```

This sends a curated set of ~50 prompts to the orchestrator agent and flushes the resulting spans to Arize AX. The prompts live in [src/scripts/example-prompts.json](src/scripts/example-prompts.json), grouped by the tool path each one exercises — edit that file to add or change prompts.

## How it works

A single orchestrator agent coordinates three tools in sequence:

```text
User request
    ↓
Weather Orchestrator Agent
    ↓
weatherTool          → fetches current weather data (Open-Meteo)
    ↓
weatherAnalysisTool  → delegates to Weather Analysis Agent (LLM)
    ↓
activityPlanningTool → delegates to Activity Planning Agent (LLM)
    ↓
Final response
```

The analysis and planning tools don't call the model directly — they resolve a
dedicated worker agent from the Mastra instance (`mastra.getAgent(...)`) and call
`agent.generate(...)`. Because those calls run inside Mastra, their LLM spans are
captured as nested children of the tool span in the trace.

Tracing is configured in [src/mastra/index.ts](src/mastra/index.ts) using `@mastra/arize`, wired into Mastra's AI Tracing (`observability`) system. Every agent invocation and tool call is exported as an OpenInference span to `https://otlp.arize.com/v1/traces`.

## Project structure

```text
src/mastra/
├── index.ts                          # Mastra setup + Arize AX exporter
├── agents/
│   ├── weather-orchestrator-agent.ts # coordinates the tools
│   ├── weather-analysis-agent.ts     # LLM worker, called by weatherAnalysisTool
│   └── activity-planning-agent.ts    # LLM worker, called by activityPlanningTool
└── tools/
    ├── weather-tool.ts
    ├── weather-analysis-tool.ts
    └── activity-planning-tool.ts
```

## Resources

- [Arize AX Documentation](https://arize.com/docs/ax/)
- [Mastra Arize exporter (`@mastra/arize`)](https://mastra.ai/en/docs/observability/ai-tracing/exporters/arize)
- [Mastra Documentation](https://mastra.ai/docs)
