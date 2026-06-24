# Mastra Agent with Arize AX Tracing

Companion repo for the [Arize AX tutorial](https://arize.com/docs/ax/). A Mastra orchestrator agent instrumented with OpenInference tracing that sends spans to Arize AX.

## Prerequisites

- [Node.js](https://nodejs.org) v20.9.0+
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

This starts the Mastra dev server and opens the agent UI. Send requests to the weather orchestrator agent to generate traces — they will appear in your Arize AX space within a few seconds.

## How it works

A single orchestrator agent coordinates three tools in sequence:

```
User request
    ↓
Weather Orchestrator Agent
    ↓
weatherTool          → fetches current weather data
    ↓
weatherAnalysisTool  → analyzes weather patterns
    ↓
activityPlanningTool → recommends activities
    ↓
Final response
```

Tracing is configured in [src/mastra/index.ts](src/mastra/index.ts) using `@mastra/arize`, wired into Mastra's AI Tracing (`observability`) system. Every agent invocation and tool call is exported as an OpenInference span to `https://otlp.arize.com/v1/traces`.

## Project structure

```text
src/mastra/
├── index.ts                          # Mastra setup + Arize AX exporter
├── agents/
│   └── weather-orchestrator-agent.ts
└── tools/
    ├── weather-tool.ts
    ├── weather-analysis-tool.ts
    └── activity-planning-tool.ts
```

## Resources

- [Arize AX Documentation](https://arize.com/docs/ax/)
- [Mastra Arize exporter (`@mastra/arize`)](https://mastra.ai/en/docs/observability/ai-tracing/exporters/arize)
- [Mastra Documentation](https://mastra.ai/docs)
