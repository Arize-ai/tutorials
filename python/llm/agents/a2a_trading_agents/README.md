# A2A Trading Agents: Google ADK, Pydantic AI, MCP, and Arize AX

A trading analysis system where two specialist agents built on **different frameworks**
collaborate over the **Agent-to-Agent (A2A) protocol**, each with its own **MCP** tools,
all traced to **Arize AX**.

The point of the example is that the orchestrator does not know or care what framework
either specialist is built on. It discovers them through their A2A agent cards and calls
them as tools, so a Pydantic AI agent and a Google ADK agent are interchangeable behind
the protocol.

Companion guide: [Tracing an A2A Agent](https://arize.com/docs/ax/cookbooks/advanced-workflows/tracing-a2a-agent)

## Architecture

| Component | Framework | Tools | Role |
| :--- | :--- | :--- | :--- |
| Bear Risk Analyst | Pydantic AI | `risk_scanner`, `divergence_detector`, `exit_signal_monitor` | Downside catalysts and warning signals |
| Bull Market Analyst | Google ADK | `find_breakout_patterns`, `momentum_screener`, `entry_signal_detector` | Growth opportunities and bullish patterns |
| Orchestrator | Google ADK | The two agents above, as A2A tools | Coordinates both and weighs the cases |

Each specialist runs as an A2A HTTP service that publishes an agent card at
`/.well-known/agent-card.json`. Market data is synthetic, so no market data feed or API
key is needed for the tools.

```
orchestrator.py
  |
  |-- A2A --> localhost:8001  Bear (Pydantic AI)  --stdio--> mcp_tools/bear_mcp_server.py
  |
  '-- A2A --> localhost:8002  Bull (Google ADK)   --stdio--> mcp_tools/bull_mcp_server.py
```

## Files

| File | What it does |
| :--- | :--- |
| `config.py` | Environment-driven configuration and model selection for all three agents |
| `tracing.py` | Arize AX tracing for both frameworks in one tracer provider |
| `mcp_tools/` | The MCP servers and their synthetic market-data generator |
| `trading_agents/bear_agent.py` | Pydantic AI agent, its agent card, and the A2A executor that bridges it |
| `trading_agents/bull_agent.py` | ADK agent, its agent card, and ADK's built-in A2A executor |
| `a2a_servers.py` | Serves both agents as A2A services |
| `orchestrator.py` | Discovers both agents over A2A and answers one question |
| `run_local.py` | Starts the agents and sends one query, in a single command |
| `deploy_agent_engine.py` | Deploys both agents to Vertex AI Agent Engine (Vertex only) |

## Prerequisites

Python 3.10 or later and an [Arize AX account](https://app.arize.com/auth/join).

The agents run on Vertex AI: Gemini 2.5 Flash for the Bear agent and orchestrator, Llama
3.3 70B for the Bull agent. Vertex AI has no API key, so it needs:

- A Google Cloud project with billing and the [Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) enabled
- Application Default Credentials: `gcloud auth application-default login`
- Llama 3.3 accepted in [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden), which is a per-model license step
- For `deploy_agent_engine.py` only: a GCS staging bucket and permission to create Agent Engine resources

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ARIZE_SPACE_ID, ARIZE_API_KEY, and GOOGLE_CLOUD_PROJECT
```

## Run it

One command, which starts both agents and sends a single query:

```bash
python run_local.py "Should I buy NVDA stock?"
```

Or keep the agents up across several queries, in two terminals:

```bash
python a2a_servers.py                                  # terminal 1
python orchestrator.py "What are the risks for TSLA?"  # terminal 2
```

The answer comes back with both cases argued from the tools' output, for example a risk
score and stop-loss levels from the Bear agent alongside breakout targets and an entry
price from the Bull agent.

## What you see in Arize AX

Open the `a2a-trading-agents` project. A single query produces roughly 100 spans:

- **`CHAIN`** `invocation [trading_strategy_orchestrator]`, the orchestrator run
- **`AGENT`** and **`LLM`** spans for each agent's reasoning, with the model name attached
- **`TOOL`** spans for the A2A calls (`execute_tool bear_risk_analyst`) and for every MCP
  tool the specialists invoke (`execute_tool risk_scanner`, `tools/call risk_scanner`)

Two things about this trace shape are worth knowing before you go looking for them:

**The agents' work lands in separate traces from the orchestrator's.** A2A does not
propagate trace context across the HTTP hop, so one query produces one orchestrator trace
plus one trace per agent that answered, rather than a single connected tree. Group them by
time or by the project rather than expecting one root span to cover the whole exchange.

**The a2a-sdk emits its own internal spans.** Event-queue plumbing
(`EventQueue.dequeue_event` and friends) accounts for most of the span count and carries no
OpenInference span kind, so those rows sit uncategorized in Arize AX. Filter on
`attributes.openinference.span.kind` to get to the agent behavior.

## Deploy to Vertex AI Agent Engine

Turns each agent into a managed service with an authenticated A2A endpoint. Vertex only.

```bash
python deploy_agent_engine.py                                    # deploy both
python deploy_agent_engine.py --query "Analyze risks for TSLA"   # deploy, then query
python deploy_agent_engine.py --delete <resource> <resource>     # tear down
```

Deployment takes several minutes per agent and leaves billable resources running. Delete
them when you are finished.

## Notes

**`a2a-sdk` is pinned below 0.4.** `google-adk` requires `a2a-sdk>=0.3.4,<0.4.0`, and
`a2a-sdk` 1.x removed `a2a.server.apps`, `a2a.types.TextPart`,
`a2a.types.TransportProtocol`, and `a2a.utils.new_agent_text_message`. An unpinned install
resolves to 1.x, and then nothing imports. Install from `requirements.txt`.

**Shutdown prints OpenTelemetry warnings.** After the answer, you will see
`ValueError: <Token ...> was created in a different Context` from
`opentelemetry/context/contextvars_context.py`. It comes from the MCP client's async
generators being finalized as the event loop closes, it happens after all work and all
span exports are done, and the process still exits 0. It is noise, not a failure.

**What has and has not been run.** The A2A protocol, the MCP servers and all six tools,
the agent cards, both executors, the orchestrator, and Arize AX tracing were confirmed end
to end, producing 486 spans across 13 traces. That was done against a temporary model
binding, since none of those layers depend on which model answers.

The Vertex model bindings themselves were not executed: `init_vertex()`,
`GoogleProvider(vertexai=True)`, `GoogleModel("gemini-2.5-flash")`,
`LiteLlm("vertex_ai/meta/llama-3.3-70b-instruct-maas")`, and every line of
`deploy_agent_engine.py`, including `agent_engines.create()` and the `GoogleAuth` flow.
They follow the Vertex API and their imports and signatures were checked against the
installed SDKs, but confirming they run needs a billed Google Cloud project with Llama 3.3
accepted in Model Garden. Treat that path as reviewed, not exercised.
