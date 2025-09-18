# MCP Weather Demo

This backend now exposes a tiny Model Context Protocol (MCP) showcase that the `research_agent` can lean on for fresh weather color. Like the RAG flow, it stays dormant until you opt in.

## 1. Enable it

1. Copy `backend/.env.example` to `backend/.env` if you have not already.
2. Opt in by adding the flag to your `.env`:
   ```bash
   ENABLE_MCP=1
   ```
3. Restart the FastAPI server so the MCP server/client pair spins up alongside tracing.
   The backend will raise at startup if the packages are missing or Python is <3.10, making the dependency explicit for the demo.

## 2. What happens when enabled

- Startup wires a lightweight in-process FastMCP server with a single tool (`weather_mcp`) plus an in-memory client created via `mcp.shared.memory.create_connected_server_and_client_session`. The tool returns deterministic weather guidance so the demo stays offline friendly.
- **Critical for tracing**: `openinference.instrumentation.mcp.MCPInstrumentor` is registered **before** any MCP clients are created. This is essential for proper context propagation.
- The MCP tool only needs the standard `@server.tool()` decorator - the MCP instrumentation handles tracing automatically
- **Context propagation**: The MCP instrumentation provides automatic context propagation between MCP clients and servers to unify traces
- Each `/plan-trip` request now triggers an actual MCP tool round-trip. The research agent records the tool call in the API response, and the MCP instrumentation ensures proper context propagation to Arize AX.


## 3. implementation details

**Tracing initialization order** (critical for MCP):
1. Import all tracing packages at the top
2. Register `MCPInstrumentor().instrument(tracer_provider)` immediately 
3. **Then** create MCP clients - never before instrumentation

**Automatic MCP instrumentation** (no manual spans needed):
```python
@server.tool(name="weather_mcp", description="...")
def weather_tool(destination: str) -> str:
    # The MCP instrumentation handles tracing automatically
    # through context propagation - no manual spans needed!
    return build_weather_summary(destination)
```

The key is that `MCPInstrumentor` provides **context propagation only** - it doesn't create spans but ensures proper parent-child relationships between MCP client/server operations.

Toggle `ENABLE_MCP=0` to remove the tool and its spans, returning the research agent to its original two-tool setup.
