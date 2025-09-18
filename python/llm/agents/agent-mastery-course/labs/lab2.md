# Lab 2 – Observability with Arize

Goal: instrument the AI Trip Planner agent with Arize tracing and confirm spans are flowing into your workspace.

## 1. Prerequisites
- Lab 1 completed and the agent runs locally on `http://localhost:8000`
- Arize account (free tier is sufficient)
- `ARIZE_SPACE_ID` and `ARIZE_API_KEY` ready to add to `.env`
- Optional: Cursor or Claude Code with MCP support for quick setup

## 2. Enable Arize tracing locally
1. In `backend/.env`, add:
   ```bash
   ARIZE_SPACE_ID=your_space_id
   ARIZE_API_KEY=your_api_key
   ```
2. Restart the backend (`./start.sh` or `uvicorn main:app ...`).
3. Watch the backend logs: you should see confirmation that Arize tracing initialized successfully.

## 3. Install the Arize Tracing Assistant (optional but recommended if you are changing the agent)
- Cursor: `Cursor Settings → Features → MCP → New MCP Server`
  ```json
  {
    "mcpServers": {
      "arize-tracing-assistant": {
        "command": "uvx",
        "args": ["arize-tracing-assistant@latest"]
      }
    }
  }
  ```
  Save, then restart Cursor and ensure the server is enabled.
- Claude Code:
  ```bash
  claude mcp add arize-tracing-assistant uvx 'arize-tracing-assistant@latest'
  ```

## 4. Run the agent
1. From the frontend, submit at least one trip request.
2. Confirm a `200` response with an itinerary payload.
3. The backend logs should show spans being exported via Arize.

## 5. Verify in Arize
1. Log in to https://app.arize.com
2. Select your space, then open **Tracing → Live**
3. Filter by project name `ai-trip-planner` (default) or the value set in `main.py`
4. You should see the recent request with agent, tool, and LLM spans.

## 6. Optional enhancements
- Capture a screenshot of the full trace graph for documentation
- Add Phoenix instrumentation side-by-side for comparison
- Configure evaluations or alerts once multiple traces are available

Checkpoint: share the trace URL or screenshot; you are ready to move on to the next module.
