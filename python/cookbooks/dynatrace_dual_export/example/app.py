"""Support-desk example: one request, one trace, two backends.

POST /ask produces, under one trace id:
  POST /ask                  HTTP server span          (FastAPI instrumentation)
  SELECT                     database span, customers  (sqlite3 instrumentation)
  support_agent              AGENT span                (OpenInference, manual)
    ChatCompletion           LLM span                  (OpenInference OpenAI instrumentor)
    get_order_status         TOOL span                 (OpenInference, manual)
      SELECT                 database span, orders
    execute_tool get_weather TOOL span                 (OTel GenAI semantic conventions, manual)
      GET                    outbound HTTP span        (httpx instrumentation)
    ChatCompletion           LLM span
plus correlated logs and three metrics. With NESTED_AGENTS=1, a `triage_agent` AGENT
span sits between `POST /ask` and `support_agent`.

Run once and print the trace id:
    python example/app.py --mode collector
    python example/app.py --mode incode
Serve until stopped:
    python example/app.py --mode collector --serve
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(ROOT / ".env", override=False)

import telemetry  # noqa: E402

DB_PATH = Path(tempfile.gettempdir()) / "ax-dt-example.db"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_QUESTION = "Where is my order 1001, and will it rain in Paris when it arrives?"
# Every trace has a root in Arize without app changes: the collector's traces/arize-root
# branch sends the request span as a CHAIN, and in incode mode OpenInferenceSpanProcessor
# makes the agent span the root. AI_ROOT=server also tags the request span as a CHAIN
# with the question and answer, so the Arize root shows the request input and output.
AI_ROOT = os.getenv("AI_ROOT", "none")
NESTED_AGENTS = os.getenv("NESTED_AGENTS", "").lower() in ("1", "true", "yes")
CITIES = {"paris": (48.8566, 2.3522), "berlin": (52.52, 13.405), "new york": (40.7128, -74.006)}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the precipitation forecast for a city for the next days.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


class Ask(BaseModel):
    question: str = DEFAULT_QUESTION
    customer_id: str = "c-42"


def seed_db() -> None:
    # Runs before the sqlite3 instrumentation is enabled, so seeding is not traced.
    with sqlite3.connect(DB_PATH) as db:
        db.executescript(
            """
            DROP TABLE IF EXISTS customers;
            DROP TABLE IF EXISTS orders;
            CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT, tier TEXT);
            CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, eta TEXT, city TEXT);
            INSERT INTO customers VALUES ('c-42', 'Ada Lovelace', 'gold');
            INSERT INTO orders VALUES ('1001', 'c-42', 'in transit', date('now', '+2 day'), 'Paris');
            """
        )


def build_app():
    import httpx
    from fastapi import FastAPI
    from openai import AsyncOpenAI
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    log = logging.getLogger("ax_dt_example")
    tracer = trace.get_tracer("ax_dt_example")
    meter = metrics.get_meter("ax_dt_example")
    requests_total = meter.create_counter("example.requests", unit="{request}", description="Requests to /ask")
    agent_duration = meter.create_histogram("example.agent.duration", unit="ms", description="Agent run time")
    tokens_total = meter.create_counter("example.llm.tokens", unit="{token}", description="LLM tokens used")

    client = AsyncOpenAI()
    http = httpx.AsyncClient(timeout=10)
    app = FastAPI()

    def query(sql: str, *params):
        # cursor().execute, not Connection.execute: only the cursor path is instrumented.
        with sqlite3.connect(DB_PATH) as db:
            return db.cursor().execute(sql, params).fetchone()

    async def get_order_status(order_id: str) -> dict:
        with tracer.start_as_current_span(
            "get_order_status",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "get_order_status",
                "tool.parameters": json.dumps({"order_id": order_id}),
                "input.value": json.dumps({"order_id": order_id}),
                "input.mime_type": "application/json",
            },
        ) as span:
            row = query("SELECT id, status, eta, city FROM orders WHERE id = ?", order_id)
            result = dict(zip(("order_id", "status", "eta", "city"), row)) if row else {"error": "not found"}
            log.info("order lookup %s -> %s", order_id, result.get("status", "not found"),
                     extra={"order.id": order_id, "db.rows": int(bool(row))})
            span.set_attribute("output.value", json.dumps(result))
            span.set_attribute("output.mime_type", "application/json")
            return result

    async def get_weather(city: str, call_id: str) -> dict:
        # OTel GenAI semantic conventions instead of OpenInference: Arize maps
        # gen_ai.operation.name=execute_tool to a TOOL span at ingestion.
        arguments = json.dumps({"city": city})
        with tracer.start_as_current_span(
            "execute_tool get_weather",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.id": call_id,
                "gen_ai.tool.call.arguments": arguments,
            },
        ) as span:
            lat, lon = CITIES.get(city.lower(), CITIES["paris"])
            response = await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "daily": "precipitation_probability_max", "forecast_days": 4},
            )
            response.raise_for_status()
            daily = response.json().get("daily", {})
            result = {"city": city, "days": daily.get("time", []), "rain_probability_pct": daily.get("precipitation_probability_max", [])}
            log.info("weather lookup %s status=%s", city, response.status_code,
                     extra={"http.response.status_code": response.status_code, "weather.city": city})
            span.set_attribute("gen_ai.tool.call.result", json.dumps(result))
            return result

    async def run_agent(question: str, customer: dict) -> str:
        started = time.perf_counter()
        with tracer.start_as_current_span(
            "support_agent",
            attributes={
                "openinference.span.kind": "AGENT",
                "agent.name": "support_agent",
                "input.value": question,
                "input.mime_type": "text/plain",
            },
        ) as span:
            messages = [
                {"role": "system", "content": f"You are a concise support agent. The customer is {customer['name']} ({customer['tier']} tier). Use tools; answer in two sentences."},
                {"role": "user", "content": question},
            ]
            answer = ""
            for step in range(1, 4):
                completion = await client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
                usage = completion.usage
                if usage:
                    tokens_total.add(usage.prompt_tokens, {"gen_ai.token.type": "input", "gen_ai.request.model": MODEL})
                    tokens_total.add(usage.completion_tokens, {"gen_ai.token.type": "output", "gen_ai.request.model": MODEL})
                message = completion.choices[0].message
                calls = message.tool_calls or []
                log.info("agent step %d: model=%s tool_calls=%d", step, completion.model, len(calls),
                         extra={"agent.step": step, "gen_ai.response.model": completion.model,
                                "gen_ai.usage.input_tokens": usage.prompt_tokens if usage else 0,
                                "gen_ai.usage.output_tokens": usage.completion_tokens if usage else 0})
                if not calls:
                    answer = message.content or ""
                    break
                messages.append(message.model_dump(exclude_none=True))
                for call in calls:
                    args = json.loads(call.function.arguments or "{}")
                    if call.function.name == "get_order_status":
                        result = await get_order_status(str(args.get("order_id", "")))
                    elif call.function.name == "get_weather":
                        result = await get_weather(str(args.get("city", "Paris")), call.id)
                    else:
                        result = {"error": f"unknown tool {call.function.name}"}
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            span.set_attribute("output.value", answer)
            span.set_attribute("output.mime_type", "text/plain")
            span.set_status(Status(StatusCode.OK))
            agent_duration.record((time.perf_counter() - started) * 1000, {"agent.name": "support_agent"})
            return answer

    async def triage_agent(question: str, customer: dict) -> str:
        # An outer agent that hands the question to support_agent.
        with tracer.start_as_current_span(
            "triage_agent",
            attributes={
                "openinference.span.kind": "AGENT",
                "agent.name": "triage_agent",
                "input.value": question,
                "input.mime_type": "text/plain",
            },
        ) as span:
            answer = await run_agent(question, customer)
            span.set_attribute("output.value", answer)
            span.set_attribute("output.mime_type", "text/plain")
            return answer

    @app.post("/ask")
    async def ask(body: Ask):
        server_span = trace.get_current_span()
        trace_id = format(server_span.get_span_context().trace_id, "032x")
        if AI_ROOT == "server":
            server_span.set_attributes({
                "openinference.span.kind": "CHAIN",
                "input.value": body.question,
                "input.mime_type": "text/plain",
            })
        log.info("request received for customer %s", body.customer_id, extra={"customer.id": body.customer_id})
        try:
            row = query("SELECT name, tier FROM customers WHERE id = ?", body.customer_id)
            customer = {"name": row[0], "tier": row[1]} if row else {"name": "unknown", "tier": "none"}
            answer = await (triage_agent if NESTED_AGENTS else run_agent)(body.question, customer)
        except Exception as error:
            log.exception("request failed: %s", error)
            requests_total.add(1, {"http.route": "/ask", "outcome": "error"})
            raise
        requests_total.add(1, {"http.route": "/ask", "outcome": "ok"})
        if AI_ROOT == "server":
            server_span.set_attribute("output.value", answer)
            server_span.set_attribute("output.mime_type", "text/plain")
        log.info("request completed", extra={"customer.id": body.customer_id, "answer.length": len(answer)})
        return {"trace_id": trace_id, "answer": answer}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


def instrument_libraries() -> None:
    # Before build_app(): these patch the client classes and sqlite3.connect.
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

    OpenAIInstrumentor().instrument()
    SQLite3Instrumentor().instrument()
    HTTPXClientInstrumentor().instrument()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default=os.getenv("TELEMETRY_MODE", "collector"), choices=["collector", "incode"])
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--serve", action="store_true", help="keep serving instead of sending one request")
    args = parser.parse_args()

    seed_db()
    tel = telemetry.setup(args.mode)
    instrument_libraries()
    app = build_app()

    import uvicorn
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz", exclude_spans=["receive", "send"])

    if args.serve:
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
        finally:
            tel.shutdown()
        return

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/ask",
            data=json.dumps({"question": args.question}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        tel.shutdown()
    result["mode"] = args.mode
    print(json.dumps(result))


if __name__ == "__main__":
    main()
