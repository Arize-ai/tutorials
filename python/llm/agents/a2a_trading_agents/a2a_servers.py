"""Serve the Bear and Bull agents as A2A servers.

Each agent becomes an HTTP service that publishes an agent card at
/.well-known/agent-card.json and accepts A2A task requests. Run this in one terminal,
then run orchestrator.py in another.

    python a2a_servers.py
"""

import asyncio

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import TransportProtocol

import config
import tracing
from trading_agents.bear_agent import BearAgentExecutor, create_bear_agent_card
from trading_agents.bull_agent import build_bull_executor, create_bull_agent_card


def build_bear_app() -> A2AStarletteApplication:
    """Wrap the Pydantic AI Bear agent in an A2A server using our own executor."""
    card = create_bear_agent_card()
    card.url = f"http://localhost:{config.BEAR_PORT}"
    card.preferred_transport = TransportProtocol.jsonrpc

    handler = DefaultRequestHandler(
        agent_executor=BearAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=card, http_handler=handler)


def build_bull_app() -> A2AStarletteApplication:
    """Wrap the ADK Bull agent in an A2A server using ADK's built-in executor."""
    card = create_bull_agent_card()
    card.url = f"http://localhost:{config.BULL_PORT}"
    card.preferred_transport = TransportProtocol.jsonrpc

    handler = DefaultRequestHandler(
        agent_executor=build_bull_executor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=card, http_handler=handler)


def make_server(app: A2AStarletteApplication, port: int) -> uvicorn.Server:
    """Build a uvicorn server for one A2A application.

    Returned rather than served immediately so a caller can trigger a graceful shutdown
    via server.should_exit. Cancelling the serve() task instead tears down the agents'
    MCP subprocesses from the wrong task and produces anyio cancel-scope errors.
    """
    return uvicorn.Server(
        uvicorn.Config(
            app.build(),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            loop="none",  # reuse the caller's event loop
        )
    )


async def serve(app: A2AStarletteApplication, port: int) -> None:
    """Serve one A2A application on the given port until the process is interrupted."""
    await make_server(app, port).serve()


async def main() -> None:
    tracing.setup_tracing()

    print(f"Bear Agent (Pydantic AI, {config.BEAR_MODEL}) -> http://127.0.0.1:{config.BEAR_PORT}")
    print(f"Bull Agent (ADK, {config.BULL_MODEL}) -> http://127.0.0.1:{config.BULL_PORT}")

    await asyncio.gather(
        serve(build_bear_app(), config.BEAR_PORT),
        serve(build_bull_app(), config.BULL_PORT),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down A2A servers.")
    finally:
        tracing.flush()
