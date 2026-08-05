"""Run the whole system with one command: start both A2A agents, then send one query.

    python run_local.py "Should I buy NVDA stock?"

The two agents are started as a child process running a2a_servers.py, which is what they
are in production: independent services reached over HTTP. This script waits for both
agent cards to be served, sends one query through the orchestrator, prints the answer,
and stops the agents.

Use a2a_servers.py and orchestrator.py in two terminals instead when you want the agents
to stay up across several queries.
"""

import argparse
import asyncio
import subprocess
import sys

import httpx

import config
import tracing
from orchestrator import run_query

STARTUP_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 20


async def wait_for_agent_card(port: int) -> None:
    """Poll an agent's card endpoint until it responds, so the query never races startup."""
    url = f"http://127.0.0.1:{port}/.well-known/agent-card.json"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + STARTUP_TIMEOUT_S
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if loop.time() > deadline:
                raise TimeoutError(f"agent on port {port} did not come up")
            await asyncio.sleep(0.5)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        default="Should I buy NVDA stock? Give me both the risk and the opportunity case.",
    )
    args = parser.parse_args()

    tracing.setup_tracing()
    print("Starting the Bear and Bull A2A agents...")

    agents = subprocess.Popen(
        [sys.executable, "a2a_servers.py"],
        cwd=str(config.PROJECT_ROOT),
    )
    try:
        await asyncio.gather(
            wait_for_agent_card(config.BEAR_PORT),
            wait_for_agent_card(config.BULL_PORT),
        )
        print(f"Both agents up.\n\nQuery: {args.query}\n")

        result = await run_query(args.query)
        print(f"Result:\n{result}")
        return 0 if result else 1
    finally:
        # SIGTERM lets uvicorn shut down gracefully and close each agent's MCP subprocess.
        agents.terminate()
        try:
            agents.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            agents.kill()
        tracing.flush()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
