"""Orchestrator: coordinate the two A2A specialists and answer a question.

The orchestrator never imports the specialists. It reaches them over A2A through
RemoteA2aAgent, which fetches each agent card and calls the remote agent as a tool. That
is what makes the two frameworks interchangeable behind the protocol.

Run a2a_servers.py first, then:

    python orchestrator.py "Should I buy NVDA stock?"
"""

import argparse
import asyncio

from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

import config
import tracing

USER_ID = "local_user"
SESSION_ID = "orchestrator_session"


def build_orchestrator() -> LlmAgent:
    """Build the orchestrator with both remote specialists wrapped as tools."""
    config.init_vertex()

    remote_bear = RemoteA2aAgent(
        name="bear_risk_analyst",
        description="Analyzes downside risks and warning signals for a stock",
        agent_card=f"http://localhost:{config.BEAR_PORT}{AGENT_CARD_WELL_KNOWN_PATH}",
    )
    remote_bull = RemoteA2aAgent(
        name="bull_market_analyst",
        description="Identifies growth opportunities and bullish patterns for a stock",
        agent_card=f"http://localhost:{config.BULL_PORT}{AGENT_CARD_WELL_KNOWN_PATH}",
    )

    return LlmAgent(
        name="trading_strategy_orchestrator",
        model=config.orchestrator_model(),
        instruction=config.ORCHESTRATOR_INSTRUCTION,
        tools=[AgentTool(agent=remote_bear), AgentTool(agent=remote_bull)],
    )


async def run_query(query: str) -> str | None:
    """Send one query through the orchestrator and return its final answer."""
    orchestrator = build_orchestrator()
    runner = Runner(
        app_name=orchestrator.name,
        agent=orchestrator,
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name=orchestrator.name,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_result = None
    async for event in runner.run_async(
        session_id=session.id, user_id=USER_ID, new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_result = "".join(
                    part.text for part in event.content.parts if getattr(part, "text", None)
                )
            break

    return final_result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        default="Should I buy NVDA stock? Give me both the risk and the opportunity case.",
    )
    args = parser.parse_args()

    tracing.setup_tracing()

    print(f"Query: {args.query}\n")
    result = await run_query(args.query)
    print(f"Result:\n{result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        tracing.flush()
