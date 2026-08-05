"""Bull Agent: opportunity analysis on Google ADK, exposed over A2A.

ADK ships its own A2A executor, so unlike the Bear agent this one needs no hand-written
bridge: wrap the agent in a Runner, hand that to A2aAgentExecutor, and ADK speaks A2A.
"""

from a2a.types import AgentSkill
from google.adk import Runner
from google.adk.a2a.executor.a2a_agent_executor import (
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
)
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

import config

BULL_SKILLS = [
    AgentSkill(
        id="breakout_detection",
        name="Breakout Pattern Detection",
        description="Identify bullish breakout patterns",
        tags=["technical-analysis", "breakouts"],
        examples=["Find breakout patterns for NVDA"],
    ),
    AgentSkill(
        id="momentum_screening",
        name="Momentum Screening",
        description="Screen for stocks with strong momentum",
        tags=["momentum", "screening"],
        examples=["Find high momentum tech stocks"],
    ),
    AgentSkill(
        id="entry_signals",
        name="Entry Signal Detection",
        description="Detect optimal entry points",
        tags=["entry-points", "timing"],
        examples=["When should I buy AAPL?"],
    ),
]


def create_bull_agent_card():
    """Create the A2A Agent Card that advertises the Bull agent's skills."""
    return create_agent_card(
        agent_name="Bull Market Analyst (ADK + MCP)",
        description=(
            "An optimistic analyst powered by Google ADK, "
            "focused on growth opportunities and bullish patterns."
        ),
        skills=BULL_SKILLS,
    )


def build_bull_agent() -> LlmAgent:
    """Build the ADK agent with its MCP toolset attached."""
    config.init_vertex()

    command, args = config.mcp_server_command("mcp_tools.bull_mcp_server")
    toolset = MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=command,
                args=args,
                cwd=str(config.PROJECT_ROOT),
            ),
            timeout=60,
        ),
    )

    return LlmAgent(
        name="bull_market_analyst",
        model=config.bull_model(),
        description="Optimistic analyst focused on growth opportunities and bullish signals.",
        instruction=config.BULL_SYSTEM_PROMPT,
        tools=[toolset],
    )


def build_bull_executor() -> A2aAgentExecutor:
    """Build the ADK A2A executor for the Bull agent.

    Called by a2a_servers.py locally and by Agent Engine after deployment. Tracing is
    configured here rather than at import time so that a deployed copy instruments the
    process it actually runs in.
    """
    import tracing

    tracing.setup_tracing()

    agent = build_bull_agent()
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        session_service=InMemorySessionService(),
    )
    return A2aAgentExecutor(runner=runner, config=A2aAgentExecutorConfig())
