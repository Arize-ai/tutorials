"""Bear Agent: risk analysis on Pydantic AI, exposed over A2A.

Pydantic AI has no built-in A2A server, so this module supplies the AgentExecutor that
bridges it: BearAgentExecutor translates an A2A task into an agent run and reports
progress back through the A2A TaskUpdater.
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

import config

BEAR_SKILLS = [
    AgentSkill(
        id="risk_analysis",
        name="Risk Factor Scanner",
        description="Identifies potential downside catalysts and risk factors",
        tags=["Risk-Analysis", "Market-Analysis"],
        examples=[
            "What are the key risks for NVDA?",
            "Analyze downside catalysts for tech stocks",
        ],
    ),
    AgentSkill(
        id="divergence_detection",
        name="Divergence Detection",
        description="Finds bearish divergences and technical weakness signals",
        tags=["Technical-Analysis", "Divergence"],
        examples=["Find bearish divergences in AAPL"],
    ),
    AgentSkill(
        id="exit_signals",
        name="Exit Signal Monitoring",
        description="Tracks distribution patterns and exit signals",
        tags=["Exit-Strategy", "Risk-Management"],
        examples=["Monitor exit signals for NVDA"],
    ),
]


def create_bear_agent_card():
    """Create the A2A Agent Card that advertises the Bear agent's skills."""
    return create_agent_card(
        agent_name="Bear Risk Analyst (Pydantic AI + MCP)",
        description=(
            "A cautious risk analyst powered by Pydantic AI, "
            "focused on identifying downside catalysts and warning signals."
        ),
        skills=BEAR_SKILLS,
    )


def build_bear_agent():
    """Build the Pydantic AI agent with its MCP toolset attached."""
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset, StdioTransport

    config.init_vertex()

    command, args = config.mcp_server_command("mcp_tools.bear_mcp_server")
    toolset = MCPToolset(
        StdioTransport(command=command, args=args, cwd=str(config.PROJECT_ROOT))
    )

    # Instrumentation is not set here: tracing.setup_tracing() calls
    # Agent.instrument_all(), which covers every agent in the process.
    return Agent(
        model=config.bear_model(),
        system_prompt=config.BEAR_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=3,
    )


class BearAgentExecutor(AgentExecutor):
    """A2A executor for the Bear agent.

    The agent is built lazily rather than in __init__ because Agent Engine pickles the
    executor to deploy it, and an initialized agent holding an MCP subprocess is not
    picklable. Building on first execute() also means tracing is configured inside the
    process that actually serves traffic.
    """

    def __init__(self):
        self.agent = None
        self._traced = False

    def _init_agent(self):
        if not self._traced:
            import tracing

            tracing.setup_tracing()
            self._traced = True

        if self.agent is None:
            self.agent = build_bear_agent()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one A2A task: analyze the requested symbol's downside risk."""
        query = context.get_user_input()
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not getattr(context, "current_task", None):
            await updater.submit()
        await updater.start_work()

        try:
            # Inside the try: a build failure (bad credentials, missing model access) is
            # reported to the caller as a failed task rather than escaping as a
            # server-level JSON-RPC error the orchestrator cannot interpret.
            self._init_agent()

            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Analyzing risks..."),
            )

            result = await self.agent.run(query)
            result_text = getattr(result, "output", None) or str(result)

            separator = "=" * 50
            response = f"""
BEAR RISK ANALYSIS
{separator}

{result_text}
"""
            await updater.add_artifact([TextPart(text=response)], name="risk_analysis")
            await updater.complete()

        except Exception as exc:  # surface the failure to the A2A caller
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Analysis failed: {exc}"),
                final=True,
            )
