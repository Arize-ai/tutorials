"""Deploy both agents to Vertex AI Agent Engine as managed A2A services.

This is the production counterpart to a2a_servers.py: instead of two local uvicorn
processes, each agent becomes a managed Agent Engine service with an authenticated A2A
endpoint, and the orchestrator reaches them through Google-signed HTTP requests.

Vertex AI only. Requires GOOGLE_CLOUD_PROJECT, a staging bucket, Application Default
Credentials, and permission to create Agent Engine resources.

    python deploy_agent_engine.py                 # deploy both, print resource names
    python deploy_agent_engine.py --query "..."   # deploy, then run one query
    python deploy_agent_engine.py --delete <resource-name> [<resource-name> ...]

Deployment takes several minutes per agent and leaves billable resources running. Delete
them with --delete when you are finished.
"""

import argparse
import asyncio
import os

import httpx
import vertexai
from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import TransportProtocol
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from vertexai.preview.reasoning_engines import A2aAgent

import config
from trading_agents.bear_agent import BearAgentExecutor, create_bear_agent_card
from trading_agents.bull_agent import build_bull_executor, create_bull_agent_card

# Arize AX settings are forwarded to the deployed services so the remote agents trace to
# the same project. They are read from the environment, never hardcoded.
ARIZE_ENV_VARS = (
    "ARIZE_API_KEY",
    "ARIZE_SPACE_ID",
    "ARIZE_PROJECT_NAME",
    "ARIZE_COLLECTOR_ENDPOINT",
)

BEAR_REQUIREMENTS = [
    "a2a-sdk>=0.3.4,<0.4",
    "google-cloud-aiplatform[agent_engines,adk]",
    "fastmcp",
    "pydantic",
    "pydantic-ai",
    "numpy",
    "arize-otel",
    "openinference-instrumentation-pydantic-ai",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-api",
]

BULL_REQUIREMENTS = [
    "a2a-sdk>=0.3.4,<0.4",
    "google-cloud-aiplatform[agent_engines,adk]",
    "fastmcp",
    "numpy",
    "litellm",
    "arize-otel",
    "openinference-instrumentation-google-adk",
]


def arize_env() -> dict:
    """Collect the Arize AX settings to forward to the deployed agents."""
    return {name: os.environ[name] for name in ARIZE_ENV_VARS if os.environ.get(name)}


def staging_bucket() -> str:
    """Return the GCS staging bucket URI used to upload the agent packages."""
    explicit = os.environ.get("STAGING_BUCKET")
    if explicit:
        return explicit if explicit.startswith("gs://") else f"gs://{explicit}"
    return f"gs://{config.GOOGLE_CLOUD_PROJECT}-agent"


def deploy(client, name: str, card, executor_builder, requirements: list[str]):
    """Deploy one agent to Agent Engine and return the created resource."""
    # http_json is the transport Agent Engine serves; local runs use jsonrpc instead.
    card.preferred_transport = TransportProtocol.http_json

    print(f"Deploying {name} (this takes several minutes)...")
    created = client.agent_engines.create(
        agent=A2aAgent(agent_card=card, agent_executor_builder=executor_builder),
        config={
            "display_name": name,
            "description": card.description,
            "requirements": requirements,
            # mcp_tools ships alongside the agent; the deployed copy spawns it over stdio.
            "extra_packages": ["mcp_tools"],
            "env_vars": arize_env(),
            "staging_bucket": staging_bucket(),
        },
    )
    print(f"  {name} -> {created.api_resource.name}")
    return created


class GoogleAuth(httpx.Auth):
    """Sign every outgoing request with a Google Cloud access token."""

    def __init__(self) -> None:
        self.credentials, self.project = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.auth_request = AuthRequest()

    def auth_flow(self, request: httpx.Request):
        if not self.credentials.valid:
            self.credentials.refresh(self.auth_request)
        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


def remote_proxies(bear_resource: str, bull_resource: str):
    """Build RemoteA2aAgent proxies for the two deployed Agent Engine endpoints."""
    api_endpoint = f"https://{config.GOOGLE_CLOUD_LOCATION}-aiplatform.googleapis.com"
    authenticated_client = httpx.AsyncClient(timeout=120, auth=GoogleAuth())
    factory = A2AClientFactory(
        config=A2AClientConfig(
            httpx_client=authenticated_client,
            streaming=False,
            polling=False,
            supported_transports=[TransportProtocol.http_json],
        )
    )

    def proxy(name: str, description: str, resource: str) -> RemoteA2aAgent:
        endpoint = f"{api_endpoint}/v1beta1/{resource}/a2a"
        return RemoteA2aAgent(
            name=name,
            description=description,
            agent_card=f"{endpoint}/v1/card",
            httpx_client=authenticated_client,
            a2a_client_factory=factory,
        )

    return (
        proxy("bear_risk_analyst", "Analyzes risks and warning signals", bear_resource),
        proxy(
            "bull_market_analyst",
            "Identifies growth opportunities and bullish patterns",
            bull_resource,
        ),
    )


async def query_deployed(bear_resource: str, bull_resource: str, query: str):
    """Run one query against the deployed agents through the orchestrator."""
    remote_bear, remote_bull = remote_proxies(bear_resource, bull_resource)
    orchestrator = LlmAgent(
        name="trading_strategy_orchestrator",
        model=config.orchestrator_model(),
        instruction=config.ORCHESTRATOR_INSTRUCTION,
        tools=[AgentTool(agent=remote_bear), AgentTool(agent=remote_bull)],
    )
    runner = Runner(
        app_name=orchestrator.name,
        agent=orchestrator,
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(
        app_name=orchestrator.name, user_id="deploy_user", session_id="deploy_session"
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    async for event in runner.run_async(
        session_id=session.id, user_id="deploy_user", new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                return "".join(
                    p.text for p in event.content.parts if getattr(p, "text", None)
                )
            break
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run this query against the deployed agents")
    parser.add_argument(
        "--delete", nargs="+", metavar="RESOURCE", help="Delete deployed agents and exit"
    )
    args = parser.parse_args()

    if config.MODEL_PROVIDER != "vertex":
        raise SystemExit(
            "Agent Engine deployment requires MODEL_PROVIDER=vertex "
            f"(currently {config.MODEL_PROVIDER})."
        )
    config.init_vertex()
    client = vertexai.Client(
        project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_LOCATION
    )

    if args.delete:
        for resource in args.delete:
            print(f"Deleting {resource}")
            client.agent_engines.delete(resource, force=True)
        return

    bear = deploy(
        client,
        "Bear Risk Analyst",
        create_bear_agent_card(),
        BearAgentExecutor,
        BEAR_REQUIREMENTS,
    )
    bull = deploy(
        client,
        "Bull Market Analyst",
        create_bull_agent_card(),
        build_bull_executor,
        BULL_REQUIREMENTS,
    )

    bear_resource = bear.api_resource.name
    bull_resource = bull.api_resource.name
    print(f"\nDelete them when finished:\n  python deploy_agent_engine.py --delete "
          f"{bear_resource} {bull_resource}")

    if args.query:
        print(f"\nQuery: {args.query}")
        print(asyncio.run(query_deployed(bear_resource, bull_resource, args.query)))


if __name__ == "__main__":
    main()
