"""Shared configuration, read from the environment.

MODEL_PROVIDER selects where the three agents get their models:

  vertex  (default) Gemini 2.5 Flash for the Bear agent and orchestrator, Llama 3.3 70B
          from Vertex AI Model-as-a-Service for the Bull agent. Requires a Google Cloud
          project with the Vertex AI API enabled and Application Default Credentials.
  openai  OpenAI models for all three. Requires only OPENAI_API_KEY, which makes the
          local A2A path runnable without any Google Cloud setup.

The A2A protocol, MCP tools, and Arize AX tracing are identical either way. Only the
model bindings change, which is the point of routing every agent through a provider
layer rather than hardcoding a model string.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env before reading anything, so the values below reflect it. Real environment
# variables win over the file.
load_dotenv(PROJECT_ROOT / ".env")

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex").strip().lower()

# --- Vertex AI ---------------------------------------------------------------------
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- A2A server ports ---------------------------------------------------------------
BEAR_PORT = int(os.environ.get("BEAR_PORT", "8001"))
BULL_PORT = int(os.environ.get("BULL_PORT", "8002"))

BEAR_SYSTEM_PROMPT = (
    "You are a cautious risk analyst focused on identifying potential downside catalysts, "
    "warning signals, and protective strategies. You prioritize capital preservation. "
    "Use the available MCP tools to analyze market risks comprehensively."
)

BULL_SYSTEM_PROMPT = (
    "You are an optimistic market analyst focused on identifying growth opportunities, "
    "bullish patterns, and upside catalysts. You emphasize potential gains and momentum. "
    "Use the available tools to analyze market opportunities comprehensively."
)

ORCHESTRATOR_INSTRUCTION = (
    "You coordinate two specialist analysts to produce a balanced view of a stock. "
    "Call the bear analyst for downside risk and the bull analyst for upside "
    "opportunity, then summarize both sides and state which case is stronger."
)


def _env_model(name: str, vertex_default: str, openai_default: str) -> str:
    """Resolve a model id, letting an explicit env var override either provider default."""
    return os.environ.get(
        name, openai_default if MODEL_PROVIDER == "openai" else vertex_default
    )


def bear_model():
    """Return a Pydantic AI model for the Bear agent."""
    model_id = _env_model("BEAR_MODEL", "gemini-2.5-flash", "gpt-4.1-mini")
    if MODEL_PROVIDER == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_id)

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleModel(model_id, provider=GoogleProvider(vertexai=True))


def bull_model():
    """Return an ADK model for the Bull agent.

    ADK reaches non-Gemini models through LiteLlm, which is how the same agent code runs
    against Llama on Vertex AI or against OpenAI.
    """
    from google.adk.models.lite_llm import LiteLlm

    model_id = _env_model(
        "BULL_MODEL", "vertex_ai/meta/llama-3.3-70b-instruct-maas", "openai/gpt-4.1-mini"
    )
    return LiteLlm(model_id)


def orchestrator_model():
    """Return an ADK model for the orchestrator.

    ADK takes a bare string for Gemini models and a LiteLlm instance for everything else.
    """
    model_id = _env_model("ORCHESTRATOR_MODEL", "gemini-2.5-flash", "openai/gpt-4.1-mini")
    if MODEL_PROVIDER == "openai":
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model_id)
    return model_id


def init_vertex() -> None:
    """Initialize Vertex AI and point LiteLLM at the same project.

    A no-op when MODEL_PROVIDER is openai, so the local path needs no Google Cloud setup.
    """
    if MODEL_PROVIDER != "vertex":
        return

    import vertexai
    from google.adk.models.lite_llm import litellm

    if not GOOGLE_CLOUD_PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is required when MODEL_PROVIDER=vertex. "
            "Set it, or set MODEL_PROVIDER=openai to run without Google Cloud."
        )

    os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT
    os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

    litellm.vertex_project = GOOGLE_CLOUD_PROJECT
    litellm.vertex_location = GOOGLE_CLOUD_LOCATION

    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)


def mcp_server_command(module: str) -> tuple[str, list[str]]:
    """Command and args that launch an MCP server over stdio.

    Uses sys.executable rather than "python" so the server runs in the same interpreter
    as the agent, which matters inside a virtualenv where "python" may not exist at all.
    """
    return sys.executable, ["-m", module]
