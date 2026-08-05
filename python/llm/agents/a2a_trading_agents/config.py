"""Shared configuration, read from the environment.

The three agents run on Vertex AI: Gemini 2.5 Flash for the Bear agent and the
orchestrator, and Llama 3.3 70B from Vertex AI Model-as-a-Service for the Bull agent.

Vertex AI has no API key. It authenticates with Application Default Credentials against
a Google Cloud project that has the Vertex AI API enabled, so set GOOGLE_CLOUD_PROJECT
and run `gcloud auth application-default login` before any of the scripts here.

Model ids are read from the environment so you can point an agent at a different Vertex
model without touching the agent code.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env before reading anything, so the values below reflect it. Real environment
# variables win over the file.
load_dotenv(PROJECT_ROOT / ".env")

# --- Vertex AI ---------------------------------------------------------------------
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

BEAR_MODEL = os.environ.get("BEAR_MODEL", "gemini-2.5-flash")
BULL_MODEL = os.environ.get("BULL_MODEL", "vertex_ai/meta/llama-3.3-70b-instruct-maas")
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gemini-2.5-flash")

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


def bear_model():
    """Return the Pydantic AI model for the Bear agent."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleModel(BEAR_MODEL, provider=GoogleProvider(vertexai=True))


def bull_model():
    """Return the ADK model for the Bull agent.

    ADK reaches non-Gemini models through LiteLlm, which is how an ADK agent runs on
    Llama hosted by Vertex AI.
    """
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(BULL_MODEL)


def orchestrator_model() -> str:
    """Return the ADK model for the orchestrator.

    ADK takes a bare model id string for Gemini models.
    """
    return ORCHESTRATOR_MODEL


def init_vertex() -> None:
    """Initialize Vertex AI and point LiteLLM at the same project."""
    import vertexai
    from google.adk.models.lite_llm import litellm

    if not GOOGLE_CLOUD_PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is required. Set it in .env or the environment, and "
            "authenticate with `gcloud auth application-default login`."
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
