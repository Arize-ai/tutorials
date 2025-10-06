from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
import json
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx

# Make MCP import optional
try:
    from openinference.instrumentation.mcp import MCPInstrumentor
    _MCP_AVAILABLE = True
except ImportError:
    MCPInstrumentor = None  # type: ignore
    _MCP_AVAILABLE = False


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None
    user_input: Optional[str] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flags keep optional demos opt-in.
ENABLE_RAG = os.getenv("ENABLE_RAG", "1").lower() not in {"0", "false", "no"}
ENABLE_MCP = os.getenv("ENABLE_MCP", "0").lower() not in {"0", "false", "no"} and _MCP_AVAILABLE

if ENABLE_MCP:
    import anyio
    from mcp.server import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session


# RAG helper: materialize curated local-guide blurbs as LangChain documents.
def _load_local_documents(path: Path) -> List[Document]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for row in raw:
        description = row.get("description")
        city = row.get("city")
        if not description or not city:
            continue
        interests = row.get("interests", []) or []
        metadata = {
            "city": city,
            "interests": interests,
            "source": row.get("source"),
        }
        # Prefix city + interests in the content so embeddings capture location context.
        interest_text = ", ".join(interests) if interests else "general travel"
        content = f"City: {city}\nInterests: {interest_text}\nGuide: {description}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class LocalGuideRetriever:
    """Embeds curated blurbs and returns the best matches for a request."""
    def __init__(self, data_path: Path):
        self._docs = _load_local_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        return not self._docs

    def retrieve(self, destination: str, interests: Optional[str], *, k: int = 3) -> List[Dict[str, Any]]:
        if not ENABLE_RAG or self.is_empty:
            return []

        if not self._vectorstore:
            return self._keyword_fallback(destination, interests, k=k)

        query = destination
        if interests:
            query = f"{destination} with interests {interests}"
        try:
            # Using the LangChain retriever ensures query embeddings + searches are traced.
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(destination, interests, k=k)

        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score_val,
                }
            )

        if not results:
            return self._keyword_fallback(destination, interests, k=k)
        return results

    def _keyword_fallback(self, destination: str, interests: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        dest_lower = destination.lower()
        interest_terms = [part.strip().lower() for part in (interests or "").split(",") if part.strip()]

        def _score(doc: Document) -> int:
            score = 0
            city_match = doc.metadata.get("city", "").lower()
            if dest_lower and dest_lower.split(",")[0] in city_match:
                score += 2
            for term in interest_terms:
                if term and term in " ".join(doc.metadata.get("interests") or []).lower():
                    score += 1
                if term and term in doc.page_content.lower():
                    score += 1
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        results = []
        for score, doc in top_docs:
            city_value = doc.metadata.get("city", "").lower()
            if score <= 0 and dest_lower not in city_value:
                continue
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )
        return results


# Singleton retriever keeps embeddings warm across requests.
LOCAL_GUIDE_RETRIEVER = LocalGuideRetriever(Path(__file__).resolve().parent / "data" / "local_guides.json")


class MCPWeatherClient:
    """In-process MCP server + client pair for deterministic weather lookups."""

    def __init__(self) -> None:
        self._tool_name = "weather_mcp"
        
        self._weather_db = {
            "prague": {
                "temperature": "6-14°C",
                "conditions": "Crisp mornings, light rain",
                "advice": "Layer up and carry a compact umbrella",
            },
            "bangkok": {
                "temperature": "27-33°C",
                "conditions": "Humid afternoons, evening storms",
                "advice": "Light fabrics, hydrate, bring a poncho",
            },
            "dubai": {
                "temperature": "24-36°C",
                "conditions": "Dry heat with breezy nights",
                "advice": "High-SPF sunscreen and breathable layers",
            },
            "barcelona": {
                "temperature": "14-24°C",
                "conditions": "Sunny with a coastal breeze",
                "advice": "Light jacket for evenings, sunscreen by day",
            },
            "tokyo": {
                "temperature": "10-22°C",
                "conditions": "Cool mornings, clear afternoons",
                "advice": "Layered outfits and comfortable rainproof shoes",
            },
            "rome": {
                "temperature": "12-23°C",
                "conditions": "Mild with scattered showers",
                "advice": "Carry a light sweater and umbrella",
            },
            "lisbon": {
                "temperature": "13-21°C",
                "conditions": "Coastal breeze, patchy clouds",
                "advice": "Windbreaker plus comfy walking shoes",
            },
            "marrakech": {
                "temperature": "16-30°C",
                "conditions": "Warm days, cool desert nights",
                "advice": "Layer your outfits and pack sun protection",
            },
            "new york": {
                "temperature": "5-18°C",
                "conditions": "Variable with a chance of rain",
                "advice": "Light coat, closed shoes, compact umbrella",
            },
        }

        self._server = FastMCP(
            name="Weather MCP Demo",
            instructions="Deterministic weather tool for instrumentation demos.",
        )

        # Register MCP tool - let MCP instrumentation handle tracing automatically
        @self._server.tool(name=self._tool_name, description="Return a simple weather briefing for a destination.")
        def _weather_tool(destination: str) -> str:
            return self._build_summary(destination)

    def lookup(self, destination: str) -> str:
        return anyio.run(self._invoke_tool, destination)

    async def _invoke_tool(self, destination: str) -> str:
        assert create_connected_server_and_client_session is not None
        async with create_connected_server_and_client_session(
            self._server._mcp_server,
            raise_exceptions=True,
        ) as session:
            # Ensure tool metadata is populated for validation + traces.
            await session.list_tools()
            result = await session.call_tool(self._tool_name, {"destination": destination})
            if result.isError:
                raise RuntimeError("MCP weather tool returned an error result")

            parts = []
            for block in result.content:
                text_val = getattr(block, "text", None)
                if text_val:
                    parts.append(text_val)
            summary = "\n".join(parts).strip()
            if not summary:
                raise RuntimeError("MCP weather tool produced no content")
            return summary

    def _build_summary(self, destination: str) -> str:
        city_key = destination.split(",")[0].strip().lower()
        payload = self._weather_db.get(city_key)
        if payload:
            return (
                f"MCP Weather • {destination}: {payload['temperature']} with {payload['conditions']}. "
                f"Tip: {payload['advice']}."
            )
        return (
            f"MCP Weather • {destination}: Seasonal averages unavailable. "
            "Check a forecast a few days ahead and pack adaptable layers."
        )


MCP_WEATHER_CLIENT: Optional[MCPWeatherClient] = None
if ENABLE_MCP:
    MCP_WEATHER_CLIENT = MCPWeatherClient()


# Search API configuration and helpers
SEARCH_TIMEOUT = float(os.getenv("SEARCH_API_TIMEOUT", "8"))


def _compact(text: str, limit: int = 200) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    query = query.strip()
    if not query:
        return None

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass

    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass

    return None


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke(
        [
            SystemMessage(content="You are a concise travel assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools use live search with LLM fallback to stay concise
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    query = (
        f"{destination} travel essentials weather best time top attractions etiquette language currency safety"
    )
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} essentials", summary)

    instruction = (
        f"Summarize the climate, best visit time, standout sights, customs, language, currency, and safety tips for {destination}."
    )
    return _llm_fallback(instruction)


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    query = f"{destination} travel budget average daily costs {duration}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} budget {duration}", summary)

    instruction = (
        f"Outline lodging, meals, transport, activities, and extra costs for a {duration} trip to {destination}."
    )
    return _llm_fallback(instruction)


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    focus = interests or "local culture"
    query = f"{destination} authentic local experiences {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} {focus}", summary)

    instruction = f"Recommend authentic local experiences in {destination} that highlight {focus}."
    return _llm_fallback(instruction)


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    query = f"{destination} day {day} itinerary highlights"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Day {day} in {destination}", summary)

    instruction = f"Outline key activities for day {day} in {destination}, covering morning, afternoon, and evening."
    return _llm_fallback(instruction)


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    query = f"{destination} weather forecast travel season temperatures rainfall"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} weather", summary)

    instruction = (
        f"Give a weather brief for {destination} noting season, temperatures, rainfall, humidity, and packing guidance."
    )
    return _llm_fallback(instruction)


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance placeholder for tutorial purposes."""
    query = f"{destination} tourist visa requirements entry rules"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} visa", summary)

    instruction = (
        f"Provide a visa guidance summary for visiting {destination}, including advice to confirm with the relevant embassy."
    )
    return _llm_fallback(instruction)


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return rough placeholder prices for attractions."""
    items = attractions or ["popular attractions"]
    focus = ", ".join(items)
    query = f"{destination} attraction ticket prices {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} attraction prices", summary)

    instruction = (
        f"Share typical ticket prices and savings tips for attractions such as {focus} in {destination}."
    )
    return _llm_fallback(instruction)


@tool
def local_customs(destination: str) -> str:
    """Return simple etiquette reminders for the destination."""
    query = f"{destination} cultural etiquette travel customs"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} customs", summary)

    instruction = f"Summarize key etiquette and cultural customs travelers should know before visiting {destination}."
    return _llm_fallback(instruction)


@tool
def hidden_gems(destination: str) -> str:
    """Return a few off-the-beaten-path ideas."""
    query = f"{destination} hidden gems local secrets lesser known spots"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} hidden gems", summary)

    instruction = f"List lesser-known attractions or experiences that feel like hidden gems in {destination}."
    return _llm_fallback(instruction)


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return an approximate travel time placeholder."""
    query = f"travel time {from_location} to {to_location} by {mode}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(
            f"{from_location}→{to_location} {mode}",
            summary,
        )

    instruction = (
        f"Estimate the travel time from {from_location} to {to_location} by {mode}, providing a realistic range."
    )
    context = f"From: {from_location}\nTo: {to_location}\nMode: {mode}"
    return _llm_fallback(instruction, context=context)


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return a generic packing list summary."""
    acts = ", ".join(activities or ["sightseeing"])
    query = f"what to pack for {destination} {duration} {acts}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} packing", summary)

    instruction = (
        f"Suggest packing essentials for a {duration} trip to {destination} focused on {acts}."
    )
    context = f"Destination: {destination}\nDuration: {duration}\nActivities: {acts}"
    return _llm_fallback(instruction, context=context)


@tool
def mcp_weather(destination: str) -> str:
    """Get current weather conditions and packing advice for the destination via MCP."""
    if MCP_WEATHER_CLIENT:
        return MCP_WEATHER_CLIENT.lookup(destination)
    raise RuntimeError("MCP weather client failed to initialize.")


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    local_context: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    user_input = (req.get("user_input") or "").strip()
    if ENABLE_MCP:
        prompt_lines = [
            "You are a research assistant.",
            "First, call the mcp_weather tool to get weather for {destination}.",
            "Then use other tools as needed for additional information.",
        ]
    else:
        prompt_lines = [
            "You are a research assistant.",
            "Gather essential information about {destination}.",
            "Use at most one tool if needed.",
        ]
    if user_input:
        prompt_lines.append("User input: {user_input}")
    prompt_t = "\n".join(prompt_lines)
    vars_ = {"destination": destination, "user_input": user_input}
    tools = [essential_info, weather_brief, visa_brief]
    if ENABLE_MCP:
        tools.append(mcp_weather)
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools(tools)
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    budget = req.get("budget", "moderate")
    user_input = (req.get("user_input") or "").strip()
    prompt_lines = [
        "You are a budget analyst.",
        "Analyze costs for {destination} over {duration} with budget: {budget}.",
        "Use tools to get pricing information, then provide a detailed breakdown.",
    ]
    if user_input:
        prompt_lines.append("User input: {user_input}")
    prompt_t = "\n".join(prompt_lines)
    vars_ = {
        "destination": destination,
        "duration": duration,
        "budget": budget,
        "user_input": user_input,
    }

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [budget_basics, attraction_prices]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        follow_up = (
            f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."
        )
        if user_input:
            follow_up += f" Address this user input as well: {user_input}."
        messages.append(SystemMessage(content=follow_up))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    user_input = (req.get("user_input") or "").strip()
    interests_raw = (req.get("interests") or "").strip()
    interests = interests_raw or "local culture"
    # Pull semantic matches from curated dataset when the flag allows it.
    retrieval_focus = interests_raw or (user_input if user_input else None)
    retrieved = LOCAL_GUIDE_RETRIEVER.retrieve(destination, retrieval_focus)
    context_lines = []
    citation_lines = []
    for idx, item in enumerate(retrieved, start=1):
        meta = item.get("metadata", {})
        label = meta.get("city") or destination
        context_lines.append(f"[{idx}] {label}: {item.get('content')}")
        if meta.get("source"):
            citation_lines.append(f"[{idx}] {meta['source']}")
    if context_lines:
        context_text = "\n".join(context_lines)
    elif ENABLE_RAG:
        context_text = "No curated context available."
    else:
        context_text = "Context unavailable because ENABLE_RAG is disabled."

    prompt_t = (
        "You are a local guide.\n"
        "Use the retrieved travel notes to suggest authentic experiences in {destination}.\n"
    )
    if user_input:
        prompt_t += "User input: {user_input}\n"
    prompt_t += (
        "Focus interests: {interests}.\n"
        "Context:\n{context}\n"
        "Cite the numbered items when you rely on them."
    )
    vars_ = {
        "destination": destination,
        "interests": interests,
        "context": context_text,
        "user_input": user_input,
    }
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([local_flavor, local_customs, hidden_gems])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if retrieved:
        # Surface retrieval metadata in the trace for observability dashboards.
        calls.append(
            {
                "agent": "local",
                "tool": "local_guides_retriever",
                "args": {
                    "destination": destination,
                    "interests": interests,
                    "results": [
                        {
                            "city": item.get("metadata", {}).get("city"),
                            "source": item.get("metadata", {}).get("source"),
                            "score": round(float(item.get("score", 0.0)), 4),
                        }
                        for item in retrieved
                    ],
                },
            }
        )
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([local_flavor, local_customs, hidden_gems])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    if citation_lines:
        out = f"{out}\n\nSources:\n" + "\n".join(citation_lines)

    return {
        "messages": [SystemMessage(content=out)],
        "local": out,
        "local_context": context_text,
        "tool_calls": calls,
    }


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    user_input = (req.get("user_input") or "").strip()
    prompt_parts = [
        "Create a {duration} itinerary for {destination} ({travel_style}).",
        "",
        "Inputs:",
        "Research: {research}",
        "Budget: {budget}",
        "Local: {local}",
    ]
    if user_input:
        prompt_parts.append("User input: {user_input}")
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
        "user_input": user_input,
    }
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        with using_attributes(tags=["itinerary", "final_agent"]):
            if _TRACING:
                current_span = trace.get_current_span()
                if current_span:
                    current_span.set_attribute("metadata.itinerary", "true")
                    current_span.set_attribute("metadata.agent_type", "itinerary")
                    current_span.set_attribute("metadata.agent_node", "itinerary_agent")
                    if user_input:
                        current_span.set_attribute("metadata.user_input", user_input)
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_agent", research_agent)
    g.add_node("budget_agent", budget_agent)
    g.add_node("local_agent", local_agent)
    g.add_node("itinerary_agent", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_agent")
    g.add_edge(START, "budget_agent")
    g.add_edge(START, "local_agent")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_agent", "itinerary_agent")
    g.add_edge("budget_agent", "itinerary_agent")
    g.add_edge("local_agent", "itinerary_agent")
    
    g.add_edge("itinerary_agent", END)

    # Compile without checkpointer to avoid state persistence issues
    compiled = g.compile()
    compiled.name = "TripAgentGraph"
    return compiled


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
            if ENABLE_MCP:
                MCPInstrumentor().instrument(tracer_provider=tp)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):

    graph = build_graph()
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    # No config needed without checkpointer
    out = graph.invoke(state)
    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
