# AI Trip Planner — Agent Mastery Course

This repo has two things:

1. **Agent Mastery Course labs** (start here) — Jupyter notebooks that build an AI travel agent step by step using the [Agno](https://github.com/agno-agi/agno) framework, with tracing on [Arize](https://app.arize.com).
2. **Trip Planner app** (optional) — A FastAPI + LangGraph backend with a minimal UI, RAG, and MCP demos.

---

## Quickstart

### 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

### 2. Get API keys (free)

You need 3 free accounts — the labs will prompt you for these keys:

| Service | Sign up | What for |
|---------|---------|----------|
| **Arize** | https://app.arize.com | Observe agent traces |
| **OpenAI** | https://platform.openai.com/api-keys | LLM calls |
| **Tavily** | https://auth.tavily.com | Web search tool |

### 3. Set up

```bash
git clone <repo-url>
cd agent-mastery-course

# Copy and fill in your API keys
cp .env.example.labs .env
```

Open `.env` with a text editor and add your keys.

### 4. Start

```bash
docker compose up --build
```

Docker builds the environment with all dependencies and starts Jupyter Lab.
The terminal will show logs — keep it running and open a new browser tab.

### 5. Open the lab

Open http://localhost:8888 in your browser. You'll see the lab notebooks:

```
labs/
├── lab1and2_base_agent.ipynb   ← start here
├── lab3_agent_architectures.ipynb
├── lab4_tools.ipynb
├── lab5_RAG.ipynb
└── lab6_evals.ipynb
```

Double-click **lab1and2_base_agent.ipynb** and run the cells one by one.

### 6. What you'll see

- The agent plans a trip to Tokyo
- Each tool call and LLM request is traced to Arize
- Open https://app.arize.com to explore the traces

---

## Labs overview

| Lab | Topic | What you build |
|-----|-------|----------------|
| 1 & 2 | Base agent + tracing | A trip planner agent with tools |
| 3 | Agent architectures | Orchestrator-worker & parallel patterns |
| 4 | Tools | Enhance tools with real APIs |
| 5 | RAG | Add retrieval-augmented generation |
| 6 | Evals | Evaluate and log agent performance |

---

## Project structure

```
agent-mastery-course/
├── labs/                    ← Jupyter notebooks (start here)
│   ├── lab1and2_base_agent.ipynb
│   ├── lab3_agent_architectures.ipynb
│   ├── lab4_tools.ipynb
│   ├── lab5_RAG.ipynb
│   └── lab6_evals.ipynb
├── backend/                 ← FastAPI + LangGraph app (optional)
├── frontend/                ← Minimal UI (optional)
├── Dockerfile.labs          ← Container for the labs
├── docker-compose.yml       ← Runs Jupyter Lab
├── requirements-labs.txt    ← Python deps for all labs
├── .env.example.labs        ← API key template
└── README.md
```

---

## Optional: Run the Trip Planner app

The `backend/` directory contains a FastAPI server with a LangGraph agent, optional RAG, MCP weather demo, and a minimal frontend.

### Backend setup

```bash
cd backend
cp .env.example .env
# Set OPENAI_API_KEY or OPENROUTER_API_KEY in backend/.env
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs to try the API.

### Backend with Docker

```bash
docker compose -f docker-compose.backend.yml up --build
```

(Not yet created — the backend docker-compose is coming soon.)

### Backend features

- POST `/plan-trip` — generates a travel itinerary
- GET `/health` — health check
- GET `/` — minimal frontend
- Optional Arize tracing via `ARIZE_SPACE_ID` / `ARIZE_API_KEY`
- Optional RAG via `ENABLE_RAG=1` (curated local guide content)
- Optional MCP weather demo via `ENABLE_MCP=1`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No module named ..." in notebook | `Kernel → Restart Kernel` after the `!pip install` cell |
| API returns 401 | Verify the key is correct in your `.env` |
| No traces in Arize | Check `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set |
| Port 8888 in use | Change the port in `docker-compose.yml`: `"8889:8888"` |
| Docker build slow | Only the first build downloads packages. Rebuilds are instant. |
