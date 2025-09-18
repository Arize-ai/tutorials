# Lab 1 – Agent Setup

Goal: bring the AI Trip Planner agent online on your local machine.

## 1. Clone and prepare the repository
- `git clone https://github.com/Arize-ai/arize.git`
- `cd tutorials/python/llm/agents/agent-mastery-course` (the lab workspace)


## 2. Configure environment variables
- Copy `backend/env_example.txt` to `backend/.env`
- Add one LLM key: `OPENAI_API_KEY=`
- add `ARIZE_SPACE_ID` and `ARIZE_API_KEY` if you plan to instrument tracing later

## 3. Install dependencies
- `cd backend`
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)
- Return to the repo root when finished

## 4. Launch the agent locally
- From the repo root run `./start.sh`
  - Backend: FastAPI on `http://localhost:8000`
  - Static frontend: `http://localhost:8000/`
- Alternative: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload`


Checkpoint: capture the command output showing the server started successfully. You are ready for Lab 2.
