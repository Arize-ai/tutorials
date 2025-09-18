# Agent Mastery Course Labs

Hands-on labs that accompany the Agent Mastery curriculum. Work through the labs in order to launch an agent locally, instrument it for observability, and explore advanced topics.

## Curriculum Modules

- Introduction to Agents
- Agent Engineering
- Agent Frameworks and Architectures
- TypeScript Deep Dive
- Python Deep Dive
- Tools and MCP
- Planning and Reasoning
- APIs and Integrations
- Model Context Protocol (MCP)
- RAG and Agentic RAG
- Observability and Evaluation
- Post-Deployment Monitoring

## Labs Overview

### Lab 1 – Agent Setup
- Read `labs/lab1.md` for the quick-start checklist
- Install dependencies, load environment variables, and boot the backend and frontend
- Verify the starter itinerary agent responds to a sample trip request locally

### Lab 2 – Observability Setup
- Follow `labs/lab2.md` to connect the agent to Arize observability
- Enable Arize tracing via MCP or CLI helpers
- Capture at least one trace in Arize after exercising the local agent

### Lab 3 – Agent Architectures (Optional Preview)
- Explore `lab3_agent_architectures.ipynb` for a guided tour of LangGraph patterns

## Getting Started

1. Clone the repository and create a virtual environment (Python 3.10+ recommended)
2. Copy `.env.example` (if available) to `.env` and populate API keys
3. Complete Lab 1 before attempting later labs
4. Use the labs in sequence; each builds on the previous setup

## Prerequisites

- Git, Node.js 18+, and Python 3.10+
- Access to an OpenAI or OpenRouter API key
- Optional: Arize account for observability and evaluation

## Support

If you run into issues, open a discussion in the course channel or file a GitHub issue with reproduction steps.
