# Trailhead Outfitters support agent

A small customer support agent for a fictional outdoor gear store. It is the sample app for the Arize AX cookbook [Find and fix issues in your AI app with a coding agent](https://arize.com/docs/ax/cookbooks/improve/find-and-fix-issues-with-a-coding-agent).

The agent answers questions about orders, product stock, and store policies using three tools. It runs on OpenAI or Anthropic, and ships without any tracing so you can add it with your coding agent.

## Run it

You need [uv](https://docs.astral.sh/uv/) and an OpenAI or Anthropic API key.

```bash
cd python/cookbooks/find_and_fix_with_coding_agent
cp .env.example .env    # then add your API key
uv sync
uv run support.py "Do you have the Summit 2 Tent in stock?"
```

Run `uv run support.py` with no question to chat interactively.

## Choose a provider

The agent uses whichever key you set:

| Key set | Provider | Model |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | `gpt-4.1-mini` |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-haiku-4-5` |

If both keys are set, OpenAI is used.

## Generate traffic

`seed.py` sends a batch of realistic customer questions through the agent, so you have data to analyze once tracing is in place:

```bash
uv run seed.py              # 100 questions
uv run seed.py --count 20   # fewer questions
```

## Files

| File | What it does |
|---|---|
| `agent.py` | The tool-calling loop for OpenAI and Anthropic |
| `tools.py` | The `lookup_order`, `check_stock`, and `get_policy` tools |
| `store.py` | In-memory customers, orders, products, and policies |
| `support.py` | Ask the agent a question from the command line |
| `seed.py` | Send a batch of questions through the agent |

_Last updated: 2026-09-30_
