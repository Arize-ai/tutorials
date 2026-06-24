# Order-pricing demo

A small, typed Python service used by the Arize AX cookbook
[Observe and Optimize Coding Agent Workflows](https://arize.com/docs/ax/cookbooks/ai-engineering-workflows/observe-and-optimize-coding-agent-workflows).
You point a traced coding agent at this repo, give it the ticket below, and then observe,
evaluate, and improve the agent from its trace.

## Layout

- `store/pricing.py` — order total with discount-code support.
- `tests/test_pricing.py` — the test suite.
- `Makefile` — `make check` runs the full gate: lint (ruff), types (mypy), tests (pytest).
- `CLAUDE.md` — agent rules, intentionally empty to start.
- `.claude/settings.json` — turns Arize tracing on for sessions run in this repo, so it stays scoped here while the cookbook sets your global default off.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # isolate the project's tools
make install   # ruff, mypy, pytest
make check     # should pass on a clean checkout
```

Start your coding agent from this activated shell so its commands and the `make check` gate
resolve to the tools you just installed.

## The ticket

> Add a `FREESHIP` code to the order-pricing service: an order with a `FREESHIP` code should
> have its shipping fee set to 0, without changing the subtotal discount. Update
> `store/pricing.py`.

The ticket says nothing about verification on purpose. Most agents edit the code and report
done without running `make check`, which is the behavior the cookbook teaches you to catch and fix.
