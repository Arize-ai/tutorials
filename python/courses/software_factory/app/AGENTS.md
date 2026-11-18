# AGENTS.md

Guidance for coding agents working on this repo. Read README.md for what the app does.

## Commands

Run all three before you say a change is done. CI runs the same three.

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

`uv run python -m helpdesk smoke` drives the API and UI end to end on a fresh in-memory
app. Releases are gated on it, so keep it passing when you change a flow it checks.

## Rules that keep the app safe

- **Workspace isolation is the most important property of this app.** Every query that
  reads or writes customer data filters on `tenant_id`, and tenant-owned rows are looked
  up with both their ID and the caller's tenant. An ID from the request is never enough
  on its own. Anything outside the caller's workspace returns 404, not 403, so its
  existence isn't leaked. Add a test in `tests/test_tenancy.py` for any new endpoint
  or query.
- **Business logic goes in `helpdesk/services.py`.** The API (`api.py`) and the web UI
  (`web.py`) are thin layers over it, so a feature usually needs a service function, an
  API route, a UI change and tests for each.
- **API routes declare their scope** with `CanRead`, `CanWrite` or `IsAdmin` from
  `deps.py`. Web routes use `WebUser` or `WebAdmin`.
- **Every web form posts a CSRF token** (`<input type="hidden" name="csrf_token"
  value="{{ csrf_token }}">`). The router checks it on every POST.
- **Never log or return secrets.** API keys and webhook secrets are shown once, at
  creation. Passwords and keys are stored hashed.
- **Templates autoescape.** Don't mark user content `|safe`.
- **Webhook payloads are part of the public API.** Add fields freely, but don't rename
  or remove them.

## Conventions

- Python 3.12, type hints, `from __future__ import annotations`.
- Datetimes are timezone-aware UTC: use `helpdesk.db.utcnow()`, never `datetime.now()`.
- Tests use the fixtures in `tests/conftest.py`: `client`, `acme` and `globex` (two
  workspaces, each with an admin, an agent and read, write and admin API keys), and
  `receiver` (records outgoing webhook requests). `new_ticket()` and `login()` are
  helpers.
- There are no migrations yet. The schema is created with `create_all`, which doesn't
  alter existing tables, so say so in the PR when you change a model.
- US English in code, comments and UI text.
