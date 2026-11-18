# Helpdesk

A small multi-tenant helpdesk. Support teams sign in to a web UI to work a ticket queue,
and other systems use the JSON API and signed webhooks.

- **Workspaces.** Each customer company is a workspace (a tenant) with its own team,
  tickets, API keys and webhooks. Nothing crosses workspaces.
- **Tickets.** Subject, description, requester, status (open, pending, solved, closed),
  priority and assignee. Tickets are numbered per workspace: #1, #2 and so on.
- **Conversation.** Replies and agent-only internal notes on each ticket.
- **API keys.** Scoped bearer keys (`tickets:read`, `tickets:write`, `admin`), stored
  hashed and shown once.
- **Webhooks.** `ticket.created`, `ticket.updated` and `comment.created`, signed with
  HMAC-SHA256, with every delivery attempt recorded.
- **Web UI.** Sign in, a ticket queue with views and filters, a ticket page with the
  conversation, and admin settings for the team, API keys and webhooks.

It's built with FastAPI, SQLAlchemy and SQLite, and the UI is server-rendered HTML with no
JavaScript build. It's also the app that the
[Arize AX software factory course](https://github.com/Arize-ai/tutorials/tree/main/python/courses/software_factory)
works on: the course creates your own copy of it on GitHub, you file issues there, and
the factory takes each one from spec to release.

## Run it

You need Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m helpdesk seed     # demo workspaces, users, tickets and API keys
uv run python -m helpdesk serve    # http://127.0.0.1:8000
```

Sign in with workspace `acme`, email `ada@acme.test` (an admin) or `sam@acme.test` (an
agent), and password `helpdesk-demo`. A second workspace, `globex`, has
`hank@globex.test` and `frank@globex.test`. `seed` prints an admin API key for each
workspace.

To add your own user, creating the workspace if it doesn't exist:

```bash
uv run python -m helpdesk create-user acme you@example.com "Your Name" --role admin
```

Settings come from environment variables; see [`.env.example`](.env.example). To use a
file, run `uv run --env-file .env python -m helpdesk serve`. Outside dev
(`HELPDESK_ENV` set to anything else), the app won't start without `HELPDESK_SECRET_KEY`.

## The API

Interactive docs are at `/docs` while the app is running. Every call under `/api/v1`
needs an API key:

```bash
KEY=hd_...   # from `seed`, or Settings → API keys

curl -s localhost:8000/api/v1/tickets -H "Authorization: Bearer $KEY"

curl -s localhost:8000/api/v1/tickets -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"subject": "Can'\''t reset my password", "requester_email": "pat@example.com", "priority": "high"}'

curl -s -X PATCH localhost:8000/api/v1/tickets/4 -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{"status": "pending"}'
```

| Method and path | Scope | What it does |
|---|---|---|
| `GET /api/v1/me` | `tickets:read` | The workspace and scopes of this key |
| `GET /api/v1/users` | `tickets:read` | The workspace's team, for assigning tickets |
| `GET /api/v1/tickets` | `tickets:read` | Up to 100 tickets, most recently updated first. Filter with `status` (repeatable), `priority`, `assignee_id` |
| `POST /api/v1/tickets` | `tickets:write` | Create a ticket |
| `GET /api/v1/tickets/{number}` | `tickets:read` | One ticket |
| `PATCH /api/v1/tickets/{number}` | `tickets:write` | Change the fields you send. `"assignee_id": null` unassigns |
| `GET /api/v1/tickets/{number}/comments` | `tickets:read` | The conversation, oldest first |
| `POST /api/v1/tickets/{number}/comments` | `tickets:write` | Reply, or add an internal note with `"internal": true` |
| `GET, POST /api/v1/api-keys`, `DELETE /api/v1/api-keys/{id}` | `admin` | List, create and revoke API keys |
| `GET, POST /api/v1/webhooks`, `DELETE /api/v1/webhooks/{id}` | `admin` | List, create and delete webhooks |
| `GET /api/v1/webhooks/{id}/deliveries` | `admin` | Recent delivery attempts |
| `GET /health` | none | Liveness check |

Errors are JSON, `{"detail": ...}`: 401 for a missing or bad key, 403 for a missing
scope, 404 for anything outside your workspace, and 422 for invalid input.

## Webhooks

Each delivery is a `POST` with a JSON body:

```json
{"event": "ticket.updated", "created_at": "2026-11-17T10:00:00+00:00",
 "data": {"number": 4, "status": "pending", "changed": ["status"], "...": "..."}}
```

Its headers include `X-Helpdesk-Event`, `X-Helpdesk-Delivery` and
`X-Helpdesk-Signature: t=<unix time>,v1=<signature>`, where the signature is the hex
HMAC-SHA256 of `<t>.<raw body>` keyed with the webhook's signing secret. To verify a
delivery:

```python
import hashlib, hmac


def verify(secret: str, header: str, body: bytes) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    expected = hmac.new(secret.encode(), f"{parts['t']}.".encode() + body, hashlib.sha256)
    return hmac.compare_digest(parts["v1"], expected.hexdigest())
```

Deliveries are sent once, right after the request that caused them. There are no
retries yet.

## Develop

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run python -m helpdesk smoke   # end-to-end check of the API and UI on a fresh app
```

CI runs all three on every pull request. The database schema is created at startup with
`create_all`, so there are no migrations yet: if you change a model, delete your local
`helpdesk.db` and run `seed` again.

```
helpdesk/
  app.py        application factory, middleware and error handling
  api.py        JSON API routes (/api/v1)
  web.py        web UI routes
  services.py   business logic shared by the API and the UI
  models.py     SQLAlchemy models
  schemas.py    API request and response bodies
  deps.py       sessions, API-key auth, sign-in and CSRF
  security.py   password hashing, API keys and webhook signatures
  webhooks.py   queuing and delivering webhooks
  seed.py       demo data
  templates/    Jinja templates
  static/       CSS
tests/          pytest suite
```
