"""An end-to-end smoke test against a fresh in-memory app: `python -m helpdesk smoke`.

It seeds the demo data, then drives the API and the web UI the way a user would. Each
check prints PASS or FAIL, and the command exits non-zero if any check failed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from .app import create_app
from .config import Settings
from .seed import DEMO_PASSWORD, seed


def run_smoke(echo: Callable[[str], None] = print) -> bool:
    sent: list[httpx.Request] = []

    def receiver(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(204)

    app = create_app(
        Settings(database_url="sqlite://", secret_key="smoke-test"),  # noqa: S106
        http_client=httpx.Client(transport=httpx.MockTransport(receiver)),
    )
    with app.state.sessionmaker() as db:
        keys = seed(db)
    api = {"Authorization": f"Bearer {keys['acme']}"}
    results: list[bool] = []

    def check(name: str, ok: bool) -> None:
        results.append(ok)
        echo(f"{'PASS' if ok else 'FAIL'}  {name}")

    with TestClient(app) as c:
        check("GET /health is ok", c.get("/health").json().get("status") == "ok")
        check("API rejects a request without a key", c.get("/api/v1/tickets").status_code == 401)
        check(
            "API lists the seeded tickets", len(c.get("/api/v1/tickets", headers=api).json()) == 3
        )
        hook = c.post(
            "/api/v1/webhooks",
            json={"url": "https://smoke.test/hook", "events": ["ticket.created"]},
            headers=api,
        )
        check("API creates a webhook", hook.status_code == 201)
        created = c.post(
            "/api/v1/tickets",
            json={"subject": "Smoke test", "requester_email": "smoke@example.test"},
            headers=api,
        )
        check("API creates a ticket", created.status_code == 201)
        number = created.json().get("number")
        check("the new ticket fired a webhook", len(sent) == 1)
        globex = {"Authorization": f"Bearer {keys['globex']}"}
        other = c.get(f"/api/v1/tickets/{number}", headers=globex)
        check("another workspace can't see the ticket", other.status_code == 404)

        login_page = c.get("/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
        check("login page renders with a CSRF token", token is not None)
        if token:
            signed_in = c.post(
                "/login",
                data={
                    "csrf_token": token.group(1),
                    "workspace": "acme",
                    "email": "sam@acme.test",
                    "password": DEMO_PASSWORD,
                },
                follow_redirects=False,
            )
            check("demo agent can sign in", signed_in.status_code == 303)
            queue = c.get("/tickets")
            check("ticket queue shows the new ticket", "Smoke test" in queue.text)
            page = c.get(f"/tickets/{number}")
            check("ticket page renders", page.status_code == 200 and "Smoke test" in page.text)
            check("agents can't open settings", c.get("/settings").status_code == 403)

    passed = all(results)
    echo(f"\n{sum(results)}/{len(results)} smoke checks passed")
    return passed
