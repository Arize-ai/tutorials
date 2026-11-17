"""Shared fixtures: an app on an in-memory database, two workspaces, and a fake webhook receiver."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi.testclient import TestClient

from helpdesk import security, services
from helpdesk.app import create_app
from helpdesk.config import Settings
from helpdesk.models import Role, Scope, Tenant, User

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    monkeypatch.setattr(security, "PBKDF2_ITERATIONS", 1_000)


@dataclass
class Receiver:
    """Stands in for the internet: records every webhook POST and answers with `status`."""

    status: int = 200
    fail: bool = False
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("connection refused", request=request)
        self.requests.append(request)
        return httpx.Response(self.status)


@pytest.fixture
def receiver() -> Receiver:
    return Receiver()


@pytest.fixture
def app(receiver):
    settings = Settings(database_url="sqlite://", secret_key="test-secret")
    return create_app(settings, http_client=httpx.Client(transport=httpx.MockTransport(receiver)))


@pytest.fixture
def db(app):
    with app.state.sessionmaker() as session:
        yield session


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@dataclass
class Workspace:
    tenant: Tenant
    admin: User
    agent: User
    admin_key: str
    read_key: str
    write_key: str

    @staticmethod
    def auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @property
    def admin_h(self) -> dict[str, str]:
        return self.auth(self.admin_key)

    @property
    def read_h(self) -> dict[str, str]:
        return self.auth(self.read_key)

    @property
    def write_h(self) -> dict[str, str]:
        return self.auth(self.write_key)


def make_workspace(db, slug: str) -> Workspace:
    tenant = services.create_tenant(db, slug, slug.title())
    admin = services.create_user(
        db,
        tenant,
        email=f"admin@{slug}.test",
        name=f"{slug} admin",
        password=PASSWORD,
        role=Role.admin,
    )
    agent = services.create_user(
        db, tenant, email=f"agent@{slug}.test", name=f"{slug} agent", password=PASSWORD
    )
    keys = {}
    for name, scopes in {
        "admin": [Scope.admin],
        "read": [Scope.tickets_read],
        "write": [Scope.tickets_read, Scope.tickets_write],
    }.items():
        keys[name] = services.create_api_key(db, tenant.id, name=name, scopes=scopes)[1]
    return Workspace(tenant, admin, agent, keys["admin"], keys["read"], keys["write"])


@pytest.fixture
def acme(db) -> Workspace:
    return make_workspace(db, "acme")


@pytest.fixture
def globex(db) -> Workspace:
    return make_workspace(db, "globex")


def new_ticket(client, ws: Workspace, **over) -> dict:
    body = {"subject": "Printer on fire", "requester_email": "pat@customer.test"} | over
    r = client.post("/api/v1/tickets", json=body, headers=ws.write_h)
    assert r.status_code == 201, r.text
    return r.json()


def csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "page has no CSRF token"
    return match.group(1)


def login(client, ws: Workspace, user: User | None = None):
    user = user or ws.agent
    token = csrf(client.get("/login").text)
    return client.post(
        "/login",
        data={
            "csrf_token": token,
            "workspace": ws.tenant.slug,
            "email": user.email,
            "password": PASSWORD,
        },
        follow_redirects=False,
    )
