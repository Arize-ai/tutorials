"""Demo data: two workspaces, a few agents and some tickets, so there's something to look at."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import services
from .models import Priority, Role, Scope, Status, Tenant
from .principal import Principal

DEMO_PASSWORD = "helpdesk-demo"  # noqa: S105 - demo data only

WORKSPACES = {
    "acme": {
        "name": "Acme Rockets",
        "users": [
            ("ada@acme.test", "Ada Admin", Role.admin),
            ("sam@acme.test", "Sam Support", Role.agent),
        ],
        "tickets": [
            (
                "Rocket skates won't stop",
                "wile@desert.test",
                Priority.urgent,
                Status.open,
                "Both skates accelerate past the canyon edge. The brake button does nothing.",
            ),
            (
                "Invoice shows the wrong billing address",
                "billing@coyote.test",
                Priority.normal,
                Status.pending,
                "Our March invoice still has the old PO box.",
            ),
            (
                "How do I add a second user?",
                "ops@roadrunner.test",
                Priority.low,
                Status.solved,
                "We'd like our night shift to see tickets too.",
            ),
        ],
    },
    "globex": {
        "name": "Globex Corporation",
        "users": [
            ("hank@globex.test", "Hank Admin", Role.admin),
            ("frank@globex.test", "Frank Agent", Role.agent),
        ],
        "tickets": [
            (
                "SSO login loops back to the sign-in page",
                "it@cypress.test",
                Priority.high,
                Status.open,
                "Since Tuesday, every SSO login returns to the sign-in page.",
            ),
            (
                "Export all tickets to CSV",
                "pm@cypress.test",
                Priority.normal,
                Status.open,
                "We need a monthly export for our QBR deck.",
            ),
        ],
    },
}


def seed(db: Session) -> dict[str, str]:
    """Create the demo workspaces if they're missing. Returns a new admin API key per workspace."""
    keys: dict[str, str] = {}
    for slug, spec in WORKSPACES.items():
        if db.scalar(select(Tenant).where(Tenant.slug == slug)):
            continue
        tenant = services.create_tenant(db, slug, spec["name"])
        users = [
            services.create_user(db, tenant, email=e, name=n, password=DEMO_PASSWORD, role=r)
            for e, n, r in spec["users"]
        ]
        admin, agent = users[0], users[1]
        for subject, requester, priority, status, description in spec["tickets"]:
            ticket = services.create_ticket(
                db,
                Principal.for_user(admin),
                subject=subject,
                requester_email=requester,
                description=description,
                priority=priority,
                assignee_id=agent.id if priority in (Priority.high, Priority.urgent) else None,
            )
            services.add_comment(
                db,
                Principal.for_user(agent),
                ticket,
                body="Thanks for reaching out, we're looking into it.",
            )
            if status != Status.open:
                services.update_ticket(db, Principal.for_user(agent), ticket, {"status": status})
        _, token = services.create_api_key(
            db, tenant.id, name="Demo admin key", scopes=[Scope.admin]
        )
        keys[slug] = token
    return keys
