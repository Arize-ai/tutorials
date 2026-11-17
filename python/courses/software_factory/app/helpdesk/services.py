"""Business logic shared by the JSON API and the web UI.

Every function takes the tenant (or a Principal, which carries it) and only ever reads
or writes that tenant's rows. Route handlers stay thin and call these.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import security, webhooks
from .db import utcnow
from .models import (
    ApiKey,
    Comment,
    Priority,
    Role,
    Scope,
    Status,
    Tenant,
    Ticket,
    User,
    Webhook,
    WebhookDelivery,
)
from .principal import Principal
from .schemas import CommentOut, TicketOut
from .webhooks import Event

ACTIVE_STATUSES = (Status.open, Status.pending)
MAX_PAGE = 100


class NotFoundError(Exception):
    pass


class InvalidError(Exception):
    pass


# ---------- tickets ----------


def ticket_payload(ticket: Ticket) -> dict:
    return TicketOut.model_validate(ticket).model_dump(mode="json")


def list_tickets(
    db: Session,
    tenant_id: int,
    *,
    statuses: tuple[Status, ...] | None = None,
    priority: Priority | None = None,
    assignee_id: int | None = None,
    limit: int = MAX_PAGE,
) -> list[Ticket]:
    """The tenant's tickets, most recently updated first."""
    stmt = select(Ticket).where(Ticket.tenant_id == tenant_id)
    if statuses:
        stmt = stmt.where(Ticket.status.in_(statuses))
    if priority:
        stmt = stmt.where(Ticket.priority == priority)
    if assignee_id is not None:
        stmt = stmt.where(Ticket.assignee_id == assignee_id)
    stmt = stmt.order_by(Ticket.updated_at.desc(), Ticket.number.desc()).limit(min(limit, MAX_PAGE))
    return list(db.scalars(stmt))


def count_by_status(db: Session, tenant_id: int) -> dict[Status, int]:
    rows = db.execute(
        select(Ticket.status, func.count())
        .where(Ticket.tenant_id == tenant_id)
        .group_by(Ticket.status)
    ).all()
    counts = dict.fromkeys(Status, 0)
    counts.update({status: n for status, n in rows})
    return counts


def get_ticket(db: Session, tenant_id: int, number: int) -> Ticket:
    ticket = db.scalar(select(Ticket).where(Ticket.tenant_id == tenant_id, Ticket.number == number))
    if ticket is None:
        raise NotFoundError(f"ticket {number} not found")
    return ticket


def _tenant_user(db: Session, tenant_id: int, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None or user.tenant_id != tenant_id:
        raise InvalidError(f"user {user_id} is not an agent in this workspace")
    return user


def create_ticket(
    db: Session,
    actor: Principal,
    *,
    subject: str,
    requester_email: str,
    description: str = "",
    priority: Priority = Priority.normal,
    assignee_id: int | None = None,
) -> Ticket:
    if assignee_id is not None:
        _tenant_user(db, actor.tenant_id, assignee_id)
    # SQLite serializes writers, so max + 1 can't hand two tickets the same number here.
    last = db.scalar(select(func.max(Ticket.number)).where(Ticket.tenant_id == actor.tenant_id))
    ticket = Ticket(
        tenant_id=actor.tenant_id,
        number=(last or 0) + 1,
        subject=subject,
        description=description,
        requester_email=requester_email,
        priority=priority,
        assignee_id=assignee_id,
    )
    db.add(ticket)
    db.flush()
    db.refresh(ticket)
    webhooks.enqueue(db, actor.tenant_id, Event.ticket_created, ticket_payload(ticket))
    db.commit()
    return ticket


UPDATABLE = ("subject", "description", "status", "priority", "assignee_id")


def update_ticket(db: Session, actor: Principal, ticket: Ticket, changes: dict) -> Ticket:
    """Apply changes (a dict of UPDATABLE fields) and return the ticket."""
    unknown = set(changes) - set(UPDATABLE)
    if unknown:
        raise InvalidError(f"can't update {', '.join(sorted(unknown))}")
    if changes.get("assignee_id") is not None:
        _tenant_user(db, actor.tenant_id, changes["assignee_id"])
    for field in ("subject", "status", "priority"):
        if field in changes and changes[field] is None:
            raise InvalidError(f"{field} can't be empty")
    changed = [f for f, v in changes.items() if getattr(ticket, f) != v]
    if not changed:
        return ticket
    for field in changed:
        setattr(ticket, field, changes[field])
    db.flush()
    db.refresh(ticket)
    data = ticket_payload(ticket) | {"changed": sorted(changed), "actor": actor.name}
    webhooks.enqueue(db, actor.tenant_id, Event.ticket_updated, data)
    db.commit()
    return ticket


def add_comment(
    db: Session, actor: Principal, ticket: Ticket, *, body: str, internal: bool = False
) -> Comment:
    if ticket.tenant_id != actor.tenant_id:
        raise NotFoundError(f"ticket {ticket.number} not found")
    comment = Comment(
        tenant_id=actor.tenant_id,
        ticket=ticket,
        author_id=actor.user.id if actor.user else None,
        author_name=actor.name,
        body=body,
        internal=internal,
    )
    db.add(comment)
    db.flush()
    data = {
        "ticket": ticket.number,
        "comment": CommentOut.model_validate(comment).model_dump(mode="json"),
    }
    webhooks.enqueue(db, actor.tenant_id, Event.comment_created, data)
    db.commit()
    return comment


# ---------- users ----------


def list_users(db: Session, tenant_id: int) -> list[User]:
    return list(db.scalars(select(User).where(User.tenant_id == tenant_id).order_by(User.name)))


def authenticate(db: Session, workspace: str, email: str, password: str) -> User | None:
    user = db.scalar(
        select(User)
        .join(Tenant)
        .where(Tenant.slug == workspace.strip().lower(), User.email == email.strip().lower())
    )
    if user is None:
        security.burn_password_check(password)
        return None
    return user if security.verify_password(password, user.password_hash) else None


def create_tenant(db: Session, slug: str, name: str) -> Tenant:
    tenant = Tenant(slug=slug.lower(), name=name)
    db.add(tenant)
    db.commit()
    return tenant


def create_user(
    db: Session, tenant: Tenant, *, email: str, name: str, password: str, role: Role = Role.agent
) -> User:
    user = User(
        tenant_id=tenant.id,
        email=email.lower(),
        name=name,
        role=role,
        password_hash=security.hash_password(password),
    )
    db.add(user)
    db.commit()
    return user


# ---------- API keys ----------


def list_api_keys(db: Session, tenant_id: int) -> list[ApiKey]:
    return list(
        db.scalars(select(ApiKey).where(ApiKey.tenant_id == tenant_id).order_by(ApiKey.id.desc()))
    )


def create_api_key(
    db: Session, tenant_id: int, *, name: str, scopes: list[Scope]
) -> tuple[ApiKey, str]:
    """Returns the key and its token. The token can't be recovered later."""
    token, prefix, key_hash = security.new_api_key()
    key = ApiKey(
        tenant_id=tenant_id,
        name=name,
        prefix=prefix,
        key_hash=key_hash,
        scopes=" ".join(sorted({Scope(s).value for s in scopes})),
    )
    db.add(key)
    db.commit()
    return key, token


def revoke_api_key(db: Session, tenant_id: int, key_id: int) -> ApiKey:
    key = db.get(ApiKey, key_id)
    if key is None or key.tenant_id != tenant_id:
        raise NotFoundError(f"API key {key_id} not found")
    if key.revoked_at is None:
        key.revoked_at = utcnow()
        db.commit()
    return key


# ---------- webhooks ----------


def list_webhooks(db: Session, tenant_id: int) -> list[Webhook]:
    return list(
        db.scalars(select(Webhook).where(Webhook.tenant_id == tenant_id).order_by(Webhook.id))
    )


def get_webhook(db: Session, tenant_id: int, webhook_id: int) -> Webhook:
    hook = db.get(Webhook, webhook_id)
    if hook is None or hook.tenant_id != tenant_id:
        raise NotFoundError(f"webhook {webhook_id} not found")
    return hook


def create_webhook(db: Session, tenant_id: int, *, url: str, events: list[Event]) -> Webhook:
    if not url.startswith(("https://", "http://")):
        raise InvalidError("webhook URL must be http or https")
    hook = Webhook(
        tenant_id=tenant_id,
        url=url,
        secret=security.new_webhook_secret(),
        events=" ".join(sorted({Event(e).value for e in events})),
    )
    db.add(hook)
    db.commit()
    return hook


def delete_webhook(db: Session, tenant_id: int, webhook_id: int) -> None:
    hook = get_webhook(db, tenant_id, webhook_id)
    db.query(WebhookDelivery).filter(WebhookDelivery.webhook_id == hook.id).delete()
    db.delete(hook)
    db.commit()


def list_deliveries(
    db: Session, tenant_id: int, webhook_id: int | None = None, limit: int = 50
) -> list[WebhookDelivery]:
    stmt = select(WebhookDelivery).where(WebhookDelivery.tenant_id == tenant_id)
    if webhook_id is not None:
        stmt = stmt.where(WebhookDelivery.webhook_id == webhook_id)
    return list(db.scalars(stmt.order_by(WebhookDelivery.id.desc()).limit(limit)))
