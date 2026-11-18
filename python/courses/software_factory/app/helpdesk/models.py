"""Database models.

Every row that belongs to a customer carries tenant_id, and every query that reads one
must filter on it. Tickets are addressed by their per-tenant number, never by their
global primary key, so an ID from one tenant means nothing in another.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, UTCDateTime, utcnow


def _enum(cls: type[enum.Enum]) -> Enum:
    return Enum(cls, native_enum=False, length=16, values_callable=lambda e: [m.value for m in e])


class Role(enum.StrEnum):
    admin = "admin"
    agent = "agent"


class Status(enum.StrEnum):
    open = "open"
    pending = "pending"  # waiting on the requester
    solved = "solved"
    closed = "closed"


class Priority(enum.StrEnum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class Scope(enum.StrEnum):
    tickets_read = "tickets:read"
    tickets_write = "tickets:write"
    admin = "admin"  # API keys, webhooks and users; implies every other scope


ROLE_SCOPES: dict[Role, frozenset[Scope]] = {
    Role.admin: frozenset(Scope),
    Role.agent: frozenset({Scope.tickets_read, Scope.tickets_write}),
}


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(254))
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[Role] = mapped_column(_enum(Role), default=Role.agent)
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    tenant: Mapped[Tenant] = relationship()


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    # The first characters of the key, kept in clear so a key can be found and recognized.
    prefix: Mapped[str] = mapped_column(String(16), unique=True)
    key_hash: Mapped[str] = mapped_column(String(64))
    scopes: Mapped[str] = mapped_column(String(200))  # space-separated Scope values
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    tenant: Mapped[Tenant] = relationship()

    @property
    def scope_set(self) -> frozenset[Scope]:
        return frozenset(Scope(s) for s in self.scopes.split())

    @property
    def scope_list(self) -> list[str]:
        return sorted(self.scopes.split())


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (UniqueConstraint("tenant_id", "number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    subject: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[Status] = mapped_column(_enum(Status), default=Status.open)
    priority: Mapped[Priority] = mapped_column(_enum(Priority), default=Priority.normal)
    requester_email: Mapped[str] = mapped_column(String(254))
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    assignee: Mapped[User | None] = relationship()
    comments: Mapped[list[Comment]] = relationship(
        back_populates="ticket", order_by="Comment.id", cascade="all, delete-orphan"
    )


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    # Kept as text so comments made with an API key still say who made them.
    author_name: Mapped[str] = mapped_column(String(120))
    body: Mapped[str] = mapped_column(Text)
    # Internal notes are for agents only and are never sent to the requester.
    internal: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="comments")


class Webhook(Base):
    __tablename__ = "webhooks"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    url: Mapped[str] = mapped_column(String(2000))
    secret: Mapped[str] = mapped_column(String(80))
    events: Mapped[str] = mapped_column(String(200))  # space-separated event names
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    @property
    def event_list(self) -> list[str]:
        return sorted(self.events.split())


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    webhook_id: Mapped[int] = mapped_column(ForeignKey("webhooks.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(String(40))
    payload: Mapped[str] = mapped_column(Text)  # the exact JSON body that is signed and sent
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    attempted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status_code: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(String(500))

    webhook: Mapped[Webhook] = relationship()

    @property
    def delivered(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300
