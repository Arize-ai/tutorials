"""Request and response bodies for the JSON API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from .models import Priority, Role, Scope, Status
from .webhooks import Event

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
Email = Annotated[str, Field(max_length=254, pattern=EMAIL_PATTERN)]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True, validate_by_name=True)


class UserOut(_Out):
    id: int
    email: str
    name: str
    role: Role


class TenantOut(_Out):
    slug: str
    name: str


class MeOut(BaseModel):
    tenant: TenantOut
    actor: str
    scopes: list[Scope]


class TicketCreate(_In):
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=20_000)
    requester_email: Email
    priority: Priority = Priority.normal
    assignee_id: int | None = None


class TicketUpdate(_In):
    """Only the fields you send are changed. Send "assignee_id": null to unassign."""

    subject: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=20_000)
    status: Status | None = None
    priority: Priority | None = None
    assignee_id: int | None = None


class TicketOut(_Out):
    number: int
    subject: str
    description: str
    status: Status
    priority: Priority
    requester_email: str
    assignee: UserOut | None
    created_at: datetime
    updated_at: datetime


class CommentCreate(_In):
    body: str = Field(min_length=1, max_length=20_000)
    internal: bool = False


class CommentOut(_Out):
    id: int
    author_name: str
    body: str
    internal: bool
    created_at: datetime


class ApiKeyCreate(_In):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[Scope] = Field(min_length=1)


class ApiKeyOut(_Out):
    id: int
    name: str
    prefix: str
    scopes: list[str] = Field(validation_alias="scope_list")
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str = Field(description="The full key. It is only ever shown in this response.")


class WebhookCreate(_In):
    url: AnyHttpUrl
    events: list[Event] = Field(min_length=1)


class WebhookOut(_Out):
    id: int
    url: str
    events: list[str] = Field(validation_alias="event_list")
    active: bool
    created_at: datetime


class WebhookCreated(WebhookOut):
    secret: str = Field(description="Signing secret. It is only ever shown in this response.")


class DeliveryOut(_Out):
    id: int
    event: str
    created_at: datetime
    attempted_at: datetime | None
    status_code: int | None
    error: str | None
    delivered: bool
