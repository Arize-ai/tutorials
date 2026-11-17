"""The JSON API, under /api/v1. Authenticate with 'Authorization: Bearer <API key>'."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response, status

from . import services
from .deps import DB, CanRead, CanWrite, IsAdmin
from .models import Priority, Status
from .schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    CommentCreate,
    CommentOut,
    DeliveryOut,
    MeOut,
    TenantOut,
    TicketCreate,
    TicketOut,
    TicketUpdate,
    UserOut,
    WebhookCreate,
    WebhookCreated,
    WebhookOut,
)
from .webhooks import deliver_pending

router = APIRouter(prefix="/api/v1", tags=["api"])


def send_webhooks(request: Request, tasks: BackgroundTasks) -> None:
    """Deliver queued webhooks once the response has gone out."""
    state = request.app.state
    if state.settings.deliver_webhooks:
        tasks.add_task(deliver_pending, state.sessionmaker, state.http_client)


@router.get("/me", response_model=MeOut)
def me(principal: CanRead) -> MeOut:
    return MeOut(
        tenant=TenantOut.model_validate(principal.tenant),
        actor=principal.name,
        scopes=sorted(principal.scopes),
    )


@router.get("/users", response_model=list[UserOut])
def list_users(principal: CanRead, db: DB):
    return services.list_users(db, principal.tenant_id)


# ---------- tickets ----------


@router.get("/tickets", response_model=list[TicketOut])
def list_tickets(
    principal: CanRead,
    db: DB,
    status_: list[Status] | None = Query(None, alias="status"),
    priority: Priority | None = None,
    assignee_id: int | None = None,
    limit: int = Query(services.MAX_PAGE, ge=1, le=services.MAX_PAGE),
):
    return services.list_tickets(
        db,
        principal.tenant_id,
        statuses=tuple(status_) if status_ else None,
        priority=priority,
        assignee_id=assignee_id,
        limit=limit,
    )


@router.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(
    body: TicketCreate, principal: CanWrite, db: DB, request: Request, tasks: BackgroundTasks
):
    ticket = services.create_ticket(db, principal, **body.model_dump())
    send_webhooks(request, tasks)
    return ticket


@router.get("/tickets/{number}", response_model=TicketOut)
def get_ticket(number: int, principal: CanRead, db: DB):
    return services.get_ticket(db, principal.tenant_id, number)


@router.patch("/tickets/{number}", response_model=TicketOut)
def update_ticket(
    number: int,
    body: TicketUpdate,
    principal: CanWrite,
    db: DB,
    request: Request,
    tasks: BackgroundTasks,
):
    ticket = services.get_ticket(db, principal.tenant_id, number)
    ticket = services.update_ticket(db, principal, ticket, body.model_dump(exclude_unset=True))
    send_webhooks(request, tasks)
    return ticket


@router.get("/tickets/{number}/comments", response_model=list[CommentOut])
def list_comments(number: int, principal: CanRead, db: DB):
    return services.get_ticket(db, principal.tenant_id, number).comments


@router.post(
    "/tickets/{number}/comments", response_model=CommentOut, status_code=status.HTTP_201_CREATED
)
def add_comment(
    number: int,
    body: CommentCreate,
    principal: CanWrite,
    db: DB,
    request: Request,
    tasks: BackgroundTasks,
):
    ticket = services.get_ticket(db, principal.tenant_id, number)
    comment = services.add_comment(db, principal, ticket, body=body.body, internal=body.internal)
    send_webhooks(request, tasks)
    return comment


# ---------- admin: API keys and webhooks ----------


@router.get("/api-keys", response_model=list[ApiKeyOut])
def list_api_keys(principal: IsAdmin, db: DB):
    return services.list_api_keys(db, principal.tenant_id)


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_api_key(body: ApiKeyCreate, principal: IsAdmin, db: DB):
    key, token = services.create_api_key(
        db, principal.tenant_id, name=body.name, scopes=body.scopes
    )
    return ApiKeyCreated(**ApiKeyOut.model_validate(key).model_dump(), key=token)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(key_id: int, principal: IsAdmin, db: DB) -> Response:
    services.revoke_api_key(db, principal.tenant_id, key_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/webhooks", response_model=list[WebhookOut])
def list_webhooks(principal: IsAdmin, db: DB):
    return services.list_webhooks(db, principal.tenant_id)


@router.post("/webhooks", response_model=WebhookCreated, status_code=status.HTTP_201_CREATED)
def create_webhook(body: WebhookCreate, principal: IsAdmin, db: DB):
    hook = services.create_webhook(db, principal.tenant_id, url=str(body.url), events=body.events)
    return WebhookCreated(**WebhookOut.model_validate(hook).model_dump(), secret=hook.secret)


@router.delete("/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_webhook(webhook_id: int, principal: IsAdmin, db: DB) -> Response:
    services.delete_webhook(db, principal.tenant_id, webhook_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/webhooks/{webhook_id}/deliveries", response_model=list[DeliveryOut])
def list_deliveries(webhook_id: int, principal: IsAdmin, db: DB):
    services.get_webhook(db, principal.tenant_id, webhook_id)
    return services.list_deliveries(db, principal.tenant_id, webhook_id)
