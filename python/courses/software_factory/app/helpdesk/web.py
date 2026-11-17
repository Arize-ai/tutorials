"""The web UI for support agents: server-rendered pages and plain HTML forms."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import services
from .api import send_webhooks
from .deps import DB, WebAdmin, WebUser, csrf_protect, csrf_token
from .models import Priority, Role, Scope, Status
from .principal import Principal
from .services import ACTIVE_STATUSES, InvalidError
from .webhooks import Event

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
router = APIRouter(dependencies=[Depends(csrf_protect)], include_in_schema=False)

VIEWS = {"active": ACTIVE_STATUSES, "all": None} | {s.value: (s,) for s in Status}


def flash(request: Request, text: str, kind: str = "info", *, code: str | None = None) -> None:
    request.session.setdefault("flash", []).append({"kind": kind, "text": text, "code": code})


def render(
    request: Request, name: str, user=None, status_code: int = 200, **context
) -> HTMLResponse:
    context |= {
        "user": user,
        "csrf_token": csrf_token(request),
        "flashes": request.session.pop("flash", []),
        "Status": Status,
        "Priority": Priority,
    }
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _safe_next(url: str | None) -> str:
    """Only follow same-site paths after login, never another host."""
    if url and url.startswith("/") and not url.startswith("//") and "\\" not in url:
        return url
    return "/tickets"


# ---------- sign in ----------


@router.get("/", response_class=HTMLResponse)
def home() -> RedirectResponse:
    return redirect("/tickets")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str | None = None) -> HTMLResponse:
    return render(request, "login.html", next=_safe_next(next), workspace="", email="")


@router.post("/login")
def login(
    request: Request,
    db: DB,
    workspace: Annotated[str, Form()],
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    next: Annotated[str, Form()] = "/tickets",
):
    user = services.authenticate(db, workspace, email, password)
    if user is None:
        flash(request, "That workspace, email and password don't match.", "error")
        return render(
            request,
            "login.html",
            status_code=status.HTTP_401_UNAUTHORIZED,
            next=_safe_next(next),
            workspace=workspace,
            email=email,
        )
    request.session.clear()  # new session on sign-in, so a planted session id is worthless
    request.session["user_id"] = user.id
    return redirect(_safe_next(next))


@router.post("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return redirect("/login")


# ---------- tickets ----------


@router.get("/tickets", response_class=HTMLResponse)
def ticket_list(
    request: Request,
    user: WebUser,
    db: DB,
    view: str = "active",
    mine: bool = False,
    priority: Priority | None = None,
) -> HTMLResponse:
    if view not in VIEWS:
        view = "active"
    tickets = services.list_tickets(
        db,
        user.tenant_id,
        statuses=VIEWS[view],
        priority=priority,
        assignee_id=user.id if mine else None,
    )
    counts = services.count_by_status(db, user.tenant_id)
    return render(
        request,
        "tickets.html",
        user=user,
        tickets=tickets,
        counts=counts,
        active_count=sum(counts[s] for s in ACTIVE_STATUSES),
        view=view,
        mine=mine,
        priority=priority,
        query=lambda **kw: urlencode(
            {
                k: v
                for k, v in (
                    {"view": view, "mine": mine or None, "priority": priority} | kw
                ).items()
                if v
            }
        ),
    )


@router.get("/tickets/new", response_class=HTMLResponse)
def new_ticket_page(request: Request, user: WebUser, db: DB) -> HTMLResponse:
    return render(
        request,
        "ticket_new.html",
        user=user,
        agents=services.list_users(db, user.tenant_id),
        form={},
    )


@router.post("/tickets")
def create_ticket(
    request: Request,
    user: WebUser,
    db: DB,
    tasks: BackgroundTasks,
    subject: Annotated[str, Form()],
    requester_email: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    priority: Annotated[Priority, Form()] = Priority.normal,
    assignee_id: Annotated[str, Form()] = "",
):
    form = {
        "subject": subject.strip(),
        "requester_email": requester_email.strip().lower(),
        "description": description.strip(),
        "priority": priority,
        "assignee_id": int(assignee_id) if assignee_id.isdigit() else None,
    }
    error = None
    if not form["subject"]:
        error = "Add a subject."
    elif "@" not in form["requester_email"]:
        error = "Add the requester's email address."
    if error is None:
        try:
            ticket = services.create_ticket(db, Principal.for_user(user), **form)
        except InvalidError as e:
            error = str(e)
    if error:
        flash(request, error, "error")
        agents = services.list_users(db, user.tenant_id)
        return render(
            request, "ticket_new.html", user=user, agents=agents, form=form, status_code=422
        )
    send_webhooks(request, tasks)
    flash(request, f"Ticket #{ticket.number} created.")
    return redirect(f"/tickets/{ticket.number}")


@router.get("/tickets/{number}", response_class=HTMLResponse)
def ticket_page(request: Request, number: int, user: WebUser, db: DB) -> HTMLResponse:
    ticket = services.get_ticket(db, user.tenant_id, number)
    return render(
        request,
        "ticket.html",
        user=user,
        ticket=ticket,
        agents=services.list_users(db, user.tenant_id),
    )


@router.post("/tickets/{number}")
def update_ticket(
    request: Request,
    number: int,
    user: WebUser,
    db: DB,
    tasks: BackgroundTasks,
    status_: Annotated[Status, Form(alias="status")],
    priority: Annotated[Priority, Form()],
    assignee_id: Annotated[str, Form()] = "",
):
    ticket = services.get_ticket(db, user.tenant_id, number)
    changes = {
        "status": status_,
        "priority": priority,
        "assignee_id": int(assignee_id) if assignee_id.isdigit() else None,
    }
    try:
        services.update_ticket(db, Principal.for_user(user), ticket, changes)
    except InvalidError as e:
        flash(request, str(e), "error")
    else:
        send_webhooks(request, tasks)
        flash(request, "Ticket updated.")
    return redirect(f"/tickets/{number}")


@router.post("/tickets/{number}/comments")
def add_comment(
    request: Request,
    number: int,
    user: WebUser,
    db: DB,
    tasks: BackgroundTasks,
    body: Annotated[str, Form()],
    internal: Annotated[bool, Form()] = False,
):
    ticket = services.get_ticket(db, user.tenant_id, number)
    if not body.strip():
        flash(request, "Write a reply first.", "error")
    else:
        services.add_comment(
            db, Principal.for_user(user), ticket, body=body.strip(), internal=internal
        )
        send_webhooks(request, tasks)
        flash(request, "Internal note added." if internal else "Reply added.")
    return redirect(f"/tickets/{number}#comments")


# ---------- settings (admins only) ----------


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: WebAdmin, db: DB) -> HTMLResponse:
    return render(
        request,
        "settings.html",
        user=user,
        users=services.list_users(db, user.tenant_id),
        keys=services.list_api_keys(db, user.tenant_id),
        hooks=services.list_webhooks(db, user.tenant_id),
        deliveries=services.list_deliveries(db, user.tenant_id, limit=20),
        scopes=list(Scope),
        events=list(Event),
        Role=Role,
    )


@router.post("/settings/api-keys")
def create_api_key(
    request: Request,
    user: WebAdmin,
    db: DB,
    name: Annotated[str, Form()],
    scopes: Annotated[list[Scope], Form()] = (),
):
    if not name.strip() or not scopes:
        flash(request, "Give the key a name and at least one scope.", "error")
        return redirect("/settings#api-keys")
    key, token = services.create_api_key(db, user.tenant_id, name=name.strip(), scopes=list(scopes))
    flash(request, f"Created “{key.name}”. Copy the key now: it won't be shown again.", code=token)
    return redirect("/settings#api-keys")


@router.post("/settings/api-keys/{key_id}/revoke")
def revoke_api_key(request: Request, key_id: int, user: WebAdmin, db: DB):
    key = services.revoke_api_key(db, user.tenant_id, key_id)
    flash(request, f"Revoked “{key.name}”.")
    return redirect("/settings#api-keys")


@router.post("/settings/webhooks")
def create_webhook(
    request: Request,
    user: WebAdmin,
    db: DB,
    url: Annotated[str, Form()],
    events: Annotated[list[Event], Form()] = (),
):
    if not events:
        flash(request, "Pick at least one event.", "error")
        return redirect("/settings#webhooks")
    try:
        hook = services.create_webhook(db, user.tenant_id, url=url.strip(), events=list(events))
    except InvalidError as e:
        flash(request, str(e), "error")
    else:
        flash(request, "Webhook added. Its signing secret:", code=hook.secret)
    return redirect("/settings#webhooks")


@router.post("/settings/webhooks/{webhook_id}/delete")
def delete_webhook(request: Request, webhook_id: int, user: WebAdmin, db: DB):
    services.delete_webhook(db, user.tenant_id, webhook_id)
    flash(request, "Webhook deleted.")
    return redirect("/settings#webhooks")
