"""FastAPI dependencies: database sessions and authentication."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from . import security
from .db import utcnow
from .models import Role, Scope, User
from .principal import Principal


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.sessionmaker() as db:
        yield db


DB = Annotated[Session, Depends(get_db)]


# ---------- API: bearer API keys ----------


def api_principal(request: Request, db: DB) -> Principal:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Send an API key as 'Authorization: Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    key = security.find_api_key(db, token.strip())
    if key is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    key.last_used_at = utcnow()
    db.commit()
    return Principal.for_api_key(key)


def require(scope: Scope):
    def check(principal: Annotated[Principal, Depends(api_principal)]) -> Principal:
        if not principal.can(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"This API key needs the {scope} scope")
        return principal

    return Depends(check)


CanRead = Annotated[Principal, require(Scope.tickets_read)]
CanWrite = Annotated[Principal, require(Scope.tickets_write)]
IsAdmin = Annotated[Principal, require(Scope.admin)]


# ---------- web UI: session cookie ----------


class LoginRequired(Exception):
    pass


def web_user(request: Request, db: DB) -> User:
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id else None
    if user is None:
        request.session.pop("user_id", None)
        raise LoginRequired
    return user


def web_admin(user: Annotated[User, Depends(web_user)]) -> User:
    if user.role != Role.admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only workspace admins can open settings")
    return user


WebUser = Annotated[User, Depends(web_user)]
WebAdmin = Annotated[User, Depends(web_admin)]


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


async def csrf_protect(request: Request) -> None:
    """Every form POST in the web UI must echo the session's CSRF token."""
    if request.method != "POST":
        return
    form = await request.form()
    expected = request.session.get("csrf")
    sent = form.get("csrf_token")
    if not expected or not isinstance(sent, str) or not secrets.compare_digest(sent, expected):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "The form expired. Reload the page and retry."
        )
