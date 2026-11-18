"""The application factory. Run it with `uv run python -m helpdesk serve`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware

from . import __version__, api, web
from .config import Settings
from .db import Base, make_engine, make_sessionmaker
from .deps import LoginRequired
from .services import InvalidError, NotFoundError


def create_app(
    settings: Settings | None = None, *, http_client: httpx.Client | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.check()
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    client = http_client or httpx.Client(timeout=settings.webhook_timeout_s, follow_redirects=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        client.close()
        engine.dispose()

    app = FastAPI(
        title="Helpdesk",
        version=__version__,
        description="A multi-tenant helpdesk API. Send `Authorization: Bearer <API key>`.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = make_sessionmaker(engine)
    app.state.http_client = client

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="helpdesk_session",
        same_site="lax",
        https_only=settings.secure_cookies,
        max_age=60 * 60 * 12,
    )
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(api.router)
    app.include_router(web.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    def wants_json(request: Request) -> bool:
        return request.url.path.startswith("/api/")

    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, _exc: LoginRequired):
        target = "/login?" + urlencode({"next": request.url.path})
        return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError):
        if wants_json(request):
            return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_404_NOT_FOUND)
        return web.render(
            request, "error.html", status_code=404, title="Not found", message=str(exc)
        )

    @app.exception_handler(InvalidError)
    async def invalid(request: Request, exc: InvalidError):
        if wants_json(request):
            return JSONResponse(
                {"detail": str(exc)}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
            )
        return web.render(
            request, "error.html", status_code=422, title="Can't do that", message=str(exc)
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if wants_json(request) or exc.status_code < 400:
            return JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )
        return web.render(
            request,
            "error.html",
            status_code=exc.status_code,
            title="Something's wrong",
            message=exc.detail,
        )

    return app
