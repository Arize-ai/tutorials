"""Command line: `python -m helpdesk serve | seed | smoke | create-user`."""

from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import select

from . import services
from .config import Settings
from .db import Base, make_engine, make_sessionmaker
from .models import Role, Tenant


def _session():
    settings = Settings.from_env()
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    return make_sessionmaker(engine)()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "helpdesk.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload
    )


def cmd_seed(_args: argparse.Namespace) -> None:
    from .seed import DEMO_PASSWORD, WORKSPACES, seed

    with _session() as db:
        keys = seed(db)
    if not keys:
        print("Demo workspaces already exist.")
        return
    print(f"Created demo workspaces. Every demo user's password is {DEMO_PASSWORD!r}.\n")
    for slug, token in keys.items():
        print(f"  workspace {slug}")
        for email, _name, role in WORKSPACES[slug]["users"]:
            print(f"    {role.value:<6} {email}")
        print(f"    admin API key: {token}\n")


def cmd_smoke(_args: argparse.Namespace) -> None:
    from .smoke import run_smoke

    if not run_smoke():
        sys.exit(1)


def cmd_create_user(args: argparse.Namespace) -> None:
    with _session() as db:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == args.workspace.lower()))
        if tenant is None:
            tenant = services.create_tenant(
                db, args.workspace, args.workspace_name or args.workspace
            )
            print(f"Created workspace {tenant.slug}.")
        password = getpass.getpass(f"Password for {args.email}: ")
        if len(password) < 10:
            sys.exit("Use a password of at least 10 characters.")
        services.create_user(
            db, tenant, email=args.email, name=args.name, password=password, role=Role(args.role)
        )
    print(f"Created {args.role} {args.email} in {args.workspace}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m helpdesk")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web app and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="restart on code changes")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("seed", help="create demo workspaces, users and tickets").set_defaults(
        func=cmd_seed
    )

    sub.add_parser("smoke", help="run an end-to-end check on a fresh in-memory app").set_defaults(
        func=cmd_smoke
    )

    user = sub.add_parser("create-user", help="add a user, creating the workspace if needed")
    user.add_argument("workspace")
    user.add_argument("email")
    user.add_argument("name")
    user.add_argument("--role", choices=[r.value for r in Role], default=Role.agent.value)
    user.add_argument("--workspace-name", help="display name if the workspace is new")
    user.set_defaults(func=cmd_create_user)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
