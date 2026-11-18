"""Settings, read from the environment (and a .env file if you load one yourself)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEV_SECRET = "dev-only-secret-change-me"  # noqa: S105 - a placeholder, refused outside dev


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./helpdesk.db"
    secret_key: str = DEV_SECRET
    # Send webhook deliveries after each request. Off means they queue up undelivered.
    deliver_webhooks: bool = True
    webhook_timeout_s: float = 5.0
    # Mark the session cookie Secure. Turn on whenever the app is served over HTTPS.
    secure_cookies: bool = False
    # dev allows the placeholder secret key; anything else refuses to start without a real one.
    env: str = "dev"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.getenv("HELPDESK_DATABASE_URL", cls.database_url),
            secret_key=os.getenv("HELPDESK_SECRET_KEY", DEV_SECRET),
            deliver_webhooks=_flag("HELPDESK_DELIVER_WEBHOOKS", True),
            webhook_timeout_s=float(os.getenv("HELPDESK_WEBHOOK_TIMEOUT_S", "5")),
            secure_cookies=_flag("HELPDESK_SECURE_COOKIES", False),
            env=os.getenv("HELPDESK_ENV", "dev"),
        )

    def check(self) -> None:
        if self.env != "dev" and self.secret_key == DEV_SECRET:
            raise RuntimeError(
                "Set HELPDESK_SECRET_KEY: the placeholder key is only allowed in dev"
            )
