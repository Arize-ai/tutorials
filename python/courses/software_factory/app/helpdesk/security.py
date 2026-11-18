"""Passwords, API keys and signatures."""

from __future__ import annotations

import functools
import hashlib
import hmac
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ApiKey

PBKDF2_ITERATIONS = 600_000
API_KEY_PREFIX = "hd_"
_PREFIX_LEN = 8


def hash_password(password: str, *, iterations: int | None = None) -> str:
    iterations = iterations or PBKDF2_ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
    )
    return hmac.compare_digest(candidate.hex(), digest)


@functools.cache
def _dummy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


def burn_password_check(password: str) -> None:
    """Run when a login names an unknown user, so a miss takes as long as a wrong password."""
    verify_password(password, _dummy_hash())


def _hash_key(token: str) -> str:
    # API keys are 256 bits of randomness, so a fast hash is enough; no salt or stretching.
    return hashlib.sha256(token.encode()).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    """Returns (token, prefix, hash). Show the token once; store only the prefix and hash."""
    token = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return token, token[len(API_KEY_PREFIX) : len(API_KEY_PREFIX) + _PREFIX_LEN], _hash_key(token)


def find_api_key(db: Session, token: str) -> ApiKey | None:
    """The active key this token belongs to, or None."""
    if not token.startswith(API_KEY_PREFIX):
        return None
    prefix = token[len(API_KEY_PREFIX) : len(API_KEY_PREFIX) + _PREFIX_LEN]
    key = db.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
    if key is None or key.revoked_at is not None:
        return None
    if not hmac.compare_digest(key.key_hash, _hash_key(token)):
        return None
    return key


def new_webhook_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(24)


def sign_payload(secret: str, timestamp: int, body: bytes) -> str:
    """The X-Helpdesk-Signature header: t=<unix time>,v1=<hex HMAC-SHA256 of "<t>.<body>">."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={mac}"
