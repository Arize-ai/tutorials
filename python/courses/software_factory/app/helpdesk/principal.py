"""Who is making a request: a signed-in user or an API key, always within one tenant."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ROLE_SCOPES, ApiKey, Scope, Tenant, User


@dataclass(frozen=True)
class Principal:
    tenant: Tenant
    scopes: frozenset[Scope]
    user: User | None = None
    api_key: ApiKey | None = None

    @classmethod
    def for_user(cls, user: User) -> Principal:
        return cls(tenant=user.tenant, scopes=ROLE_SCOPES[user.role], user=user)

    @classmethod
    def for_api_key(cls, key: ApiKey) -> Principal:
        return cls(tenant=key.tenant, scopes=key.scope_set, api_key=key)

    @property
    def tenant_id(self) -> int:
        return self.tenant.id

    @property
    def name(self) -> str:
        if self.user is not None:
            return self.user.name
        if self.api_key is not None:
            return f"API key: {self.api_key.name}"
        return "unknown"

    def can(self, scope: Scope) -> bool:
        return Scope.admin in self.scopes or scope in self.scopes
