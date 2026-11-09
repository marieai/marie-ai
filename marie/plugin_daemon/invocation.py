"""Action identity and installed-package claims for plugin invocation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from marie.secret_store import CredentialRequirement

PluginActionType = Literal[
    'tool', 'model', 'datasource', 'trigger', 'endpoint', 'mcp', 'webapp', 'stub'
]


class PluginInvocationSpec(BaseModel):
    """Resolved invocation claims; callers must authorize the declared action."""

    action_type: PluginActionType
    action_id: str
    install_id: str | None = None
    provider_id: str | None = None
    provider_ref: str | None = None
    package_id: str | None = None
    package_ref: str | None = None
    package_digest: str | None = None
    package_trust_level: str | None = None
    credential_requirements: list[CredentialRequirement] = Field(default_factory=list)
    credential_binding_ids: list[str] = Field(default_factory=list)

    @field_validator('action_id')
    @classmethod
    def require_action_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('plugin invocation requires an action_id')
        return value
