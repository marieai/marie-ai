from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class AgentPluginModelProfile(_StrictModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    request_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)


class AgentPluginRoute(_StrictModel):
    package: str = Field(min_length=1)
    action: str = Field(min_length=1)
    model_profile: AgentPluginModelProfile | None = None
    requires_workspace: bool = False


class AgentPluginRequest(_StrictModel):
    agent_ref: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    model_profile: str | None = None
    idempotency_key: str = Field(min_length=1)

    @model_validator(mode='after')
    def reject_host_configuration(self) -> 'AgentPluginRequest':
        forbidden = _find_forbidden_key(self.input)
        if forbidden:
            raise ValueError(f'request input cannot provide {forbidden}')
        for name, uri in self.artifacts.items():
            if not name.strip() or not uri:
                raise ValueError('artifacts must use non-empty names and URIs')
            parsed = urlparse(uri)
            relative_parts = PurePosixPath(parsed.path).parts
            if (
                parsed.scheme.casefold() == 'file'
                or uri.startswith('/')
                or '\\' in uri
                or '..' in relative_parts
            ):
                raise ValueError(
                    'artifact paths must not select a host filesystem path'
                )
        return self


class AgentPluginResponse(_StrictModel):
    agent_ref: str
    result: dict[str, Any]
    frames: list[dict[str, Any]]
    request_id: str
    trace_id: str


_FORBIDDEN_INPUT_KEYS = {
    'apikey',
    'baseurl',
    'credentialbindingids',
    'credentials',
    'daemonaddr',
    'module',
    'modulename',
    'package',
    'packageid',
    'packagepath',
    'plugindaemonaddr',
    'runtimepolicy',
    'secretref',
    'workspace',
    'workspaceroot',
}


def _find_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = ''.join(
                character
                for character in str(key).strip().casefold()
                if character.isalnum()
            )
            if normalized in _FORBIDDEN_INPUT_KEYS:
                return normalized
            nested = _find_forbidden_key(item)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_forbidden_key(item)
            if nested:
                return nested
    return None
