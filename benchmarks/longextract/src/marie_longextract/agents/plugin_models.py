"""Validated input models for daemon-managed LongExtract agent actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class WorkspaceCapability(_StrictModel):
    root: str = Field(min_length=1)
    access: Literal['read_only']

    def path(self) -> Path:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise ValueError(f'Workspace does not exist: {root}')
        return root


class ModelProfile(_StrictModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0, le=1800)


class BoundaryRepairInput(_StrictModel):
    page_number: int = Field(ge=1)
    record_index: int = Field(ge=0)


class LeafRepairInput(_StrictModel):
    page_numbers: list[int] = Field(min_length=1)
    field_names: list[str] | None = None

    @field_validator('page_numbers')
    @classmethod
    def validate_page_numbers(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError('page_numbers must contain positive integers')
        if len(values) != len(set(values)):
            raise ValueError('page_numbers must not contain duplicates')
        return values


class AgentInvocation(_StrictModel):
    agent_ref: str = Field(min_length=1)
    input: dict[str, Any]
    artifacts: dict[str, str]
    idempotency_key: str = Field(min_length=1)
    model_profile: ModelProfile
    workspace: WorkspaceCapability
    action: str = Field(min_length=1)
    execution: dict[str, Any]
    user_id: str = ''
    credentials: dict[str, str]

    def api_key(self) -> str:
        value = self.credentials.get('llm_api_key', '').strip()
        if not value:
            raise ValueError('llm_api_key credential is required')
        return value

    def artifact_path(self, name: str) -> Path:
        relative = self.artifacts.get(name)
        if not relative:
            raise ValueError(f'{name} artifact is required')
        root = self.workspace.path()
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f'{name} artifact is outside the workspace or missing')
        return candidate
