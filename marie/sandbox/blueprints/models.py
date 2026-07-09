"""Pydantic models for the blueprint-import gateway endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactResult(BaseModel):
    """Outcome of installing one blueprint artifact."""

    ref: str = Field(
        ..., description='Blueprint-manifest artifact ref (e.g. workflow/my-plan)'
    )
    kind: str = Field(..., description='Artifact kind as declared in the manifest')
    status: Literal['applied', 'deferred', 'failed'] = Field(
        ...,
        description=(
            'applied — installed into this gateway; '
            'deferred — no home for this kind yet (dify-parity gap or missing config); '
            'failed — attempted and errored'
        ),
    )
    reason: str | None = Field(
        None, description='Human-readable explanation for deferred/failed'
    )


class BlueprintImportResponse(BaseModel):
    """Response body for POST /api/v1/blueprints/import."""

    blueprint_id: str
    status: Literal['completed', 'partial', 'failed'] = Field(
        ...,
        description=(
            'completed — all artifacts applied; '
            'partial — some applied, some deferred (expected during dify-parity landing); '
            'failed — no artifacts could be applied or a hard error occurred'
        ),
    )
    applied: list[ArtifactResult] = Field(default_factory=list)
    deferred: list[ArtifactResult] = Field(default_factory=list)
    failed: list[ArtifactResult] = Field(default_factory=list)
    message: str | None = None
