"""Provider-neutral document extraction dispatch."""

from __future__ import annotations

from .artifacts import write_document_artifact
from .detection import detect_format
from .models import (
    ExtractionSuccess,
    NotExtractable,
    ProviderProvenance,
)
from .providers.base import (
    ProviderNotExtractableError,
    ProviderUnavailableError,
)
from .registry import output_format_ids, provider_ids, providers_for


def extract_document(
    *,
    path: str,
    format_hint: str | None = None,
    mime_type: str | None = None,
    intent: str = 'semantic',
    output_dir: str | None = None,
    provider: str | None = None,
    fallback: bool = True,
    provider_options: dict | None = None,
    output_format: str = 'markdown',
) -> dict:
    """Detect, dispatch, and write one terminal extraction result artifact."""
    if intent != 'semantic':
        raise ValueError(f'Unsupported extraction intent: {intent!r}')
    if provider_options is not None and not isinstance(provider_options, dict):
        raise ValueError('provider_options must be an object')
    if output_format not in output_format_ids():
        raise ValueError(f'unknown output format: {output_format!r}')
    detection = detect_format(path, format_hint=format_hint, mime_type=mime_type)
    canonical_format = detection.canonical_format
    warnings = []
    candidates = []
    for candidate in providers_for(canonical_format):
        if output_format in candidate.output_formats:
            candidates.append(candidate)
        else:
            warnings.append(
                f'{candidate.provider_id}: cannot produce {output_format} output'
            )
    if provider is not None:
        if provider not in provider_ids():
            raise ValueError(f'unknown provider: {provider!r}')
        preferred = [
            candidate for candidate in candidates if candidate.provider_id == provider
        ]
        if not preferred:
            warnings.append(
                f'{provider}: requested provider is not ready for {canonical_format}'
            )
        others = [
            candidate for candidate in candidates if candidate.provider_id != provider
        ]
        candidates = preferred + others if fallback else preferred
    elif not fallback:
        candidates = candidates[:1]
    if not candidates:
        return NotExtractable(
            canonical_format=canonical_format,
            reason='no_ready_provider',
            warnings=warnings,
        ).model_dump(mode='json')

    attempted = []
    for candidate in candidates:
        attempted.append(candidate.provider_id)
        try:
            document = candidate.extract(
                path,
                canonical_format,
                options=provider_options,
                output_format=output_format,
            )
        except (ProviderUnavailableError, ProviderNotExtractableError) as error:
            warnings.append(f'{candidate.provider_id}: {error}')
            continue

        artifact = write_document_artifact(
            document.content,
            output_dir=output_dir,
            media_type=document.media_type,
        )
        result = ExtractionSuccess(
            result_kind=document.result_kind,
            artifact=artifact,
            provenance=ProviderProvenance(
                provider=document.provider,
                provider_version=document.provider_version,
                canonical_format=canonical_format,
                backend=document.backend,
            ),
            metadata={
                **document.metadata,
                'detection_evidence': list(detection.evidence),
            },
            warnings=[*warnings, *document.warnings],
        )
        return result.model_dump(mode='json')

    return NotExtractable(
        canonical_format=canonical_format,
        reason='providers_exhausted',
        attempted_providers=attempted,
        warnings=warnings,
    ).model_dump(mode='json')
