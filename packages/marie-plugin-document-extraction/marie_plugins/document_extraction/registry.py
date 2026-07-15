"""Provider registry and aggregate capability reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .detection import EXTENSIONS, MIME_TYPES
from .models import CapabilitySnapshot, FormatCapability, ResultKind
from .providers.base import ExtractionProvider
from .providers.docling import DoclingProvider
from .providers.markitdown import MarkItDownProvider
from .providers.tree_sitter import TreeSitterProvider

PLUGIN_VERSION = '0.2.0'

_PROVIDERS: tuple[ExtractionProvider, ...] = (
    DoclingProvider(),
    MarkItDownProvider(),
    TreeSitterProvider(),
)


def providers_for(canonical_format: str) -> list[ExtractionProvider]:
    """Return ready providers in deterministic preference order."""
    return [
        provider
        for provider in _PROVIDERS
        if canonical_format in provider.formats and provider.is_ready(canonical_format)
    ]


def provider_ids() -> frozenset[str]:
    """Return the registered provider identifiers."""
    return frozenset(provider.provider_id for provider in _PROVIDERS)


def output_format_ids() -> frozenset[str]:
    """Return every output format some registered provider can produce."""
    return frozenset(
        output_format
        for provider in _PROVIDERS
        for output_format in provider.output_formats
    )


def capability_snapshot_model(
    providers: Iterable[ExtractionProvider] | None = None,
) -> CapabilitySnapshot:
    """Build the aggregate snapshot from installed provider edges."""
    selected = tuple(providers) if providers is not None else _PROVIDERS
    ready_providers: dict[str, list[str]] = defaultdict(list)
    for provider in selected:
        for canonical_format in sorted(provider.formats):
            if provider.is_ready(canonical_format):
                ready_providers[canonical_format].append(provider.provider_id)

    formats = []
    for canonical_format, provider_ids in sorted(ready_providers.items()):
        extensions = sorted(
            extension
            for extension, value in EXTENSIONS.items()
            if value == canonical_format
        )
        mime_types = sorted(
            mime for mime, value in MIME_TYPES.items() if value == canonical_format
        )
        aliases = sorted(set(extensions) - {canonical_format})
        formats.append(
            FormatCapability(
                canonical_format=canonical_format,
                aliases=aliases,
                extensions=extensions,
                mime_types=mime_types,
                result_kinds=[ResultKind.SEMANTIC_DOCUMENT],
                providers=provider_ids,
            )
        )
    return CapabilitySnapshot(
        plugin_version=PLUGIN_VERSION,
        ready=bool(formats),
        formats=formats,
    )


def capability_snapshot() -> dict:
    return capability_snapshot_model().model_dump(mode='json')
