"""Capability-fed routing for document extraction and OCR."""

from __future__ import annotations

from threading import RLock

from marie.extraction.models import CapabilitySnapshot


class FormatRouter:
    """Route formats from an atomically replaceable plugin capability snapshot."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._formats: frozenset[str] = frozenset()
        self._aliases: dict[str, str] = {}
        self._plugin_version: str | None = None

    @property
    def plugin_version(self) -> str | None:
        with self._lock:
            return self._plugin_version

    @property
    def plugin_formats(self) -> frozenset[str]:
        with self._lock:
            return self._formats

    def ingest_capabilities(self, payload: object) -> CapabilitySnapshot:
        """Validate and replace the complete plugin capability snapshot."""
        snapshot = CapabilitySnapshot.model_validate(payload)
        formats: set[str] = set()
        aliases: dict[str, str] = {}
        if snapshot.ready:
            for capability in snapshot.formats:
                if "semantic" not in capability.intents:
                    continue
                canonical = capability.canonical_format.lower()
                formats.add(canonical)
                for value in [canonical, *capability.aliases, *capability.extensions]:
                    aliases[value.lower().lstrip(".")] = canonical

        with self._lock:
            self._formats = frozenset(formats)
            self._aliases = aliases
            self._plugin_version = snapshot.plugin_version
        return snapshot

    def clear_capabilities(self) -> None:
        """Replace the current snapshot with an empty one."""
        with self._lock:
            self._formats = frozenset()
            self._aliases = {}
            self._plugin_version = None

    def route(
        self,
        file_type: str,
        parse_mode: str | None,
        *,
        ocr_supported: bool,
    ) -> str:
        """Return ``plugin``, ``ocr``, or ``unsupported`` for the source."""
        if parse_mode == "ocr":
            return "ocr" if ocr_supported else "unsupported"

        normalized = (file_type or "").lower().lstrip(".")
        with self._lock:
            canonical = self._aliases.get(normalized, normalized)
            plugin_supported = canonical in self._formats
        if plugin_supported:
            return "plugin"
        if ocr_supported:
            return "ocr"
        return "unsupported"
