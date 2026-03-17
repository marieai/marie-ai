"""Refinement pass support for LLMAnnotator.

Provides:
- ProcessingKey: identifies an output file by page and optional unit index
- RefinementContextProvider: injects prior extraction results into prompts
- PassValidationReport: format-agnostic validation report for a pass
- compute_fingerprints: extracts format-dependent fingerprints from JSON output
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set

from marie.extract.annotators.context_provider import ContextProvider

if TYPE_CHECKING:
    from marie.extract.annotators.context_provider import ProcessingUnit
    from marie.extract.structures.unstructured_document import UnstructuredDocument


# ---------------------------------------------------------------------------
# Processing Key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessingKey:
    """Identifies a single output artifact by page number and optional unit index.

    Examples:
        ProcessingKey(1, None)  — page-level output
        ProcessingKey(1, 0)     — first unit on page 1
    """

    page_number: int
    unit_index: Optional[int] = None

    # Pattern: frame_0001.json  or  frame_0001_t0.json
    _FILENAME_RE = re.compile(r"^frame_(\d+?)(?:_t(\d+))?\.json$", re.IGNORECASE)

    @classmethod
    def from_filename(cls, filename: str) -> "ProcessingKey":
        """Parse page number and optional unit index from an output filename.

        Supported patterns:
            frame_0001.json   -> ProcessingKey(1, None)
            frame_0001_t0.json -> ProcessingKey(1, 0)
            frame_0002_t1.json -> ProcessingKey(2, 1)

        Raises:
            ValueError: If the filename does not match the expected pattern.
        """
        m = cls._FILENAME_RE.match(filename)
        if not m:
            raise ValueError(f"Cannot parse ProcessingKey from filename: {filename!r}")
        page = int(m.group(1))
        unit = int(m.group(2)) if m.group(2) is not None else None
        return cls(page_number=page, unit_index=unit)


# ---------------------------------------------------------------------------
# Refinement Context Provider
# ---------------------------------------------------------------------------


class RefinementContextProvider(ContextProvider):
    """Injects prior extraction results into prompts for refinement passes."""

    def __init__(
        self,
        previous_results: Dict[ProcessingKey, str],
        annotator_name: str,
    ):
        super().__init__(run_context=None, annotator_name=annotator_name)
        self._results = previous_results

    def get_eligible_pages(self, document: "UnstructuredDocument") -> Set[int]:
        return set(range(1, document.page_count + 1))

    def get_variables(
        self,
        document: "UnstructuredDocument",
        page_number: int,
        unit: Optional["ProcessingUnit"] = None,
    ) -> Dict[str, str]:
        key = ProcessingKey(
            page_number=page_number,
            unit_index=unit.index if unit else None,
        )
        payload = self._results.get(key, "")
        if not payload:
            return {"PREVIOUS_EXTRACTION": ""}
        return {
            "PREVIOUS_EXTRACTION": (
                "## Previous Extraction Results\n"
                "Review and refine the following extraction. "
                "Fix errors, preserve correct results, and fill missing fields.\n"
                f"{payload}"
            ),
        }


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def compute_fingerprints(data: dict) -> set[tuple]:
    """Compute format-dependent fingerprints from a parsed JSON output.

    The fingerprint set captures the *identity* of extracted elements so that
    two passes can be compared for regression without assuming a fixed schema.

    Strategy (auto-detected):
    1. ``extractions`` list with ``label`` + ``line_number``
       → ``{(label, line_number), ...}``
    2. ``extractions`` list with ``line_number`` but no ``label``
       → ``{(line_number, truncated_value), ...}``
    3. ``canonical_mapping`` dict
       → ``{(key,), ...}`` for each mapping key
    4. Fallback → ``{(key,), ...}`` for each top-level JSON key
    """
    extractions = data.get("extractions")
    if isinstance(extractions, list) and extractions:
        first = extractions[0] if isinstance(extractions[0], dict) else {}
        has_label = "label" in first
        has_line = "line_number" in first

        if has_label and has_line:
            return {
                (item.get("label", ""), int(item.get("line_number", 0)))
                for item in extractions
                if isinstance(item, dict)
            }
        if has_line:
            return {
                (
                    int(item.get("line_number", 0)),
                    str(item.get("value", ""))[:50],
                )
                for item in extractions
                if isinstance(item, dict)
            }

    canonical = data.get("canonical_mapping")
    if isinstance(canonical, dict):
        return {(k,) for k in canonical}

    # Fallback: top-level keys
    return {(k,) for k in data}


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------


@dataclass
class PassValidationReport:
    """Format-agnostic validation report for a single refinement pass."""

    json_valid: bool
    processing_keys: set[ProcessingKey] = field(default_factory=set)
    file_count: int = 0
    total_element_count: int = 0
    fingerprints_by_key: dict[ProcessingKey, set[tuple]] = field(default_factory=dict)
    total_json_size: int = 0
    errors: list[str] = field(default_factory=list)
