"""Shared utilities for the record-backed MatchSection pipeline.

Provides record loading and value normalization used by both the builder
and population visitors.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from marie.utils.json import load_json_file

logger = logging.getLogger(__name__)


def load_extracted_records(
    output_dir: str,
    data_source: str,
    envelope_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load extracted claim records from ``agent-output/<data_source>/``.

    Reuses the same JSON loading and envelope-detection logic that
    ``core_parsers.py`` uses for the region-builder stage.

    Args:
        output_dir: The working/output directory containing ``agent-output/``.
        data_source: Folder name inside ``agent-output/`` (e.g.
            ``"claim-extract-aggregated"``).
        envelope_key: Optional JSON key wrapping the records array
            (e.g. ``"claims"``).  When ``None`` the format is auto-detected.

    Returns:
        Flattened list of normalized record dicts across all JSON files.
        Each record is expected to contain at least ``record_uid`` and
        ``source`` keys.
    """
    # Resolve agent-output directory.  ``output_dir`` may point to a
    # subdirectory (e.g. ``working_dir/parsed-result/``) while
    # ``agent-output/`` lives at the working-dir root.  Try the given
    # path first, then walk up one level.
    output_dir_str = str(output_dir)
    data_source_dir = _resolve_data_source_dir(output_dir_str, data_source)

    if data_source_dir is None:
        logger.warning(
            f"Data source directory 'agent-output/{data_source}' not found "
            f"under '{output_dir_str}' or its parent"
        )
        return []

    json_files = sorted(f for f in os.listdir(data_source_dir) if f.endswith(".json"))
    if not json_files:
        logger.warning(f"No JSON files found in {data_source_dir}")
        return []

    all_records: List[Dict[str, Any]] = []

    for json_file in json_files:
        file_path = os.path.join(data_source_dir, json_file)
        try:
            json_data = load_json_file(file_path, safe_parse=True)
            if not json_data:
                continue
            records = _extract_records_from_json(json_data, envelope_key)
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Error loading records from {json_file}: {e}")
            continue

    logger.info(
        f"Loaded {len(all_records)} extracted records from "
        f"{len(json_files)} file(s) in {data_source}"
    )
    return all_records


def _extract_records_from_json(
    json_data: Any, envelope_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract a list of record dicts from parsed JSON data.

    Resolution order mirrors ``core_parsers._extract_records_from_json``:

    1. **Bare array** at root level.
    2. **Explicit envelope** key from config.
    3. **Auto-detect** — first dict key whose value is a list of dicts.
    4. **Single object** at root with a ``"source"`` marker.
    """
    if isinstance(json_data, list):
        return json_data

    if not isinstance(json_data, dict):
        return []

    if envelope_key and envelope_key in json_data:
        val = json_data[envelope_key]
        return val if isinstance(val, list) else [val]

    for _key, val in json_data.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val

    if "source" in json_data:
        return [json_data]

    return []


def _resolve_data_source_dir(output_dir: str, data_source: str) -> Optional[str]:
    """Locate ``agent-output/<data_source>`` starting from *output_dir*.

    Tries *output_dir* first, then its parent directory.  This handles
    the common layout where ``output_dir`` is ``working_dir/parsed-result/``
    but ``agent-output/`` sits at ``working_dir/``.
    """
    for base in (output_dir, os.path.dirname(output_dir)):
        candidate = os.path.join(base, "agent-output", data_source)
        if os.path.isdir(candidate):
            return candidate
    return None


def normalize_record_value(field_data: Any) -> Optional[str]:
    """Normalize an extracted KV value that may be a wrapped object or scalar.

    Handles both formats emitted by the LLM extraction agent::

        {"value": "012654275-01"}  →  "012654275-01"
        "012654275-01"             →  "012654275-01"
        None                       →  None
    """
    if field_data is None:
        return None
    if isinstance(field_data, dict):
        value = field_data.get("value")
        return str(value) if value is not None else None
    return str(field_data)
