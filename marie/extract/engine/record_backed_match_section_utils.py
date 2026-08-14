"""Shared utilities for the record-backed MatchSection pipeline.

Provides record loading and value normalization used by both the builder
and population visitors.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from marie.utils.json import extract_records_from_json, load_json_file

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

    Raises:
        FileNotFoundError: If the configured data source directory does not exist.
    """
    # Resolve agent-output directory.  ``output_dir`` may point to a
    # subdirectory (e.g. ``working_dir/parsed-result/``) while
    # ``agent-output/`` lives at the working-dir root.  Try the given
    # path first, then walk up one level.
    output_dir_str = str(output_dir)
    data_source_dir = _resolve_data_source_dir(output_dir_str, data_source)

    if data_source_dir is None:
        raise FileNotFoundError(
            f"Data source directory 'agent-output/{data_source}' not found "
            f"under '{output_dir_str}' or its parent"
        )

    json_files = sorted(f for f in os.listdir(data_source_dir) if f.endswith(".json"))
    if not json_files:
        logger.info(f"No JSON files found in {data_source_dir}")
        return []

    all_records: List[Dict[str, Any]] = []

    for json_file in json_files:
        file_path = os.path.join(data_source_dir, json_file)
        json_data = load_json_file(file_path)
        if not json_data:
            continue
        records = extract_records_from_json(json_data, envelope_key)
        all_records.extend(records)

    logger.info(
        f"Loaded {len(all_records)} extracted records from "
        f"{len(json_files)} file(s) in {data_source}"
    )
    return all_records


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
