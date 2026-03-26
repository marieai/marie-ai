import dataclasses
import io
import json
import logging
import os.path
from typing import Any, Dict, List, Optional

from marie.numpyencoder import NumpyEncoder


class EnhancedJSONEncoder(NumpyEncoder):
    """Enhanced JSON Encoder for dataclasses and numpy arrays"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def store_json_object(results, json_path) -> None:
    """Store JSON object"""
    with open(os.path.expanduser(json_path), "w") as json_file:
        json.dump(
            results,
            json_file,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=True,
            indent=2,
            cls=EnhancedJSONEncoder,
        )


def load_json_file(filename, safe_parse: bool = False) -> Any:
    """Read JSON File"""
    if filename is not None:
        filename = os.path.expanduser(filename)

    with io.open(filename, "r", encoding="utf-8") as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError as e:
            if safe_parse:
                logging.warning(
                    f"Failed to parse JSON file {filename}. Returning None. Error: {e}"
                )
                return None
            else:
                raise e
        return data


def deserialize_value(json_str) -> Any:
    """Deserialize a JSON string to an object."""
    if json_str is None:
        return None
    if isinstance(json_str, dict):
        return json_str
    data = json.loads(json_str)
    return data


def to_json(results, **json_kwargs) -> str:
    """Convert object to a JSON object"""
    try:
        return json.dumps(
            results,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=True,
            indent=2,
            cls=EnhancedJSONEncoder,
            **json_kwargs,
        )
    except TypeError as e:
        raise TypeError(
            f"Object of type {type(results)} with value of {str(results)} is not JSON serializable"
        ) from e


def extract_records_from_json(
    json_data: Any, envelope_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract a list of record dicts from parsed JSON data.

    Domain-agnostic envelope detection with the following resolution order:

    1. **Bare array** at root level.
    2. **Explicit envelope** key from caller/config.
    3. **Auto-detect** — first dict key whose value is a list of dicts.
    4. **Single object** at root with a ``"source"`` marker.

    Args:
        json_data: Parsed JSON (list or dict).
        envelope_key: Optional key name for the records array
            (e.g. ``"wrapper"``, ``"extraction"``).  When provided the
            function looks for this key first before falling back to
            auto-detection.

    Returns:
        List of record dicts, or empty list if nothing matched.
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
