import dataclasses
import io
import json
import os.path
from typing import Any

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


def load_json_file(filename) -> Any:
    """Read JSON File"""
    if filename is not None:
        filename = os.path.expanduser(filename)

    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def deserialize_value(json_str) -> Any:
    """Deserialize a JSON string to an object."""
    data = json.load(json_str)
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
