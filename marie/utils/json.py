import io
import json
from typing import Any

from marie.numpyencoder import NumpyEncoder


def store_json_object(results, json_path) -> None:
    """Store JSON object"""
    with open(json_path, "w") as json_file:
        json.dump(
            results,
            json_file,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=True,
            indent=2,
            cls=NumpyEncoder,
        )


def load_json_file(filename) -> Any:
    """Read JSON File"""
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def to_json(results) -> str:
    """Convert object to a JSON object"""
    return json.dumps(
        results,
        sort_keys=False,
        separators=(",", ": "),
        ensure_ascii=True,
        indent=2,
        cls=NumpyEncoder,
    )
