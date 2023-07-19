import io
import os
import json
import logging

__tmp_path__ = "/tmp/marie"


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def ensure_exists(dir_to_validate) -> str:
    """Ensure directory exists"""
    if not os.path.exists(dir_to_validate):
        os.makedirs(dir_to_validate, exist_ok=True)
    return dir_to_validate
