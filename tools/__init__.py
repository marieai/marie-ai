import io
import json
import os
import shutil

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


def file_exists(path):
    """check if file is already downloaded"""
    return os.path.exists(path) and os.path.getsize(path) > 0


def copyFiles(src_file_paths, dest_path):
    """Copies a list of files to a destination directory."""
    os.makedirs(dest_path, exist_ok=True)
    for file_path in src_file_paths:
        shutil.copyfile(file_path, os.path.join(dest_path, os.path.split(file_path)[1]))
