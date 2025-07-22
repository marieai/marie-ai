import os
import sys
import time
import traceback
from typing import Any, Optional, Union

import yaml


def ensure_exists(
    dir_to_validate, validate_dir_is_empty: Optional[bool] = False
) -> str:
    """Ensure directory exists and is empty if required.
    :param dir_to_validate: Directory to validate.
    :param validate_dir_is_empty: If True, the directory must be empty.
    :return: Directory to validate.
    """
    if not os.path.exists(dir_to_validate):
        os.makedirs(dir_to_validate, exist_ok=True)

    if validate_dir_is_empty and os.path.exists(dir_to_validate):
        if len(os.listdir(dir_to_validate)) > 0:
            raise ValueError(f"Directory {dir_to_validate} is not empty.")

    return dir_to_validate


def current_milli_time():
    """Get current time in milliseconds"""
    return round(time.time() * 1000)


def batchify(iterable, batch_size=1):
    """Batchify iterable"""
    size = len(iterable)
    for ndx in range(0, size, batch_size):
        yield iterable[ndx : min(ndx + batch_size, size)]


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def filter_node(node: Any, filters: Union[list, tuple]) -> None:
    """Filter node by removing keys from a dictionary or list of dictionaries"""
    if isinstance(node, (list, tuple)):
        for v in node:
            filter_node(v, filters)
    elif isinstance(node, dict):
        for flt in filters:
            try:
                del node[flt]
            except KeyError:
                pass
        for _, value in node.items():
            filter_node(value, filters)
    else:
        pass


class FileSystem:
    @staticmethod
    def __get_base_dir():
        """At most all application packages are just one level deep"""
        current_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.abspath(os.path.join(current_path, "../.."))

    @staticmethod
    def __get_config_directory() -> str:
        base_dir = FileSystem.__get_base_dir()
        return os.path.join(base_dir, "config")

    @staticmethod
    def get_plugins_directory() -> str:
        base_dir = FileSystem.__get_base_dir()
        return os.path.join(base_dir, "plugins")

    @staticmethod
    def get_share_directory() -> str:
        base_dir = os.environ.get("MARIE_DEFAULT_SHARE_PATH")
        if base_dir:
            return base_dir

        base_dir = FileSystem.__get_base_dir()
        return os.path.abspath(os.path.join(base_dir, "share"))

    @staticmethod
    def load_configuration(
        name: str = "marie.yaml", config_directory: Optional[str] = None
    ) -> dict:
        if config_directory is None:
            config_directory = FileSystem.__get_config_directory()
        with open(os.path.join(config_directory, name)) as file:
            input_data = yaml.safe_load(file)
        return input_data
