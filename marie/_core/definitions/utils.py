import keyword
import os
import re
from glob import glob
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import yaml

import marie.check as check
from marie._core.errors import (
    DagsterInvalidDefinitionError,
    DagsterInvariantViolationError,
)
from marie._core.storage.tags import check_reserved_tags
from marie.utils.json import to_json, deserialize_value

DEFAULT_OUTPUT = "result"
DEFAULT_GROUP_NAME = "default"  # asset group_name used when none is provided
DEFAULT_IO_MANAGER_KEY = "io_manager"

DISALLOWED_NAMES = set(
    [
        "context",
        "conf",
        "config",
        "meta",
        "arg_dict",
        "dict",
        "input_arg_dict",
        "output_arg_dict",
        "int",
        "str",
        "float",
        "bool",
        "input",
        "output",
        "type",
    ]
    + list(keyword.kwlist)  # just disallow all python keywords
)

VALID_NAME_REGEX_STR = r"^[A-Za-z0-9_]+$"
VALID_NAME_REGEX = re.compile(VALID_NAME_REGEX_STR)


class NoValueSentinel:
    """Sentinel value to distinguish unset from None."""


def has_valid_name_chars(name: str) -> bool:
    return bool(VALID_NAME_REGEX.match(name))


def check_valid_name(name: str, allow_list: Optional[List[str]] = None) -> str:
    check.str_param(name, "name")

    if allow_list and name in allow_list:
        return name

    if name in DISALLOWED_NAMES:
        raise DagsterInvalidDefinitionError(
            f'"{name}" is not a valid name in Dagster. It conflicts with a Dagster or python'
            " reserved keyword."
        )

    check_valid_chars(name)

    check.invariant(is_valid_name(name))
    return name


def check_valid_chars(name: str):
    if not has_valid_name_chars(name):
        raise DagsterInvalidDefinitionError(
            f'"{name}" is not a valid name in Dagster. Names must be in regex'
            f" {VALID_NAME_REGEX_STR}."
        )


def is_valid_name(name: str) -> bool:
    check.str_param(name, "name")

    return name not in DISALLOWED_NAMES and has_valid_name_chars(name)


def _kv_str(key: object, value: object) -> str:
    return f'{key}="{repr(value)}"'


def struct_to_string(name: str, **kwargs: object) -> str:
    # Sort the kwargs to ensure consistent representations across Python versions
    props_str = ", ".join(
        [_kv_str(key, value) for key, value in sorted(kwargs.items())]
    )
    return f"{name}({props_str})"


def validate_tags(
    tags: Optional[Mapping[str, Any]], allow_reserved_tags: bool = True
) -> Mapping[str, str]:
    valid_tags: Dict[str, str] = {}
    for key, value in check.opt_mapping_param(tags, "tags", key_type=str).items():
        if not isinstance(value, str):
            valid = False
            err_reason = f'Could not JSON encode value "{value}"'
            str_val = None
            try:
                str_val = to_json(value)
                err_reason = 'JSON encoding "{json}" of value "{val}" is not equivalent to original value'.format(
                    json=str_val, val=value
                )

                valid = deserialize_value(str_val) == value
            except Exception:
                pass

            if not valid:
                raise DagsterInvalidDefinitionError(
                    'Invalid value for tag "{key}", {err_reason}. Tag values must be strings '
                    "or meet the constraint that json.loads(json.dumps(value)) == value.".format(
                        key=key, err_reason=err_reason
                    )
                )

            valid_tags[key] = str_val  # type: ignore  # (possible none)
        else:
            valid_tags[key] = value

    if not allow_reserved_tags:
        check_reserved_tags(valid_tags)

    return valid_tags


def validate_group_name(group_name: Optional[str]) -> str:
    """Ensures a string name is valid and returns a default if no name provided."""
    if group_name:
        check_valid_chars(group_name)
        return group_name
    return DEFAULT_GROUP_NAME


def config_from_files(config_files: Sequence[str]) -> Mapping[str, Any]:
    raise NotImplementedError()


def config_from_yaml_strings(yaml_strings: Sequence[str]) -> Mapping[str, Any]:
    raise NotImplementedError()


def config_from_pkg_resources(
    pkg_resource_defs: Sequence[Tuple[str, str]]
) -> Mapping[str, Any]:
    raise NotImplementedError()
