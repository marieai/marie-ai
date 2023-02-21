import urllib
from typing import Dict, Any
from typing import List, Optional, TextIO, Tuple, Union

import yaml

from marie.excepts import BadConfigSource
from marie.helper import is_yaml_filepath
from marie.jaml import JAML
from marie.jaml.helper import complete_path
from marie.utils import json


def parse_config_source(
    path: Union[str, TextIO, Dict],
    allow_stream: bool = True,
    allow_yaml_file: bool = True,
    allow_raw_yaml_content: bool = True,
    allow_dict: bool = True,
    allow_json: bool = True,
    allow_url: bool = True,
    extra_search_paths: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> Tuple[TextIO, Optional[str]]:
    """
    Check if the text or text stream is valid.

    .. # noqa: DAR401
    :param path: the multi-kind source of the configs.
    :param allow_stream: flag
    :param allow_yaml_file: flag
    :param allow_raw_yaml_content: flag
    :param allow_dict: flag
    :param allow_json: flag
    :param allow_url: flag
    :param extra_search_paths: extra paths to search for
    :param args: unused
    :param kwargs: unused
    :return: a tuple, the first element is the text stream, the second element is the file path associate to it
            if available.
    """
    import io

    if not path:
        raise BadConfigSource
    elif allow_dict and isinstance(path, dict):
        tmp = yaml.dump(
            path, stream=None, default_flow_style=False, sort_keys=False, **kwargs
        )
        return io.StringIO(tmp), None
    elif allow_stream and hasattr(path, "read"):
        # already a readable stream
        return path, None
    elif allow_yaml_file and is_yaml_filepath(path):
        comp_path = complete_path(path, extra_search_paths)
        return open(comp_path, encoding="utf8"), comp_path
    elif allow_url and urllib.parse.urlparse(path).scheme in {"http", "https"}:
        req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as fp:
            return io.StringIO(fp.read().decode("utf-8")), None
    elif allow_raw_yaml_content:
        # possible YAML content
        path = path.replace("|", "\n    with: ")
        return io.StringIO(path), None
    elif allow_json and isinstance(path, str):
        try:
            tmp = json.loads(path)
            tmp = yaml.dump(
                tmp, stream=None, default_flow_style=False, sort_keys=False, **kwargs
            )
            return io.StringIO(tmp), None
        except json.JSONDecodeError:
            raise BadConfigSource(path)
    else:
        raise BadConfigSource(
            f"{path} can not be resolved, it should be a readable stream,"
            " or a valid file path, or a supported class name."
        )


def load_yaml(
    source: Union[str, TextIO, Dict],
    substitute=True,
    context: Dict[str, Any] = None,
    **kwargs,
) -> Dict:
    """
    Load YAML configs
    :param source: the multi-kind source of the configs.
    :param substitute:
    :param context:
    :return:
    """
    import yaml

    stream, s_path = parse_config_source(source, extra_search_paths=None, **kwargs)
    with stream as fp:
        content = fp.read()
        d = yaml.safe_load(content)

    if substitute:
        d = JAML.expand_dict(d, context)

    return d


def load_yamXXXl(config_file):
    with open(config_file, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return config_data


def executor_config(config_data, executor_name):
    if "executors" not in config_data:
        return {}

    for executor in config_data["executors"]:
        if "uses" in executor:
            if executor["uses"] == executor_name:
                return executor
    return {}


def storage_provider_config(provider_name, config_data) -> Dict[str, str]:
    if "storage" not in config_data:
        raise Exception("Storage config not present")
    for storage_provider in config_data["storage"]:
        if "provider" in storage_provider:
            if storage_provider["provider"] == provider_name:
                return storage_provider
    raise Exception(f"Config not present for provider : {provider_name}")
