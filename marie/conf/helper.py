from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import yaml


def load_yaml(config_file):
    with open(config_file, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(f"Config read successfully : {config_file}")
        return config_data


def storage_provider_config(provider_name, config_data) -> Dict[str, str]:
    if "storage" not in config_data:
        raise Exception("Storage config not present")
    for storage_provider in config_data["storage"]:
        if "provider" in storage_provider:
            if storage_provider["provider"] == provider_name:
                return storage_provider
    raise Exception(f"Config not present for provider : {provider_name}")
