from typing import Any, Dict, List, Union

from marie.extract.registry import component_registry


def initialize_components_from_config(config: Union[Dict[str, Any], List[str]]):
    """
    Initialize components from configuration.

    Args:
        config: Either a dict with configuration or a list of module names

    Returns:
        Dict with loading results
    """
    if isinstance(config, list):
        config = {'external_parser_modules': config}

    return component_registry.initialize_from_config(config)


def load_external_parsers(module_names: List[str]):
    """Load external parser modules by name"""
    return component_registry.initialize_external_components(module_names)
