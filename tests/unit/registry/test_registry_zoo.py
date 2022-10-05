import json
import os.path

import pytest

from marie.registry.model_registry import ModelRegistry


def create_model_assets(model_zoo_dir, model_name):
    data = {"_name_or_path": model_name}
    os.makedirs(os.path.join(model_zoo_dir, model_name), exist_ok=True)

    with open(os.path.join(model_zoo_dir, model_name, "marie.json"), "w") as json_file:
        json.dump(
            data,
            json_file,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=True,
            indent=2,
        )


@pytest.fixture(scope="function")
def model_zoo_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("model_zoo")
    return str(fn)


@pytest.mark.parametrize(
    "test_input",
    [["marie/model-001"], ["marie/model-002", "marie/model-003"]],
)
def test_model_discovery(test_input, model_zoo_dir):
    for model_name in test_input:
        create_model_assets(model_zoo_dir, model_name)
    __model_path__ = model_zoo_dir
    kwargs = {"__model_path__": __model_path__}
    resolved = ModelRegistry.discover(**kwargs)
    assert len(resolved) == len(test_input)


@pytest.mark.parametrize(
    "test_input",
    ["marie/model-001", "marie/model-002", "marie/model-003"],
)
def test_get_local_path_from_name(test_input, model_zoo_dir):
    kwargs = {"__model_path__": model_zoo_dir}
    model_dir = os.path.join(model_zoo_dir, test_input)
    create_model_assets(model_zoo_dir, test_input)

    config = ModelRegistry.get_local_path(test_input, **kwargs)

    assert config == model_dir


@pytest.mark.parametrize(
    "test_input",
    ["marie/model-001", "marie/model-002", "marie/model-003"],
)
def test_get_local_path_from_path(test_input, model_zoo_dir):
    model_dir = os.path.join(model_zoo_dir, test_input)
    create_model_assets(model_zoo_dir, test_input)

    kwargs = {}
    config = ModelRegistry.get_local_path(model_dir, **kwargs)

    assert config == model_dir
