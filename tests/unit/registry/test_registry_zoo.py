from marie.common.file_io import PathManager
from marie.common.volume_handler import VolumeHandler

PathManager.register_handler(VolumeHandler(volume_base_dir=""))

from marie.registry.model_registry import ModelRegistryHandler
from marie.registry.model_registry import ModelRegistry

PathManager.register_handler(VolumeHandler(volume_base_dir=""))


def test_local_registry():

    ModelRegistry.discover()

    # ModelRegistry.register_handler()

    print(1)

    assert 1 == 1
