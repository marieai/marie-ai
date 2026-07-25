import subprocess
import sys


def test_gateway_imports_without_imaging_or_ml_dependencies() -> None:
    code = """
import builtins
import sys

blocked = {
    "albumentations",
    "cv2",
    "imagecodecs",
    "matplotlib",
    "networkx",
    "pandas",
    "pdf2image",
    "PIL",
    "pyarrow",
    "skimage",
    "torch",
    "torchvision",
    "wand",
}
original_import = builtins.__import__

def import_without_imaging(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_imaging

from marie.serve.runtimes.gateway.marie import MarieGateway
from marie.api.routes import create_mcp_router

assert MarieGateway.__name__ == "MarieGateway"
assert callable(create_mcp_router)
assert "marie.agent" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
