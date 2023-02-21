""" Local and remote storage support """
from .manager import StorageManager
from .manager import PathHandler

from .native_handler import NativePathHandler
from .s3_storage import S3StorageHandler

__all__ = ["StorageManager", "S3StorageHandler", "NativePathHandler", "PathHandler"]


# Register the default handlers

StorageManager.register_handler(NativePathHandler(), native=True)
