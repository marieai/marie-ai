import os
from typing import Any, Dict

from marie.constants import __cache_path__
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.native_handler import NativeToastHandler
from marie.messaging.psql_handler import PsqlToastHandler
from marie.messaging.rabbit_handler import RabbitMQToastHandler
from marie.messaging.toast_registry import Toast
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.types import strtobool


def setup_auth(auth_config: Dict[str, Any]) -> None:
    """Set up the auth handler"""
    from marie.auth.api_key_manager import APIKeyManager

    if auth_config is None or not auth_config:
        logger.warning("No auth config provided")
        return

    APIKeyManager.from_config(auth_config)


def setup_toast_events(toast_config: Dict[str, Any]):
    """
    Setup the toast events for the server notification system
    :param toast_config: The toast config
    """
    if toast_config is None or not toast_config:
        logger.warning("No toast config provided")
        return

    native_config = toast_config["native"]
    psql_config = toast_config["psql"]
    rabbitmq_config = toast_config["rabbitmq"]

    Toast.register(
        NativeToastHandler(os.path.join(__cache_path__, "events.json")), native=True
    )

    if psql_config is not None:
        Toast.register(PsqlToastHandler(psql_config), native=False)

    if rabbitmq_config is not None:
        Toast.register(RabbitMQToastHandler(rabbitmq_config), native=False)


def setup_storage(storage_config: Dict[str, Any]) -> None:
    """Setup the storage handler"""

    if storage_config is None or not storage_config:
        logger.warning("No storage config provided")
        return

    if "s3" in storage_config and strtobool(storage_config["s3"]["enabled"]):
        logger.info("Setting up storage handler for S3")
        handler = S3StorageHandler(config=storage_config["s3"], prefix="S3_")
        StorageManager.register_handler(handler=handler)
        StorageManager.ensure_connection("s3://", silence_exceptions=False)

        StorageManager.mkdir("s3://marie")
