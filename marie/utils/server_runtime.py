import os
from typing import Any, Dict, Union

from marie.constants import __cache_path__
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.native_handler import NativeToastHandler
from marie.messaging.psql_handler import PsqlToastHandler
from marie.messaging.rabbit_handler import RabbitMQToastHandler
from marie.messaging.sse_broker import SseBroker
from marie.messaging.sse_toast_handler import SseToastHandler
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


def setup_toast_events(toast_config: Dict[str, Any]) -> Union[SseBroker | None]:
    """
    Setup the toast events for the server notification system
    :param toast_config: The toast config
    :return: SseBroker or None
    """
    if toast_config is None or not toast_config:
        logger.warning("No toast config provided")
        return None

    native_config = toast_config["native"]
    psql_cfg = toast_config["psql"]
    rabbitmq_cfg = toast_config["rabbitmq"]
    sse_cfg = toast_config.get("sse", {})

    Toast.register(
        NativeToastHandler(os.path.join(__cache_path__, "events.json")), native=True
    )

    if psql_cfg is not None:
        if bool(psql_cfg.get("enabled", False)):
            Toast.register(PsqlToastHandler(psql_cfg), native=False)

    if rabbitmq_cfg is not None:
        if bool(rabbitmq_cfg.get("enabled", False)):
            Toast.register(RabbitMQToastHandler(rabbitmq_cfg), native=False)

    sse_broker = None
    if sse_cfg is not None:
        if bool(sse_cfg.get("enabled", True)):
            logger.info("Setting up sse broker")
            broker_cfg = sse_cfg.get("broker", {}) or {}
            sse_broker = SseBroker(
                replay_size=int(broker_cfg.get("replay_size", 200)),
                subscriber_q_max=int(broker_cfg.get("subscriber_q_max", 1024)),
                heartbeat_interval_s=float(
                    broker_cfg.get("heartbeat_interval_s", 15.0)
                ),
            )
            Toast.register(SseToastHandler(sse_cfg, broker=sse_broker), native=False)
    return sse_broker


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
