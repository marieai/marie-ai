import atexit
import os
import threading
from typing import Any, Dict, Optional, Union

from marie.constants import __cache_path__
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.messaging.grpc_toast_handler import GrpcToastHandler
from marie.messaging.native_handler import NativeToastHandler
from marie.messaging.psql_handler import PsqlToastHandler
from marie.messaging.rabbit_handler import RabbitMQToastHandler
from marie.messaging.toast_registry import Toast
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.types import strtobool

# Global reference to LLM tracking worker for cleanup
_llm_tracking_worker: Optional[Any] = None
_llm_tracking_thread: Optional[threading.Thread] = None


def setup_auth(auth_config: Dict[str, Any]) -> None:
    """Set up the auth handler"""
    from marie.auth.api_key_manager import APIKeyManager

    if auth_config is None or not auth_config:
        logger.warning("No auth config provided")
        return

    APIKeyManager.from_config(auth_config)


def setup_toast_events(toast_config: Dict[str, Any]) -> Optional[GrpcEventBroker]:
    """
    Setup the toast events for the server notification system.

    :param toast_config: The toast config
    :return: GrpcEventBroker or None
    """
    if toast_config is None or not toast_config:
        logger.warning("No toast config provided")
        return None

    native_config = toast_config.get("native", {})
    psql_cfg = toast_config.get("psql")
    rabbitmq_cfg = toast_config.get("rabbitmq")
    grpc_cfg = toast_config.get("grpc", {})

    Toast.configure(
        warn_qsize_threshold=256,  # absolute threshold wins
        warn_interval_s=3.0,  # rate-limit warnings
    )

    Toast.register(
        NativeToastHandler(os.path.join(__cache_path__, "events.json")), native=True
    )

    if psql_cfg is not None:
        if bool(psql_cfg.get("enabled", False)):
            Toast.register(PsqlToastHandler(psql_cfg), native=False)

    if rabbitmq_cfg is not None:
        if bool(rabbitmq_cfg.get("enabled", False)):
            Toast.register(RabbitMQToastHandler(rabbitmq_cfg), native=False)

    grpc_broker = None
    if grpc_cfg is not None:
        if bool(grpc_cfg.get("enabled", True)):
            logger.info("Setting up gRPC event broker")
            broker_cfg = grpc_cfg.get("broker", {}) or {}
            grpc_broker = GrpcEventBroker(
                replay_size=int(broker_cfg.get("replay_size", 200)),
                max_in_flight=int(broker_cfg.get("max_in_flight", 100)),
                ack_timeout_s=float(broker_cfg.get("ack_timeout_s", 30.0)),
                heartbeat_interval_s=float(
                    broker_cfg.get("heartbeat_interval_s", 15.0)
                ),
                redelivery_delay_s=float(broker_cfg.get("redelivery_delay_s", 5.0)),
                backpressure_threshold_pct=int(
                    broker_cfg.get("backpressure_threshold_pct", 80)
                ),
                max_redelivery_attempts=int(
                    broker_cfg.get("max_redelivery_attempts", 5)
                ),
            )
            handler = GrpcToastHandler(grpc_cfg, broker=grpc_broker)
            Toast.register(handler, native=False)

    return grpc_broker


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


def setup_llm_tracking(
    llm_tracking_config: Dict[str, Any],
    storage_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Setup the LLM tracking worker for observability.

    This starts a background thread that consumes LLM events from RabbitMQ
    and writes them to ClickHouse for analytics.

    Config example (YAML):
        llm_tracking:
          enabled: true
          exporter: rabbitmq
          project_id: marie-ai
          worker:
            enabled: true
          rabbitmq:
            <<: *rabbitmq_conf_shared
            exchange: llm_tracking
            queue: llm_events
          clickhouse:
            host: localhost
            port: 8123
            database: marie

    :param llm_tracking_config: The llm_tracking section from YAML config
    :param storage_config: Optional storage section for shared S3 config
    """
    global _llm_tracking_worker, _llm_tracking_thread

    if llm_tracking_config is None or not llm_tracking_config:
        logger.debug("No llm_tracking config provided, skipping")
        return

    # Check if tracking is enabled
    if not strtobool(llm_tracking_config.get("enabled", False)):
        logger.info("LLM tracking is disabled")
        return

    # Configure settings from YAML (single source of truth)
    from marie.llm_tracking.config import configure_from_yaml

    configure_from_yaml(llm_tracking_config, storage_config)
    logger.info("LLM tracking configured from YAML config")

    # Check if worker should be started
    worker_config = llm_tracking_config.get("worker", {})
    if not strtobool(worker_config.get("enabled", True)):
        logger.info("LLM tracking worker is disabled, only tracking is enabled")
        return

    try:
        from marie.llm_tracking.worker import LLMTrackingWorker

        logger.info("Starting LLM tracking worker...")

        # Create the worker
        _llm_tracking_worker = LLMTrackingWorker()

        # Start in a background thread
        _llm_tracking_thread = threading.Thread(
            target=_run_llm_tracking_worker,
            name="llm-tracking-worker",
            daemon=True,  # Will be stopped when main process exits
        )
        _llm_tracking_thread.start()

        # Register cleanup
        atexit.register(_stop_llm_tracking_worker)

        logger.info("LLM tracking worker started in background thread")

    except ImportError as e:
        logger.warning(f"LLM tracking dependencies not available: {e}")
    except Exception as e:
        logger.error(f"Failed to start LLM tracking worker: {e}")


def _run_llm_tracking_worker() -> None:
    """Run the LLM tracking worker (called in background thread)."""
    if _llm_tracking_worker is None:
        return

    try:
        _llm_tracking_worker.run()
    except Exception as e:
        logger.error(f"LLM tracking worker failed: {e}")


def _stop_llm_tracking_worker() -> None:
    """Stop the LLM tracking worker (called on shutdown)."""
    global _llm_tracking_worker, _llm_tracking_thread

    if _llm_tracking_worker is not None:
        logger.info("Stopping LLM tracking worker...")
        try:
            _llm_tracking_worker.stop()
        except Exception as e:
            logger.warning(f"Error stopping LLM tracking worker: {e}")
        finally:
            _llm_tracking_worker = None

    if _llm_tracking_thread is not None and _llm_tracking_thread.is_alive():
        _llm_tracking_thread.join(timeout=5.0)
        _llm_tracking_thread = None


def stop_llm_tracking() -> None:
    """
    Public function to stop the LLM tracking worker.

    Can be called manually for graceful shutdown.
    """
    _stop_llm_tracking_worker()
