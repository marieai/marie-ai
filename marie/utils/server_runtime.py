import atexit
import os
import threading
from typing import Any, Dict, Optional, Union

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


def setup_llm_tracking(llm_tracking_config: Dict[str, Any]) -> None:
    """
    Setup the LLM tracking worker for observability.

    This starts a background thread that consumes LLM events from RabbitMQ
    and writes them to ClickHouse for analytics.

    Config example:
        llm_tracking:
          enabled: true
          worker:
            enabled: true  # Start the ingestion worker

    Environment variables (can override config):
        MARIE_LLM_TRACKING_ENABLED: Enable/disable tracking
        MARIE_LLM_TRACKING_EXPORTER: console or rabbitmq
        MARIE_LLM_TRACKING_PROJECT_ID: Project identifier

    :param llm_tracking_config: The llm_tracking section from config
    """
    global _llm_tracking_worker, _llm_tracking_thread

    if llm_tracking_config is None or not llm_tracking_config:
        logger.debug("No llm_tracking config provided, skipping")
        return

    # Check if tracking is enabled
    if not strtobool(llm_tracking_config.get("enabled", False)):
        logger.info("LLM tracking is disabled")
        return

    # Set environment variables from config (if not already set)
    _apply_llm_tracking_env(llm_tracking_config)

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


def _apply_llm_tracking_env(config: Dict[str, Any]) -> None:
    """
    Apply LLM tracking config to environment variables.

    Config values are only applied if the env var is not already set.
    """
    env_mappings = {
        "enabled": "MARIE_LLM_TRACKING_ENABLED",
        "exporter": "MARIE_LLM_TRACKING_EXPORTER",
        "project_id": "MARIE_LLM_TRACKING_PROJECT_ID",
        "debug": "MARIE_LLM_TRACKING_DEBUG",
        "sampling_rate": "MARIE_LLM_TRACKING_SAMPLING_RATE",
    }

    for config_key, env_var in env_mappings.items():
        if config_key in config and env_var not in os.environ:
            os.environ[env_var] = str(config[config_key])

    # Nested configs
    if "rabbitmq" in config:
        rabbitmq = config["rabbitmq"]
        if "url" in rabbitmq and "MARIE_LLM_TRACKING_RABBITMQ_URL" not in os.environ:
            os.environ["MARIE_LLM_TRACKING_RABBITMQ_URL"] = rabbitmq["url"]
        if (
            "exchange" in rabbitmq
            and "MARIE_LLM_TRACKING_RABBITMQ_EXCHANGE" not in os.environ
        ):
            os.environ["MARIE_LLM_TRACKING_RABBITMQ_EXCHANGE"] = rabbitmq["exchange"]
        if (
            "queue" in rabbitmq
            and "MARIE_LLM_TRACKING_RABBITMQ_QUEUE" not in os.environ
        ):
            os.environ["MARIE_LLM_TRACKING_RABBITMQ_QUEUE"] = rabbitmq["queue"]

    if "clickhouse" in config:
        clickhouse = config["clickhouse"]
        if (
            "host" in clickhouse
            and "MARIE_LLM_TRACKING_CLICKHOUSE_HOST" not in os.environ
        ):
            os.environ["MARIE_LLM_TRACKING_CLICKHOUSE_HOST"] = clickhouse["host"]
        if (
            "port" in clickhouse
            and "MARIE_LLM_TRACKING_CLICKHOUSE_PORT" not in os.environ
        ):
            os.environ["MARIE_LLM_TRACKING_CLICKHOUSE_PORT"] = str(clickhouse["port"])
        if (
            "database" in clickhouse
            and "MARIE_LLM_TRACKING_CLICKHOUSE_DATABASE" not in os.environ
        ):
            os.environ["MARIE_LLM_TRACKING_CLICKHOUSE_DATABASE"] = clickhouse[
                "database"
            ]

    if "postgres" in config:
        postgres = config["postgres"]
        if "url" in postgres and "MARIE_LLM_TRACKING_POSTGRES_URL" not in os.environ:
            os.environ["MARIE_LLM_TRACKING_POSTGRES_URL"] = postgres["url"]

    if "s3" in config:
        s3 = config["s3"]
        if "bucket" in s3 and "MARIE_LLM_TRACKING_S3_BUCKET" not in os.environ:
            os.environ["MARIE_LLM_TRACKING_S3_BUCKET"] = s3["bucket"]
        if "endpoint" in s3 and "MARIE_LLM_TRACKING_S3_ENDPOINT" not in os.environ:
            os.environ["MARIE_LLM_TRACKING_S3_ENDPOINT"] = s3["endpoint"]


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
