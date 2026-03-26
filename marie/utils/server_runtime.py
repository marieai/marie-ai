import asyncio
import atexit
import os
import threading
from typing import Any, Dict, Optional, Union

from marie.constants import __cache_path__
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.messaging.grpc_toast_handler import GrpcToastHandler
from marie.messaging.native_handler import NativeToastHandler
from marie.messaging.otel_toast_handler import OTELToastHandler
from marie.messaging.psql_handler import PsqlToastHandler
from marie.messaging.rabbit_handler import RabbitMQToastHandler
from marie.messaging.toast_registry import Toast
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.types import strtobool

# Global reference to sensor worker for cleanup
_sensor_worker: Optional[Any] = None
_sensor_worker_thread: Optional[threading.Thread] = None


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

    otel_cfg = toast_config.get("otel")
    if otel_cfg is not None:
        if bool(otel_cfg.get("enabled", False)):
            logger.info("Setting up OTEL toast handler")
            Toast.register(OTELToastHandler(otel_cfg), native=False)

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
    Setup LLM tracking instrumentation.

    Configures InstrumentationSettings from YAML and ensures a global
    OI TracerProvider exists. No background worker -- OTel's
    BatchSpanProcessor handles async export internally.
    """
    if not llm_tracking_config or not strtobool(
        llm_tracking_config.get("enabled", False)
    ):
        logger.debug("LLM tracking is disabled or not configured")
        return

    from marie.instrumentation.config import ExporterType, configure_from_yaml

    settings = configure_from_yaml(llm_tracking_config, storage_config)

    if settings.EXPORTER == ExporterType.OTEL:
        _ensure_tracer_provider(
            require_openinference=True,
            console_export=settings.CONSOLE_SPANS,
        )
        logger.info("LLM tracking configured with OTel exporter")
    else:
        _ensure_tracer_provider(console_export=settings.CONSOLE_SPANS)
        logger.info("LLM tracking configured with console exporter")


def _ensure_tracer_provider(
    require_openinference: bool = False,
    console_export: bool = False,
) -> None:
    """Ensure a compatible global OI TracerProvider exists."""
    from opentelemetry import trace

    current = trace.get_tracer_provider()
    is_proxy = type(current).__name__ == "ProxyTracerProvider"

    if is_proxy:
        from marie.instrumentation import register

        register(console_export=console_export)
        logger.info("Created global OI TracerProvider for LLM tracking")
        return

    if require_openinference:
        tracer = trace.get_tracer("marie.instrumentation.bootstrap")
        is_openinference = all(
            hasattr(tracer, attr) for attr in ("agent", "chain", "tool", "llm")
        )
        if not is_openinference:
            raise RuntimeError(
                "Global TracerProvider already exists but does not expose "
                "OpenInference tracer capabilities required by exporter=otel."
            )


def setup_sensor_worker(
    sensor_config: Dict[str, Any],
    db_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Setup the SensorWorker in a dedicated background thread."""
    global _sensor_worker, _sensor_worker_thread

    if sensor_config is None or not sensor_config:
        return

    if not strtobool(sensor_config.get("enabled", False)):
        logger.info("SensorWorker is disabled")
        return

    try:
        from marie.sensors.daemon.worker import SensorWorker

        logger.info("Starting SensorWorker...")

        _sensor_worker = SensorWorker(config=sensor_config)

        _sensor_worker_thread = threading.Thread(
            target=_run_sensor_worker,
            args=(db_config,),
            name="sensor-worker",
            daemon=True,
        )
        _sensor_worker_thread.start()

        atexit.register(_stop_sensor_worker)
        logger.info("SensorWorker started in background thread")

    except ImportError as e:
        logger.warning(f"SensorWorker dependencies not available: {e}")
    except Exception as e:
        logger.error(f"Failed to start SensorWorker: {e}")


def _run_sensor_worker(db_config: Optional[Dict[str, Any]] = None) -> None:
    """Run the SensorWorker in its own event loop."""

    if _sensor_worker is None:
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_init_sensor_storage(db_config))
        loop.run_until_complete(_sensor_worker.start())

        if _sensor_worker._daemon_task is not None:
            loop.run_until_complete(_sensor_worker._daemon_task)
    except Exception as e:
        logger.error(f"SensorWorker failed: {e}")
    finally:
        loop.close()


async def _init_sensor_storage(db_config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize PostgreSQL storage for the sensor worker."""
    if _sensor_worker is None or db_config is None:
        return

    try:
        from marie.sensors.state.psql_storage import PostgreSQLSensorStorage

        storage = PostgreSQLSensorStorage(db_config)
        await storage.initialize()
        _sensor_worker.set_storage(storage)
        logger.info("SensorWorker storage initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SensorWorker storage: {e}")


def _stop_sensor_worker() -> None:
    """Stop the SensorWorker."""
    global _sensor_worker, _sensor_worker_thread

    if _sensor_worker is not None:
        logger.info("Stopping SensorWorker...")
        try:
            _sensor_worker._shutdown_event.set()
        except Exception as e:
            logger.warning(f"Error signaling SensorWorker shutdown: {e}")

    if _sensor_worker_thread is not None and _sensor_worker_thread.is_alive():
        _sensor_worker_thread.join(timeout=10.0)

    _sensor_worker = None
    _sensor_worker_thread = None


def stop_sensor_worker() -> None:
    """Public function to stop the SensorWorker."""
    _stop_sensor_worker()
