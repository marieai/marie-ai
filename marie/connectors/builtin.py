from marie.connectors.model import ConnectorsConf
from marie.connectors.registry import ConnectorRegistry
from marie.logging_core.predefined import default_logger as logger


def register_all_known_connectors(connectors_conf: ConnectorsConf) -> None:
    """Register all known connectors. Called at scheduler startup."""
    logger.info("Registering all known connectors")
    result = ConnectorRegistry.initialize_from_config(connectors_conf)

    logger.info(f"Connector initialization results:")
    logger.info(f"  Loaded: {result['loaded']}")
    if result["failed"]:
        logger.warning(f"  Failed: {result['failed']}")
    logger.info(f"  Total connectors: {result['total']}")
