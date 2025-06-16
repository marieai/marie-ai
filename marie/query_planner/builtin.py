from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import QueryPlanRegistry
from marie.query_planner.model import QueryPlannersConf
from marie.query_planner.ocr_planner import PLAN_ID as EXTRACT_PLAN_ID
from marie.query_planner.ocr_planner import query_planner_extract


def register_from_module(planner_module: str) -> None:
    """
    Registers a planner from the specified module.

    :param planner_module: The name of the module to register the planner from.
    :type planner_module: str
    :return: None
    """
    return QueryPlanRegistry.register_from_module(planner_module)


def register_all_known_planners(query_planners_conf: QueryPlannersConf):
    """
    Registers all known query planners in the QueryPlanRegistry.
    Query Planners are effectivery DAGS(Directed Acyclic Graph) definitions.

    This function is responsible for registering all the available query
    planners to the QueryPlanRegistry using their specific identifiers and
    corresponding implementations. It ensures that the planners are
    available to be used in the query execution framework.

    Additionally, dynamically loads and registers query planners from
    external modules based on their identifiers, and supports loading
    planners from wheel packages with persistent directory watching.

    :param query_planners_conf: Configuration containing planner modules and wheel settings
    :return: None
    """
    logger.info("Registering all known planners")

    # Register built-in planners
    QueryPlanRegistry.register(EXTRACT_PLAN_ID, query_planner_extract)

    # Initialize from configuration with wheel support
    result = QueryPlanRegistry.initialize_from_config(query_planners_conf)

    # Log results
    logger.info(f"Planner initialization results:")
    logger.info(f"  Loaded modules: {result['loaded']}")
    logger.info(f"  Failed modules: {result['failed']}")
    logger.info(f"  Wheel results: {result['wheel_results']}")
    logger.info(f"  Total planners: {result['total_planners']}")

    # Log all discovered planners
    logger.info("Discovered all known query planners")
    planners = QueryPlanRegistry.list_planners()
    for planner in planners:
        logger.info(f" - {planner}")
