import importlib
import warnings

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import QueryPlanRegistry
from marie.query_planner.ocr_planner import PLAN_ID as EXTRACT_PLAN_ID
from marie.query_planner.ocr_planner import query_planner_extract


def register_from_module(planner_module: str) -> None:
    """
    Registers a planner from the specified module.

    :param planner_module: The name of the module to register the planner from.
    :type planner_module: str
    :return: None
    """
    try:
        importlib.import_module(planner_module)
    except Exception as e:
        logger.error(f"Registering planner from {planner_module}")
        warnings.warn(
            f"Error importing {planner_module} : some configs may not be available\n\n\tRoot cause: {e}\n"
        )


def register_all_known_planners():
    """
    Registers all known query planners in the QueryPlanRegistry.

    This function is responsible for registering all the available query
    planners to the QueryPlanRegistry using their specific identifiers and
    corresponding implementations. It ensures that the planners are
    available to be used in the query execution framework.

    Additionally, dynamically loads and registers query planners from
    external modules based on their identifiers.

    :return: None
    """
    logger.info("Registering all known planners")
    QueryPlanRegistry.register(EXTRACT_PLAN_ID, query_planner_extract)

    # TODO : This needs to load from CONFIG
    planner_module = "grapnel_g5.tid_100985.query"
    register_from_module(planner_module)
