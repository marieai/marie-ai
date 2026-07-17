from typing import Union

from . import EngineLM, get_engine


def validate_engine_or_get_default(engine: Union[EngineLM, str, None]) -> EngineLM:
    """
    Validate the engine or get the default engine.
    :param engine:  The engine to use for the LLM call. If None, the default engine will be used.
    :return: The engine to use for the LLM call.
    """
    if engine is None:
        raise Exception(
            "No engine provided. Please provide an engine or use the default engine."
        )
    if isinstance(engine, str):
        engine = get_engine(engine)
    return engine
