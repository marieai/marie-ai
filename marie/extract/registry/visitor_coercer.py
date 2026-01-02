import inspect
from typing import Type, Union

from marie.extract.engine.processing_visitor import ProcessingVisitor


def coerce_visitor_instance(
    obj: Union[ProcessingVisitor, Type[ProcessingVisitor]],
) -> ProcessingVisitor:
    """
    Normalize to a ProcessingVisitor instance.

    If a class is provided, it must have a zero-argument constructor
    and will be instantiated immediately.
    """
    # Handle Classes
    if inspect.isclass(obj):
        if not issubclass(obj, ProcessingVisitor):
            raise TypeError(
                f"{obj.__name__} must inherit from marie.extract.engine.processing_visitor.ProcessingVisitor."
            )
        return obj  # type: ignore[return-value]

    # Handle Instances
    if isinstance(obj, ProcessingVisitor):
        return obj

    raise TypeError(
        f"Unsupported visitor type: {type(obj)}. Expected subclass or instance of ProcessingVisitor."
    )
