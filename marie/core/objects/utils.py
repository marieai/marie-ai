from typing import Any, Callable, Optional, Sequence

from marie.core.tools import BaseTool
from marie.core.objects.base import SimpleObjectNodeMapping
from marie.core.objects.base_node_mapping import BaseObjectNodeMapping
from marie.core.objects.fn_node_mapping import FnNodeMapping
from marie.core.objects.tool_node_mapping import SimpleToolNodeMapping
from marie.core.schema import BaseNode


def get_object_mapping(
    objects: Sequence[Any],
    from_node_fn: Optional[Callable[[BaseNode], Any]] = None,
    to_node_fn: Optional[Callable[[Any], BaseNode]] = None,
) -> BaseObjectNodeMapping:
    """Get object mapping according to object."""
    if from_node_fn is not None and to_node_fn is not None:
        return FnNodeMapping.from_objects(objects, from_node_fn, to_node_fn)
    elif all(isinstance(obj, BaseTool) for obj in objects):
        return SimpleToolNodeMapping.from_objects(objects)
    else:
        return SimpleObjectNodeMapping.from_objects(objects)
