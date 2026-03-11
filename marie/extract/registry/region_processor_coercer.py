import inspect
from typing import Type, Union

from .region_processor_types import RegionProcessorFn, RegionProcessorProto
from .sig import count_required_params, has_zero_arg_init


def coerce_region_processor_fn(
    obj: Union[RegionProcessorFn, RegionProcessorProto, Type[RegionProcessorProto]],
) -> RegionProcessorFn:
    """Normalize to (context, parent, section, region_parser_config, regions_config) -> List[Dict]."""
    if inspect.isclass(obj):
        if not hasattr(obj, "process"):
            raise TypeError(
                f"{obj.__name__} must define .process(self, context, parent, section, region_parser_config, regions_config)."
            )
        if not has_zero_arg_init(obj):
            raise TypeError(
                f"{obj.__name__}.__init__ must be zero-arg; register an instance or a function instead."
            )
        return lambda context, parent, section, region_parser_config, regions_config: obj().process(
            context, parent, section, region_parser_config, regions_config
        )  # type: ignore[misc,call-arg]

    if hasattr(obj, "process") and callable(getattr(obj, "process")):
        return lambda context, parent, section, region_parser_config, regions_config: obj.process(
            context, parent, section, region_parser_config, regions_config
        )  # type: ignore[misc,call-arg]

    if callable(obj):
        sig = inspect.signature(obj)
        if count_required_params(sig) != 5:
            raise TypeError(
                "Region processor function must have exactly five required parameters: "
                "(context, parent, section, region_parser_config, regions_config)."
            )
        return obj  # type: ignore[return-value]

    raise TypeError(
        "Unsupported region processor; expected function, instance/class with .process."
    )
