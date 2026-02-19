import inspect
from typing import Type, Union

from .builder_types import TemplateBuilderFn, TemplateBuilderProto
from .sig import count_required_params, has_zero_arg_init


def coerce_builder_fn(
    obj: Union[TemplateBuilderFn, TemplateBuilderProto, Type[TemplateBuilderProto]],
) -> TemplateBuilderFn:
    """Normalize to (config: OmegaConf) -> Template."""
    if inspect.isclass(obj):
        if not hasattr(obj, "build_template"):
            raise TypeError(
                f"{obj.__name__} must define .build_template(self, config)."
            )
        if not has_zero_arg_init(obj):
            raise TypeError(
                f"{obj.__name__}.__init__ must be zero-arg; register an instance or a function instead."
            )
        return lambda conf: obj().build_template(conf)  # type: ignore[misc,call-arg]

    if hasattr(obj, "build_template") and callable(getattr(obj, "build_template")):
        return lambda conf: obj.build_template(conf)  # type: ignore[misc,call-arg]

    if callable(obj):
        sig = inspect.signature(obj)
        if count_required_params(sig) != 1:
            raise TypeError(
                "Template builder function must have exactly one required parameter: (config)."
            )
        return obj  # type: ignore[return-value]

    raise TypeError(
        "Unsupported builder; expected function, instance/class with .build_template."
    )
