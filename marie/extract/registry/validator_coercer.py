import inspect
from typing import Any

from .sig import has_zero_arg_init


def coerce_validator_instance(obj: Any, name: str):
    """
    Normalize to a BaseValidator *instance*.
    Accepts: BaseValidator subclass (zero-arg), BaseValidator instance, or callable(context).
    """
    from marie.extract.validator.base import BaseValidator, FunctionValidatorWrapper

    if inspect.isclass(obj) and issubclass(obj, BaseValidator):
        if not has_zero_arg_init(obj):
            raise TypeError(
                f"{obj.__name__}.__init__ must be zero-arg; register an instance or a function instead."
            )
        inst = obj()
        inst.name = name
        return inst

    if isinstance(obj, BaseValidator):
        obj.name = name
        return obj

    if callable(obj):
        return FunctionValidatorWrapper(name, obj)

    raise TypeError(
        f"Invalid validator type: {type(obj)}. Must be BaseValidator class/instance or callable."
    )
