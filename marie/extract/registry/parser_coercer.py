import inspect
from typing import Type, Union

from .parser_types import ParserFn, ParserProto
from .sig import count_required_params, has_zero_arg_init


def coerce_parser_fn(obj: Union[ParserFn, ParserProto, Type[ParserProto]]) -> ParserFn:
    """Normalize to (doc, working_dir, src_dir, conf) -> Optional[ExtractionResult]."""
    if inspect.isclass(obj):
        if not hasattr(obj, "parse"):
            raise TypeError(
                f"{obj.__name__} must define .parse(self, doc, working_dir, src_dir, conf)."
            )
        if not has_zero_arg_init(obj):
            raise TypeError(
                f"{obj.__name__}.__init__ must be zero-arg; register an instance or a function instead."
            )
        return lambda doc, wd, sd, conf: obj().parse(doc, wd, sd, conf)  # type: ignore[misc,call-arg]

    if hasattr(obj, "parse") and callable(getattr(obj, "parse")):
        return lambda doc, wd, sd, conf: obj.parse(doc, wd, sd, conf)  # type: ignore[misc,call-arg]

    if callable(obj):
        sig = inspect.signature(obj)
        if count_required_params(sig) != 4:
            raise TypeError(
                "Parser function must have exactly four required parameters: (doc, working_dir, src_dir, conf)."
            )
        return obj  # type: ignore[return-value]

    raise TypeError(
        "Unsupported parser; expected function, instance/class with .parse."
    )
