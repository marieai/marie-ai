import inspect


def has_zero_arg_init(cls: type) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return True
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if (
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ):
            return False
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is p.empty:
            return False
    return True


def count_required_params(sig: inspect.Signature) -> int:
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            n += 1
    return n
