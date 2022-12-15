def list_all_runtimes():
    """List all public runtimes that can be used directly with :class:`jina.orchestrate.pods.Pod`

    # noqa: DAR101
    # noqa: DAR201
    """
    from marie.serve.runtimes.base import BaseRuntime
    from marie.serve.runtimes.worker import WorkerRuntime

    return [
        k
        for k, s in locals().items()
        if isinstance(s, type) and issubclass(s, BaseRuntime)
    ]


def get_runtime(name: str):
    """Get a public runtime by its name

    # noqa: DAR101
    # noqa: DAR201
    """
    from marie.serve.runtimes.base import BaseRuntime
    from marie.serve.runtimes.gateway import GatewayRuntime
    from marie.serve.runtimes.head import HeadRuntime
    from marie.serve.runtimes.worker import WorkerRuntime

    s = locals()[name]
    if isinstance(s, type) and issubclass(s, BaseRuntime):
        return s
    else:
        raise TypeError(f'{s!r} is not in type {BaseRuntime!r}')
