import contextlib
import inspect
import os
import warnings
from types import SimpleNamespace
from typing import Optional, Dict

from marie import env_var_regex, __default_endpoint__
from marie.helper import typename, iscoroutinefunction
from marie.importer import ImportExtensions
from marie.jaml import JAMLCompatible
from marie.serve.executors.decorators import wrap_func, store_init_kwargs


class ExecutorType(type(JAMLCompatible), type):
    def __new__(cls, *args, **kwargs):
        _cls = super().__new__(cls, *args, **kwargs)
        return cls.register_class(_cls)

    @staticmethod
    def register_class(cls):
        reg_cls_set = getattr(cls, "_registered_class", set())
        cls_id = f"{cls.__module__}.{cls.__name__}"

        if cls_id not in reg_cls_set:
            arg_spec = inspect.getfullargspec(cls.__init__)
            if not arg_spec.varkw:
                raise TypeError(
                    f"{cls.__init__} does not follow the full signature of `Executor.__init__`, "
                    f"please add `**kwargs` to your __init__ function"
                )

            wrap_func(cls, ["__init__"], store_init_kwargs)
            reg_cls_set.add(cls_id)
            setattr(cls, "_registered_class", reg_cls_set)
        return cls


class BaseExecutor(JAMLCompatible, metaclass=ExecutorType):
    """
    The base class of the executor, can be used to build encoder, indexer, etc.

    .. highlight:: python
    .. code-block:: python

        class MyAwesomeExecutor:
            def __init__(awesomeness=5):
                pass
    """

    def __init__(
        self,
        metas: Optional[Dict] = None,
        requests: Optional[Dict] = None,
        runtime_args: Optional[Dict] = None,
        **kwargs,
    ):
        """`metas` and `requests` are always auto-filled with values from YAML config.

        :param metas: a dict of metas fields
        :param requests: a dict of endpoint-function mapping
        :param runtime_args: a dict of arguments injected from :class:`Runtime` during runtime
        :param kwargs: additional extra keyword arguments to avoid failing when extra params ara passed that are not expected
        """
        self._add_metas(metas)
        self._add_requests(requests)
        self._add_runtime_args(runtime_args)
        self._init_monitoring()

    def _add_runtime_args(self, _runtime_args: Optional[Dict]):
        if _runtime_args:
            self.runtime_args = SimpleNamespace(**_runtime_args)
        else:
            self.runtime_args = SimpleNamespace()

    def _init_monitoring(self):
        if hasattr(self.runtime_args, "metrics_registry") and self.runtime_args.metrics_registry:
            with ImportExtensions(
                required=True,
                help_text="You need to install the `prometheus_client` to use the montitoring functionality of Marie",
            ):
                from prometheus_client import Summary

            self._summary_method = Summary(
                "process_request_seconds",
                "Time spent when calling the executor request method",
                registry=self.runtime_args.metrics_registry,
                namespace="marie",
                labelnames=("executor", "executor_endpoint", "runtime_name"),
            )
            self._metrics_buffer = {"process_request_seconds": self._summary_method}

        else:
            self._summary_method = None
            self._metrics_buffer = None

    def _add_requests(self, _requests: Optional[Dict]):
        if not hasattr(self, "requests"):
            self.requests = {}

        if _requests:
            func_names = {f.__name__: e for e, f in self.requests.items()}
            for endpoint, func in _requests.items():
                # the following line must be `getattr(self.__class__, func)` NOT `getattr(self, func)`
                # this to ensure we always have `_func` as unbound method
                if func in func_names:
                    del self.requests[func_names[func]]

                _func = getattr(self.__class__, func)
                if callable(_func):
                    # the target function is not decorated with `@requests` yet
                    self.requests[endpoint] = _func
                elif typename(_func) == "jina.executors.decorators.FunctionMapper":
                    # the target function is already decorated with `@requests`, need unwrap with `.fn`
                    self.requests[endpoint] = _func.fn
                else:
                    raise TypeError(
                        f"expect {typename(self)}.{func} to be a function, but receiving {typename(_func)}"
                    )

    def _add_metas(self, _metas: Optional[Dict]):
        from marie.serve.executors.metas import get_default_metas

        tmp = get_default_metas()

        if _metas:
            tmp.update(_metas)

        unresolved_attr = False
        target = SimpleNamespace()
        # set self values filtered by those non-exist, and non-expandable
        for k, v in tmp.items():
            if not hasattr(target, k):
                if isinstance(v, str):
                    if not env_var_regex.findall(v):
                        setattr(target, k, v)
                    else:
                        unresolved_attr = True
                else:
                    setattr(target, k, v)
            elif type(getattr(target, k)) == type(v):
                setattr(target, k, v)

        # `name` is important as it serves as an identifier of the executor
        # if not given, then set a name by the rule
        if not getattr(target, "name", None):
            setattr(target, "name", self.__class__.__name__)

        self.metas = target

    def close(self) -> None:
        """
        Always invoked as executor is destroyed.

        You can write destructor & saving logic here.
        """
        pass

    def __call__(self, req_endpoint: str, **kwargs):
        """
        # noqa: DAR101
        # noqa: DAR102
        # noqa: DAR201
        """
        if req_endpoint in self.requests:
            return self.requests[req_endpoint](self, **kwargs)  # unbound method, self is required
        elif __default_endpoint__ in self.requests:
            return self.requests[__default_endpoint__](self, **kwargs)  # unbound method, self is required

    async def __acall__(self, req_endpoint: str, **kwargs):
        """
        # noqa: DAR101
        # noqa: DAR102
        # noqa: DAR201
        """
        if req_endpoint in self.requests:
            return await self.__acall_endpoint__(req_endpoint, **kwargs)
        elif __default_endpoint__ in self.requests:
            return await self.__acall_endpoint__(__default_endpoint__, **kwargs)

    async def __acall_endpoint__(self, req_endpoint, **kwargs):
        func = self.requests[req_endpoint]

        runtime_name = self.runtime_args.name if hasattr(self.runtime_args, "name") else None

        _summary = (
            self._summary_method.labels(self.__class__.__name__, req_endpoint, runtime_name).time()
            if self._summary_method
            else contextlib.nullcontext()
        )

        with _summary:
            if iscoroutinefunction(func):
                return await func(self, **kwargs)
            else:
                return func(self, **kwargs)

    @property
    def workspace(self) -> Optional[str]:
        """
        Get the workspace directory of the Executor.

        :return: returns the workspace of the current shard of this Executor.
        """
        workspace = (
            getattr(self.runtime_args, "workspace", None)
            or getattr(self.metas, "workspace")
            or os.environ.get("MARIE_DEFAULT_WORKSPACE_BASE")
        )
        if workspace:
            complete_workspace = os.path.join(workspace, self.metas.name)
            shard_id = getattr(
                self.runtime_args,
                "shard_id",
                None,
            )
            if shard_id is not None and shard_id != -1:
                complete_workspace = os.path.join(complete_workspace, str(shard_id))
            if not os.path.exists(complete_workspace):
                os.makedirs(complete_workspace)
            return os.path.abspath(complete_workspace)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_metrics(self, name: Optional[str] = None, documentation: Optional[str] = None) -> Optional["Summary"]:
        """
        Get a given prometheus metric, if it does not exist yet, it will create it and store it in a buffer.
        :param name: the name of the metrics
        :param documentation:  the description of the metrics

        :return: the given prometheus metrics or None if monitoring is not enable.
        """

        if self._metrics_buffer:
            if name not in self._metrics_buffer:
                from prometheus_client import Summary

                self._metrics_buffer[name] = Summary(
                    name,
                    documentation,
                    registry=self.runtime_args.metrics_registry,
                    namespace="marie",
                    labelnames=("runtime_name",),
                ).labels(self.runtime_args.name)
            return self._metrics_buffer[name]
        else:
            return None
