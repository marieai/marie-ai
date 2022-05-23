import inspect
from types import SimpleNamespace
from typing import Optional, Dict

from marie.helper import typename
from marie.importer import ImportExtensions
from marie.serve.executors.decorators import wrap_func, store_init_kwargs


class ExecutorType(type):
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


class BaseExecutor(metaclass=ExecutorType):
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
        pass

    def close(self) -> None:
        """
        Always invoked as executor is destroyed.

        You can write destructor & saving logic here.
        """
        pass

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
