# based on : jina
"""Decorators and wrappers designed for wrapping :class:`BaseExecutor` functions. """
import inspect
from functools import wraps
from typing import Callable

from marie.helper import convert_tuple_to_list
from marie.serve.executors.metas import get_default_metas


def wrap_func(cls, func_lst, wrapper):
    """Wrapping a class method only once, inherited but not overridden method will not be wrapped again

    :param cls: class
    :param func_lst: function list to wrap
    :param wrapper: the wrapper
    """
    for f_name in func_lst:
        if hasattr(cls, f_name) and all(getattr(cls, f_name) != getattr(i, f_name, None) for i in cls.mro()[1:]):
            setattr(cls, f_name, wrapper(getattr(cls, f_name)))


def store_init_kwargs(func: Callable) -> Callable:
    """Mark the args and kwargs of :func:`__init__` later to be stored via :func:`save_config` in YAML
    :param func: the function to decorate
    :return: the wrapped function
    """

    @wraps(func)
    def arg_wrapper(self, *args, **kwargs):
        if func.__name__ != "__init__":
            raise TypeError("this decorator should only be used on __init__ method of an executor")
        taboo = {"self", "args", "kwargs", "metas", "requests", "runtime_args"}
        _defaults = get_default_metas()
        taboo.update(_defaults.keys())
        all_pars = inspect.signature(func).parameters
        tmp = {k: v.default for k, v in all_pars.items() if k not in taboo}
        tmp_list = [k for k in all_pars.keys() if k not in taboo]
        # set args by aligning tmp_list with arg values
        for k, v in zip(tmp_list, args):
            tmp[k] = v
        # set kwargs
        for k, v in kwargs.items():
            if k in tmp:
                tmp[k] = v

        if hasattr(self, "_init_kwargs_dict"):
            self._init_kwargs_dict.update(tmp)
        else:
            self._init_kwargs_dict = tmp
        convert_tuple_to_list(self._init_kwargs_dict)
        f = func(self, *args, **kwargs)
        return f

    return arg_wrapper


def requests():
    pass


def monitor():
    pass
