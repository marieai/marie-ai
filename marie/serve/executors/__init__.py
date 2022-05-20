import inspect

from marie.serve.executors.decorators import wrap_func, store_init_kwargs


class ExecutorType(type):
    def __new__(cls, *args, **kwargs):
        _cls = super().__new__(cls, *args, **kwargs)
        return cls.register_class(_cls)

    @staticmethod
    def register_class(cls):
        print(cls)
        reg_cls_set = getattr(cls, "_registered_class", set())
        cls_id = f"{cls.__module__}.{cls.__name__}"
        print(cls_id)
        print(reg_cls_set)

        if cls_id not in reg_cls_set:
            arg_spec = inspect.getfullargspec(cls.__init__)
            print(arg_spec)

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
    pass
