from typing import Dict


class RegistryHolder(type):

    REGISTRY: Dict[str, "RegistryHolder"] = {}

    def __new__(cls, name, bases, attrs):
        # print(f'****** : {name}')
        new_cls = type.__new__(cls, name, bases, attrs)
        """
            Here the name of the class is used as key but it could be any class
            parameter.
        """
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
