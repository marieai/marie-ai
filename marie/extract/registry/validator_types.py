from typing import Callable, TypeVar

# we’ll lazy-import BaseValidator in coercer
TValidator = TypeVar("TValidator")  # class/instance/func accepted by the decorator
