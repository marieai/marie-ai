from abc import ABC, abstractmethod
from typing import List


class Function(ABC):
    """
    The class to define a function that can be called by the model.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> str:
        pass
