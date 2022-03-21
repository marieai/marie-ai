from abc import ABC, abstractmethod


class ResultRenderer(ABC):
    def __init__(self, config={}):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def render(self, image, result, output_filename):
        pass
