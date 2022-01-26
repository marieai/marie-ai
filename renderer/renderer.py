from abc import abstractmethod, ABCMeta


class ResultRenderer(metaclass=ABCMeta):

    def __init__(self):
        print("ResultRenderer base")

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def render(self, image, result):
        pass
