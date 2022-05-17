"""
Context object of incoming request
"""


class Context(object):
    """
    Context store relevant model information
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._metrics = None

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics
