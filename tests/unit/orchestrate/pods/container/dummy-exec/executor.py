import os

from marie import Executor, requests


class DummyExec(Executor):
    @requests
    def foo(self, **kwargs):
        pass
