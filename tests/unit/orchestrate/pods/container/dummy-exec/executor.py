import os

from marie.runtime import Executor, requests


class DummyExec(Executor):
    @requests
    def foo(self, **kwargs):
        pass
