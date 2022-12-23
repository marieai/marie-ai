import numpy as np
import torch

from marie import DocumentArray, Executor, requests
from marie.logging.predefined import default_logger


class MyExecutorAA(Executor):
    @requests(on="/extract")
    def foo(self, docs: DocumentArray, **kwargs):
        default_logger.info("Executing extract")
        default_logger.info(kwargs)
        docs[0].text = 'AA - hello, world!'
        docs[1].text = 'AA - goodbye, world!'

    @requests(on='/crunch-numbers-aa')
    def bar(self, docs: DocumentArray, **kwargs):
        docs[0].text = 'crunch, aa'
