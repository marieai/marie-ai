import numpy as np
import torch
import marie
import marie.helper

from marie import DocumentArray, Executor, requests
from marie.logging.predefined import default_logger


class ExtractExecutor(Executor):
    @requests(on="/extract")
    def extract(self, docs: DocumentArray, **kwargs):
        default_logger.info("Executing extract")
        default_logger.info(kwargs)
        docs[0].text = 'AA - hello, world!'
        docs[1].text = 'AA - goodbye, world!'

    @requests(on='/work')
    def work(self, parameters, **kwargs):
        print(parameters)


def extend_rest_interface(app):
    @app.get('/extension1')
    async def root():
        return {"message": "Hello World"}

    return app


marie.helper.extend_rest_interface = extend_rest_interface
