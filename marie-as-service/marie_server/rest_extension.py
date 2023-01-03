import numpy as np
import torch
import marie
import marie.helper

from typing import Dict, Union, Optional, TYPE_CHECKING
from marie import DocumentArray, Executor, requests
from marie.logging.predefined import default_logger


if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI


def extend_rest_interface(app: 'FastAPI') -> 'FastAPI':
    @app.get('/extension-a', tags=['Extract - REST API', 'extract-rest'])
    async def extension_A():
        default_logger.info("Executing A extension")
        return {"message": "ABC"}

    @app.get('/extension-b', tags=['Extract - REST API', 'extract-rest'])
    async def extension_B():
        default_logger.info("Executing B extension")
        return {"message": "XYZ"}

    return app
