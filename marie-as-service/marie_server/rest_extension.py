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
    """Register executors REST endpoints that do not depend on DocumentArray
    :param app:
    :return:
    """

    from .executors.extract.mserve_torch import (
        extend_rest_interface_extract_mixin,
    )
    from .executors.ner.mserve_torch import (
        extend_rest_interface_ner_mixin,
    )

    extend_rest_interface_extract_mixin(app)
    extend_rest_interface_ner_mixin(app)

    return app
