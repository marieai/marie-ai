import numpy as np
import pytest
from docarray import DocumentArray
from marie import requests, safely_encoded
from marie.serve.executors import BaseExecutor


class EncodingExecutor(BaseExecutor):
    @requests
    @safely_encoded
    def encoded(self, docs, **kwargs):
        return self.retval

    def __init__(self, retval, *args, **kwargs):
        self.retval = retval

    @requests
    def raw(self, docs, **kwargs):
        return self.retval


@pytest.mark.parametrize(
    'encode,inputs,expected_values',
    [
        (
            True,
            {"k": 1, "complex": ["a", "b"], "arr": np.array([1, 2, 3])},
            {"k": 1, "complex": ["a", "b"], "arr": [1, 2, 3]},
        ),
        (
            True,
            [
                {"k": 1, "complex": ["a", "b"], "arr": np.array([1, 2, 3])},
                {"k": 2, "complex": ["c", "d"], "arr": np.array([4, 5, 6])},
            ],
            [
                {"k": 1, "complex": ["a", "b"], "arr": [1, 2, 3]},
                {"k": 2, "complex": ["c", "d"], "arr": [4, 5, 6]},
            ],
        ),
        (
            False,
            {"k": 1, "complex": ["a", "a"], "arr": [1, 2, 3]},
            {"k": 1, "complex": ["a", "a"], "arr": [1, 2, 3]},
        ),
    ],
)
def test_encoding_executor(encode, inputs, expected_values):

    exec = EncodingExecutor(inputs)

    result = (
        exec.encoded(docs=DocumentArray.empty(0))
        if encode
        else exec.raw(docs=DocumentArray.empty(0))
    )

    assert result == expected_values
