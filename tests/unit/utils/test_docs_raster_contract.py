import numpy as np
import pytest
from docarray import DocList
from PIL import Image

from marie.api.docs import MarieDoc
from marie.utils.docs import (
    UnsupportedOcrInputError,
    docs_from_file,
    frames_from_docs,
    frames_from_file,
)


def test_shared_helpers_use_tensor_backed_raster_contract(tmp_path):
    path = tmp_path / 'source.png'
    Image.new('RGB', (8, 6), (10, 20, 30)).save(path)

    docs = docs_from_file(path)
    frames = frames_from_file(path)

    assert len(docs) == len(frames) == 1
    assert np.array_equal(docs[0].tensor, frames[0])


def test_docs_from_file_rejects_semantic_documents(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_text('a,b\n1,2\n')

    with pytest.raises(UnsupportedOcrInputError):
        docs_from_file(path)


def test_frames_from_docs_rejects_tensorless_semantic_result():
    docs = DocList[MarieDoc]([MarieDoc()])

    with pytest.raises(ValueError, match='semantic results cannot be used'):
        frames_from_docs(docs)
