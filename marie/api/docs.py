from enum import Enum, auto
from typing import Any, List, Optional

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import AnyTensor

# from marie._core.definitions.events import AssetKey

# It is important to note that if the documents are not serializable we can get number or wierd exceptions

DOC_KEY_PAGE_NUMBER = "page_number"
DOC_KEY_CLASSIFICATION = "classification"
DOC_KEY_INDEXER = "indexer"
DOC_KEY_ASSET_KEY = "asset_key"


class AssetKeyDoc(BaseDoc):
    asset_key: str
    pages: Optional[list[int]] = None


class StorageDoc(BaseDoc):
    content: Optional[Any] = None
    blob: Optional[Any] = None
    tensor: Optional[AnyTensor] = None
    tags: Optional[dict] = None  # type: ignore
    embedding: Optional[AnyTensor] = None


class MarieDoc(ImageDoc):
    tags: dict = {}


class BatchableMarieDoc(MarieDoc):
    words: List
    boxes: List


class OutputDoc(BaseDoc):
    jobid: str
    status: str
