from typing import Any, List, Optional

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import AnyTensor

# from marie._core.definitions.events import AssetKey


# It is important to note that if the documents are not serializable we can get number or wierd exceptions


class AssetKeyDoc(BaseDoc):
    asset_key: str
    pages: Optional[list[int]]


class StorageDoc(BaseDoc):
    content: Optional[Any]
    blob: Optional[Any]
    tensor: Optional[AnyTensor]
    tags: Optional[dict]  # type: ignore
    embedding: Optional[AnyTensor]


class MarieDoc(ImageDoc):
    tags: dict = {}


class BatchableMarieDoc(MarieDoc):
    words: List
    boxes: List


class OutputDoc(BaseDoc):
    jobid: str
    status: str
