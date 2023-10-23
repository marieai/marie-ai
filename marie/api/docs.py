from typing import Any, Optional

from docarray import BaseDoc
from docarray.typing import AnyTensor

from marie._core.definitions.events import AssetKey


class AssetKeyDoc(BaseDoc):
    asset_key: str  # AssetKey
    pages: Optional[list[int]]

    class Config:
        arbitrary_types_allowed = True


class StorageDoc(BaseDoc):
    content: Optional[Any]
    blob: Optional[Any]
    tensor: Optional[AnyTensor]
    tags: Optional[dict]  # type: ignore


class OutputDoc(BaseDoc):
    jobid: str
    status: str
