from typing import Optional, Sequence

import fsspec
from marie.core.bridge.pydantic import BaseModel, Field, ConfigDict
from marie.core.schema import BaseNode
from marie.core.storage.docstore.utils import doc_to_json, json_to_doc
from marie.core.storage.kvstore import (
    SimpleKVStore as SimpleCache,
)
from marie.core.storage.kvstore.types import (
    BaseKVStore as BaseCache,
)

DEFAULT_CACHE_NAME = "llama_cache"


class IngestionCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nodes_key: str = "nodes"

    collection: str = Field(
        default=DEFAULT_CACHE_NAME, description="Collection name of the cache."
    )
    cache: BaseCache = Field(default_factory=SimpleCache, description="Cache to use.")

    # TODO: add async get/put methods?
    def put(
        self, key: str, nodes: Sequence[BaseNode], collection: Optional[str] = None
    ) -> None:
        """Put a value into the cache."""
        collection = collection or self.collection

        val = {self.nodes_key: [doc_to_json(node) for node in nodes]}
        self.cache.put(key, val, collection=collection)

    def get(
        self, key: str, collection: Optional[str] = None
    ) -> Optional[Sequence[BaseNode]]:
        """Get a value from the cache."""
        collection = collection or self.collection
        node_dicts = self.cache.get(key, collection=collection)

        if node_dicts is None:
            return None

        return [json_to_doc(node_dict) for node_dict in node_dicts[self.nodes_key]]

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear the cache."""
        collection = collection or self.collection
        data = self.cache.get_all(collection=collection)
        for key in data:
            self.cache.delete(key, collection=collection)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the cache to a directory, if possible."""
        if isinstance(self.cache, SimpleCache):
            self.cache.persist(persist_path, fs=fs)
        else:
            print("Warning: skipping persist, only needed for SimpleCache.")

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        collection: str = DEFAULT_CACHE_NAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "IngestionCache":
        """Create a IngestionCache from a persist directory."""
        return cls(
            collection=collection,
            cache=SimpleCache.from_persist_path(persist_path, fs=fs),
        )


__all__ = ["SimpleCache", "BaseCache"]
