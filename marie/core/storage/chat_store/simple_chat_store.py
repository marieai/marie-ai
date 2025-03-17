import json
import os
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

import fsspec
from marie.core.bridge.pydantic import Field, WrapSerializer
from marie.core.llms import ChatMessage
from marie.core.storage.chat_store.base import BaseChatStore


def chat_message_serialization(
    chat_message: Any, handler: Any, info: Any
) -> Dict[str, Any]:
    partial_result = handler(chat_message, info)

    for key, value in partial_result.get("additional_kwargs", {}).items():
        value = chat_message._recursive_serialization(value)
        if not isinstance(value, (str, int, float, bool, dict, list, type(None))):
            raise ValueError(f"Failed to serialize additional_kwargs value: {value}")
        partial_result["additional_kwargs"][key] = value

    return partial_result


AnnotatedChatMessage = Annotated[
    ChatMessage, WrapSerializer(chat_message_serialization)
]


class SimpleChatStore(BaseChatStore):
    """Simple chat store. Async methods provide same functionality as sync methods in this class."""

    store: Dict[str, List[AnnotatedChatMessage]] = Field(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "SimpleChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        self.store[key] = messages

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        return self.store.get(key, [])

    def add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """Add a message for a key."""
        if idx is None:
            self.store.setdefault(key, []).append(message)
        else:
            self.store.setdefault(key, []).insert(idx, message)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        if key not in self.store:
            return None
        return self.store.pop(key)

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        if key not in self.store:
            return None
        if idx >= len(self.store[key]):
            return None
        return self.store[key].pop(idx)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        if key not in self.store:
            return None
        return self.store[key].pop()

    def get_keys(self) -> List[str]:
        """Get all keys."""
        return list(self.store.keys())

    def persist(
        self,
        persist_path: str = "chat_store.json",
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        fs = fs or fsspec.filesystem("file")
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w") as f:
            f.write(self.json())

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str = "chat_store.json",
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleChatStore":
        """Create a SimpleChatStore from a persist path."""
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(persist_path):
            return cls()
        with fs.open(persist_path, "r") as f:
            data = json.load(f)

        if isinstance(data, str):
            return cls.model_validate_json(data)
        else:
            return cls.model_validate(data)
