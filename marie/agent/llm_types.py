"""Native LLM types for Marie agent framework.

This module provides LLM-related types that are independent of marie.core,
enabling the agent framework to work standalone. These types are compatible
with common LLM APIs (OpenAI, etc.) and provide multimodal support.
"""

from __future__ import annotations

import base64
from enum import Enum
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class MessageRole(str, Enum):
    """Message role enumeration for LLM conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class TextBlock(BaseModel):
    """Text content block for multimodal messages."""

    block_type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image content block for multimodal messages.

    Supports images from bytes, file path, or URL. Images are stored
    as base64-encoded bytes internally.
    """

    block_type: Literal["image"] = "image"
    image: bytes | None = None
    path: str | None = None
    url: str | None = None
    image_mimetype: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def image_to_base64(self) -> Self:
        """Store the image as base64 and guess the mimetype when possible.

        In case the model was built passing image data but without a mimetype,
        we try to guess it using the filetype library.
        """
        if not self.image:
            return self

        try:
            # Check if image is already base64 encoded
            decoded_img = base64.b64decode(self.image)
        except Exception:
            decoded_img = self.image
            # Not base64 - encode it
            self.image = base64.b64encode(self.image)

        self._guess_mimetype(decoded_img)
        return self

    def _guess_mimetype(self, img_data: bytes) -> None:
        """Guess the image mimetype from the data."""
        if not self.image_mimetype:
            try:
                import filetype

                guess = filetype.guess(img_data)
                self.image_mimetype = guess.mime if guess else None
            except ImportError:
                # filetype not installed, leave mimetype as None
                pass

    def resolve_image(self, as_base64: bool = False) -> BytesIO:
        """Resolve an image such that PIL can read it.

        Args:
            as_base64: Whether the resolved image should be returned as base64-encoded bytes

        Returns:
            BytesIO containing the image data

        Raises:
            ValueError: If no image source is available
        """
        if self.image is not None:
            if as_base64:
                return BytesIO(self.image)
            return BytesIO(base64.b64decode(self.image))
        elif self.path is not None:
            with open(self.path, "rb") as f:
                img_bytes = f.read()
            self._guess_mimetype(img_bytes)
            if as_base64:
                return BytesIO(base64.b64encode(img_bytes))
            return BytesIO(img_bytes)
        elif self.url is not None:
            import requests

            response = requests.get(str(self.url))
            img_bytes = response.content
            self._guess_mimetype(img_bytes)
            if as_base64:
                return BytesIO(base64.b64encode(img_bytes))
            return BytesIO(img_bytes)
        else:
            raise ValueError("No image found in the image block!")


# Union type for content blocks
ContentBlock = Annotated[
    Union[TextBlock, ImageBlock], Field(discriminator="block_type")
]


class ChatMessage(BaseModel):
    """Chat message for LLM communication.

    Provides a unified message format that supports multimodal content
    through content blocks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: MessageRole = MessageRole.USER
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[ContentBlock] = Field(default_factory=list)

    def __init__(self, /, content: Any | None = None, **data: Any) -> None:
        """Initialize with backward compatibility for the old `content` field.

        If content was passed and contained text, store a single TextBlock.
        If content was passed and it was a list, assume it's a list of content blocks.
        """
        if content is not None:
            if isinstance(content, str):
                data["blocks"] = [TextBlock(text=content)]
            elif isinstance(content, list):
                data["blocks"] = content

        super().__init__(**data)

    @property
    def content(self) -> str | None:
        """Get the cumulative text content from all TextBlock blocks.

        Returns:
            The cumulative content of the TextBlock blocks, None if there are none.
        """
        content = ""
        for block in self.blocks:
            if isinstance(block, TextBlock):
                content += block.text

        return content or None

    @content.setter
    def content(self, content: str) -> None:
        """Set the text content.

        Raises:
            ValueError: If blocks contains more than a block, or a block that's not TextBlock.
        """
        if not self.blocks:
            self.blocks = [TextBlock(text=content)]
        elif len(self.blocks) == 1 and isinstance(self.blocks[0], TextBlock):
            self.blocks = [TextBlock(text=content)]
        else:
            raise ValueError(
                "ChatMessage contains multiple blocks, use 'ChatMessage.blocks' instead."
            )

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> Self:
        """Create a ChatMessage from a string.

        Args:
            content: The text content
            role: The message role
            **kwargs: Additional arguments

        Returns:
            A new ChatMessage instance
        """
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, blocks=[TextBlock(text=content)], **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        """Recursively serialize nested models."""
        if isinstance(value, BaseModel):
            value.model_rebuild()
            return value.model_dump()
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(val) for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]
        return value

    @field_serializer("additional_kwargs", check_fields=False)
    def serialize_additional_kwargs(self, value: Any, _info: Any) -> Any:
        """Serialize additional_kwargs with proper handling of nested models."""
        return self._recursive_serialization(value)


class ChatResponse(BaseModel):
    """Response from an LLM chat completion."""

    message: ChatMessage
    raw: Optional[Any] = None
    delta: Optional[str] = None
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.message)


class CompletionResponse(BaseModel):
    """Response from an LLM completion.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response.
        raw: Optional raw JSON that was parsed to populate text.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    raw: Optional[Any] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text
