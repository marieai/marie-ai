"""Unified message schema for Marie agent framework.

This module provides a Qwen-Agent compatible message format that bridges
with marie.core.base.llms.types.ChatMessage.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from marie.core.base.llms.types import ChatMessage

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Constants for message keys (Qwen-Agent compatibility)
ROLE = "role"
CONTENT = "content"
NAME = "name"
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
FUNCTION = "function"
TOOL = "tool"

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class ContentItemType(str, Enum):
    """Content item type enumeration."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"


class ContentItem(BaseModel):
    """Multimodal content item for agent messages.

    Supports text, images (URL or base64), files, audio, and video content.
    Designed for compatibility with Qwen-Agent's ContentItem pattern.
    """

    model_config = ConfigDict(extra="allow")

    text: Optional[str] = Field(default=None, description="Text content")
    image: Optional[str] = Field(
        default=None, description="Image URL or base64-encoded data"
    )
    file: Optional[str] = Field(default=None, description="File path or URL")
    audio: Optional[str] = Field(
        default=None, description="Audio URL or base64-encoded data"
    )
    video: Optional[str] = Field(
        default=None, description="Video URL or base64-encoded data"
    )

    @model_validator(mode="after")
    def validate_at_least_one_content(self) -> "ContentItem":
        """Ensure at least one content field is set."""
        if not any([self.text, self.image, self.file, self.audio, self.video]):
            raise ValueError("ContentItem must have at least one content field set")
        return self

    @property
    def type(self) -> ContentItemType:
        """Determine the content type based on which field is set."""
        if self.text is not None:
            return ContentItemType.TEXT
        if self.image is not None:
            return ContentItemType.IMAGE
        if self.file is not None:
            return ContentItemType.FILE
        if self.audio is not None:
            return ContentItemType.AUDIO
        if self.video is not None:
            return ContentItemType.VIDEO
        raise ValueError("No content type could be determined")

    def get_content(self) -> str:
        """Get the primary content value."""
        if self.text is not None:
            return self.text
        if self.image is not None:
            return self.image
        if self.file is not None:
            return self.file
        if self.audio is not None:
            return self.audio
        if self.video is not None:
            return self.video
        return ""


class FunctionCall(BaseModel):
    """Function/tool call information.

    Represents a function call request from the LLM, compatible with
    OpenAI function calling format.
    """

    name: str = Field(..., description="Name of the function to call")
    arguments: Union[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Function arguments as string or dict"
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Union[str, Dict[str, Any]]:
        """Parse arguments from string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    def get_arguments_dict(self) -> Dict[str, Any]:
        """Get arguments as a dictionary."""
        if isinstance(self.arguments, str):
            try:
                return json.loads(self.arguments)
            except json.JSONDecodeError:
                return {"raw": self.arguments}
        return self.arguments

    def get_arguments_str(self) -> str:
        """Get arguments as a JSON string."""
        if isinstance(self.arguments, dict):
            return json.dumps(self.arguments, ensure_ascii=False)
        return self.arguments


class ToolCall(BaseModel):
    """Tool call with ID (OpenAI-style tool calls).

    Extends FunctionCall with an ID for tracking in multi-turn conversations.
    """

    id: str = Field(..., description="Unique identifier for this tool call")
    type: Literal["function"] = Field(default="function", description="Tool type")
    function: FunctionCall = Field(..., description="The function call details")


class Message(BaseModel):
    """Unified message format for agent communication.

    Provides a single message schema that works across different agent backends
    and is compatible with both Qwen-Agent and OpenAI message formats.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    role: str = Field(
        ..., description="Message role: system, user, assistant, function, tool"
    )
    content: Union[str, List[ContentItem], None] = Field(
        default=None, description="Message content (text or multimodal)"
    )
    name: Optional[str] = Field(
        default=None, description="Name of the sender (for function/tool responses)"
    )
    function_call: Optional[FunctionCall] = Field(
        default=None, description="Function call request (deprecated, use tool_calls)"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls requested by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call this message responds to"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for Qwen-Agent compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment for Qwen-Agent compatibility."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        return getattr(self, key, default)

    @property
    def text_content(self) -> str:
        """Extract text content as a single string."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Handle list of ContentItems
        text_parts = []
        for item in self.content:
            if isinstance(item, ContentItem) and item.text:
                text_parts.append(item.text)
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(item["text"])
        return "\n".join(text_parts)

    def to_chat_message(self) -> "ChatMessage":
        """Convert to marie.core.base.llms.types.ChatMessage."""
        from marie.core.base.llms.types import (
            ChatMessage,
            ImageBlock,
            MessageRole,
            TextBlock,
        )

        # Map role string to MessageRole enum
        role_map = {
            SYSTEM: MessageRole.SYSTEM,
            USER: MessageRole.USER,
            ASSISTANT: MessageRole.ASSISTANT,
            FUNCTION: MessageRole.FUNCTION,
            TOOL: MessageRole.TOOL,
        }
        message_role = role_map.get(self.role, MessageRole.USER)

        # Convert content to blocks
        blocks = []
        if self.content is not None:
            if isinstance(self.content, str):
                blocks.append(TextBlock(text=self.content))
            else:
                for item in self.content:
                    if isinstance(item, ContentItem):
                        if item.text:
                            blocks.append(TextBlock(text=item.text))
                        elif item.image:
                            blocks.append(ImageBlock(url=item.image))
                    elif isinstance(item, dict):
                        if item.get("text"):
                            blocks.append(TextBlock(text=item["text"]))
                        elif item.get("image"):
                            blocks.append(ImageBlock(url=item["image"]))

        # Build additional_kwargs
        additional_kwargs: Dict[str, Any] = {}
        if self.function_call:
            additional_kwargs["function_call"] = self.function_call.model_dump()
        if self.tool_calls:
            additional_kwargs["tool_calls"] = [
                tc.model_dump() for tc in self.tool_calls
            ]
        if self.tool_call_id:
            additional_kwargs["tool_call_id"] = self.tool_call_id
        if self.name:
            additional_kwargs["name"] = self.name

        return ChatMessage(
            role=message_role, blocks=blocks, additional_kwargs=additional_kwargs
        )

    @classmethod
    def from_chat_message(cls, chat_msg: "ChatMessage") -> "Message":
        """Create from marie.core.base.llms.types.ChatMessage."""
        from marie.core.base.llms.types import ImageBlock, TextBlock

        # Convert role
        role = chat_msg.role.value

        # Convert blocks to content
        content: Union[str, List[ContentItem], None] = None
        if chat_msg.blocks:
            if len(chat_msg.blocks) == 1 and isinstance(chat_msg.blocks[0], TextBlock):
                content = chat_msg.blocks[0].text
            else:
                content_items = []
                for block in chat_msg.blocks:
                    if isinstance(block, TextBlock):
                        content_items.append(ContentItem(text=block.text))
                    elif isinstance(block, ImageBlock):
                        if block.url:
                            content_items.append(ContentItem(image=str(block.url)))
                        elif block.image:
                            content_items.append(ContentItem(image=str(block.image)))
                content = content_items if content_items else None

        # Extract function call from additional_kwargs
        function_call = None
        if fc := chat_msg.additional_kwargs.get("function_call"):
            function_call = FunctionCall(**fc)

        # Extract tool calls
        tool_calls = None
        if tc := chat_msg.additional_kwargs.get("tool_calls"):
            tool_calls = [ToolCall(**t) for t in tc]

        return cls(
            role=role,
            content=content,
            name=chat_msg.additional_kwargs.get("name"),
            function_call=function_call,
            tool_calls=tool_calls,
            tool_call_id=chat_msg.additional_kwargs.get("tool_call_id"),
        )

    @classmethod
    def system(cls, content: str, **kwargs: Any) -> "Message":
        """Create a system message."""
        return cls(role=SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: Union[str, List[ContentItem]], **kwargs: Any) -> "Message":
        """Create a user message."""
        return cls(role=USER, content=content, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: Optional[Union[str, List[ContentItem]]] = None,
        function_call: Optional[FunctionCall] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        **kwargs: Any,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role=ASSISTANT,
            content=content,
            function_call=function_call,
            tool_calls=tool_calls,
            **kwargs,
        )

    @classmethod
    def function_result(cls, name: str, content: str, **kwargs: Any) -> "Message":
        """Create a function result message."""
        return cls(role=FUNCTION, name=name, content=content, **kwargs)

    @classmethod
    def tool_result(
        cls, tool_call_id: str, content: str, name: Optional[str] = None, **kwargs: Any
    ) -> "Message":
        """Create a tool result message."""
        return cls(
            role=TOOL,
            tool_call_id=tool_call_id,
            content=content,
            name=name,
            **kwargs,
        )


def format_messages(messages: List[Union[Dict[str, Any], Message]]) -> List[Message]:
    """Convert a list of dicts or Messages to Message objects.

    Args:
        messages: List of message dicts or Message objects

    Returns:
        List of Message objects
    """
    result = []
    for msg in messages:
        if isinstance(msg, Message):
            result.append(msg)
        elif isinstance(msg, dict):
            result.append(Message(**msg))
        else:
            raise TypeError(f"Expected dict or Message, got {type(msg)}")
    return result


def has_chinese_content(messages: List[Message]) -> bool:
    """Check if any message contains Chinese characters.

    Args:
        messages: List of messages to check

    Returns:
        True if any message contains Chinese characters
    """
    for msg in messages:
        text = msg.text_content
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return True
    return False
