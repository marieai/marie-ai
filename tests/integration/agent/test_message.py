"""Integration tests for the Message system.

Tests message creation, serialization, content handling, and conversions.
"""

import json

import pytest

from marie.agent import (
    ASSISTANT,
    FUNCTION,
    SYSTEM,
    TOOL,
    USER,
    ContentItem,
    FunctionCall,
    Message,
    ToolCall,
    format_messages,
    has_chinese_content,
)


class TestMessageCreation:
    """Test message creation and factory methods."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello, world!")
        assert msg.role == USER
        assert msg.content == "Hello, world!"
        assert msg.text_content == "Hello, world!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("I am here to help!")
        assert msg.role == ASSISTANT
        assert msg.content == "I am here to help!"

    def test_create_function_result_message(self):
        """Test creating a function result message."""
        msg = Message.function_result("search", '{"results": []}')
        assert msg.role == FUNCTION
        assert msg.name == "search"
        assert msg.content == '{"results": []}'

    def test_create_tool_result_message(self):
        """Test creating a tool result message."""
        msg = Message.tool_result("call_123", '{"data": "value"}', name="my_tool")
        assert msg.role == TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.name == "my_tool"
        assert msg.content == '{"data": "value"}'

    def test_create_message_from_dict(self):
        """Test creating a message from a dictionary."""
        msg = Message(**{"role": "user", "content": "Hello"})
        assert msg.role == USER
        assert msg.content == "Hello"

    def test_message_with_none_content(self):
        """Test message with None content."""
        msg = Message.assistant(content=None)
        assert msg.content is None
        assert msg.text_content == ""


class TestMessageWithFunctionCall:
    """Test messages containing function calls."""

    def test_message_with_function_call(self):
        """Test creating an assistant message with function call."""
        fc = FunctionCall(name="search", arguments={"query": "test"})
        msg = Message.assistant(content="Searching...", function_call=fc)

        assert msg.function_call is not None
        assert msg.function_call.name == "search"
        assert msg.function_call.get_arguments_dict() == {"query": "test"}

    def test_function_call_arguments_as_string(self):
        """Test function call with JSON string arguments."""
        fc = FunctionCall(name="calc", arguments='{"expression": "2+2"}')
        assert fc.get_arguments_dict() == {"expression": "2+2"}
        assert fc.get_arguments_str() == '{"expression": "2+2"}'

    def test_function_call_arguments_as_dict(self):
        """Test function call with dict arguments."""
        fc = FunctionCall(name="calc", arguments={"expression": "2+2"})
        assert fc.get_arguments_dict() == {"expression": "2+2"}
        assert "expression" in fc.get_arguments_str()

    def test_message_with_tool_calls(self):
        """Test message with OpenAI-style tool calls."""
        tc = ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(name="search", arguments={"query": "test"}),
        )
        msg = Message.assistant(content=None, tool_calls=[tc])

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_abc"
        assert msg.tool_calls[0].function.name == "search"


class TestMultimodalContent:
    """Test multimodal content handling."""

    def test_content_item_with_text(self):
        """Test ContentItem with text."""
        item = ContentItem(text="Hello")
        assert item.text == "Hello"
        assert item.type.value == "text"
        assert item.get_content() == "Hello"

    def test_content_item_with_image(self):
        """Test ContentItem with image."""
        item = ContentItem(image="https://example.com/image.jpg")
        assert item.image == "https://example.com/image.jpg"
        assert item.type.value == "image"

    def test_content_item_with_file(self):
        """Test ContentItem with file."""
        item = ContentItem(file="/path/to/file.pdf")
        assert item.file == "/path/to/file.pdf"
        assert item.type.value == "file"

    def test_content_item_requires_at_least_one_field(self):
        """Test that ContentItem requires at least one content field."""
        with pytest.raises(ValueError, match="at least one content field"):
            ContentItem()

    def test_message_with_multimodal_content(self):
        """Test message with multiple content items."""
        content = [
            ContentItem(text="What is in this image?"),
            ContentItem(image="https://example.com/cat.jpg"),
        ]
        msg = Message.user(content)

        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.text_content == "What is in this image?"

    def test_text_content_extraction_from_list(self):
        """Test extracting text content from multimodal message."""
        content = [
            ContentItem(text="First part."),
            ContentItem(image="image.jpg"),
            ContentItem(text="Second part."),
        ]
        msg = Message.user(content)

        assert "First part." in msg.text_content
        assert "Second part." in msg.text_content


class TestMessageDictInterface:
    """Test dict-like access for Qwen-Agent compatibility."""

    def test_getitem(self):
        """Test dict-style attribute access."""
        msg = Message.user("Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_setitem(self):
        """Test dict-style attribute assignment."""
        msg = Message.user("Hello")
        msg["content"] = "Updated"
        assert msg.content == "Updated"

    def test_get_with_default(self):
        """Test get method with default value."""
        msg = Message.user("Hello")
        # Note: get() returns None for existing fields that are None,
        # it doesn't use the default value like dict.get()
        assert msg.get("name") is None
        assert msg.get("nonexistent_field", "default") == "default"
        assert msg.get("role") == "user"


class TestMessageSerialization:
    """Test message serialization and deserialization."""

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message.user("Hello")
        data = msg.model_dump()

        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_message_round_trip(self):
        """Test message serialization round trip."""
        original = Message.assistant(
            content="Response",
            function_call=FunctionCall(name="test", arguments={"a": 1}),
        )

        data = original.model_dump()
        restored = Message(**data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.function_call.name == original.function_call.name

    def test_format_messages_from_dicts(self):
        """Test format_messages with dicts."""
        messages = format_messages([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])

        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)
        assert messages[0].role == USER
        assert messages[1].role == ASSISTANT

    def test_format_messages_mixed(self):
        """Test format_messages with mixed input."""
        messages = format_messages([
            {"role": "user", "content": "Hello"},
            Message.assistant("Hi!"),
        ])

        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)


class TestLanguageDetection:
    """Test language detection utilities."""

    def test_detect_chinese_content(self):
        """Test detecting Chinese characters."""
        messages = [Message.user("你好，世界！")]
        assert has_chinese_content(messages) is True

    def test_detect_english_content(self):
        """Test detecting English-only content."""
        messages = [Message.user("Hello, world!")]
        assert has_chinese_content(messages) is False

    def test_detect_mixed_content(self):
        """Test detecting mixed Chinese and English."""
        messages = [Message.user("Hello 你好")]
        assert has_chinese_content(messages) is True

    def test_empty_messages(self):
        """Test with empty message list."""
        assert has_chinese_content([]) is False

    def test_multimodal_chinese_content(self):
        """Test Chinese detection in multimodal messages."""
        messages = [
            Message.user([
                ContentItem(text="请描述这张图片"),
                ContentItem(image="image.jpg"),
            ])
        ]
        assert has_chinese_content(messages) is True


class TestMessageConversion:
    """Test conversion between Message and ChatMessage formats."""

    def test_to_chat_message(self):
        """Test converting Message to ChatMessage."""
        msg = Message.user("Hello")
        chat_msg = msg.to_chat_message()

        assert chat_msg.role.value == "user"
        assert len(chat_msg.blocks) == 1

    def test_from_chat_message(self):
        """Test creating Message from ChatMessage."""
        from marie.agent.llm_types import ChatMessage, MessageRole, TextBlock

        chat_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[TextBlock(text="Hello from assistant")],
        )
        msg = Message.from_chat_message(chat_msg)

        assert msg.role == ASSISTANT
        assert msg.content == "Hello from assistant"

    def test_multimodal_chat_message_conversion(self):
        """Test multimodal message conversion."""
        msg = Message.user([
            ContentItem(text="Describe this"),
            ContentItem(image="https://example.com/image.jpg"),
        ])
        chat_msg = msg.to_chat_message()

        assert len(chat_msg.blocks) == 2

    def test_round_trip_conversion(self):
        """Test Message -> ChatMessage -> Message round trip."""
        original = Message.user("Test message")
        chat_msg = original.to_chat_message()
        restored = Message.from_chat_message(chat_msg)

        assert restored.role == original.role
        assert restored.text_content == original.text_content
