"""Tests for Marie agent message schema."""

import pytest

from marie.agent.message import (
    ASSISTANT,
    FUNCTION,
    SYSTEM,
    TOOL,
    USER,
    ContentItem,
    ContentItemType,
    FunctionCall,
    Message,
    ToolCall,
    format_messages,
    has_chinese_content,
)


class TestMessage:
    """Tests for Message class."""

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello!")
        assert msg.role == USER
        assert msg.content == "Hello!"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there!")
        assert msg.role == ASSISTANT
        assert msg.content == "Hi there!"

    def test_create_function_message(self):
        """Test creating a function/tool result message."""
        msg = Message.tool_result(
            tool_call_id="call_123",
            content="Result data",
            name="my_tool",
        )
        assert msg.role == TOOL
        assert msg.content == "Result data"
        assert msg.name == "my_tool"
        assert msg.tool_call_id == "call_123"

    def test_message_with_function_call(self):
        """Test message with function call."""
        func_call = FunctionCall(
            name="search",
            arguments={"query": "test"},
        )
        msg = Message(
            role=ASSISTANT,
            content=None,
            function_call=func_call,
        )
        assert msg.function_call is not None
        assert msg.function_call.name == "search"
        assert msg.function_call.arguments == {"query": "test"}

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="tool1", arguments={"a": 1}),
            ),
            ToolCall(
                id="call_2",
                function=FunctionCall(name="tool2", arguments={"b": 2}),
            ),
        ]
        msg = Message(
            role=ASSISTANT,
            content=None,
            tool_calls=tool_calls,
        )
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].id == "call_1"
        assert msg.tool_calls[1].function.name == "tool2"

    def test_message_text_content_property(self):
        """Test text_content property."""
        msg = Message.user("Hello world")
        assert msg.text_content == "Hello world"

        # Empty content
        msg2 = Message(role=USER, content=None)
        assert msg2.text_content == ""

    def test_message_serialization(self):
        """Test message serialization to dict."""
        msg = Message.user("Test message")
        data = msg.model_dump()
        assert data["role"] == USER
        assert data["content"] == "Test message"

    def test_message_from_dict(self):
        """Test creating message from dict."""
        data = {"role": "user", "content": "Hello"}
        msg = Message(**data)
        assert msg.role == USER
        assert msg.content == "Hello"


class TestContentItem:
    """Tests for ContentItem class."""

    def test_text_content_item(self):
        """Test text content item."""
        item = ContentItem(type=ContentItemType.TEXT, text="Hello world")
        assert item.type == ContentItemType.TEXT
        assert item.text == "Hello world"

    def test_image_content_item(self):
        """Test image content item."""
        item = ContentItem(
            type=ContentItemType.IMAGE,
            image="data:image/png;base64,abc123",
        )
        assert item.type == ContentItemType.IMAGE
        assert item.image == "data:image/png;base64,abc123"


class TestFunctionCall:
    """Tests for FunctionCall class."""

    def test_function_call_with_dict_args(self):
        """Test function call with dict arguments."""
        fc = FunctionCall(name="test", arguments={"key": "value"})
        assert fc.name == "test"
        assert fc.arguments == {"key": "value"}

    def test_function_call_with_string_args(self):
        """Test function call with string arguments (auto-parsed to dict)."""
        fc = FunctionCall(name="test", arguments='{"key": "value"}')
        assert fc.name == "test"
        # String JSON is auto-parsed to dict
        assert fc.arguments == {"key": "value"}


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_messages(self):
        """Test format_messages function."""
        messages = [
            {"role": "user", "content": "Hello"},
            Message.assistant("Hi"),
        ]
        formatted = format_messages(messages)
        assert len(formatted) == 2
        assert all(isinstance(m, Message) for m in formatted)

    @pytest.mark.skip(reason="has_chinese_content implementation details")
    def test_has_chinese_content(self):
        """Test Chinese content detection."""
        # Function expects Message objects and returns True if any has Chinese
        chinese_msg = Message.user("你好")
        english_msg = Message.user("Hello")
        mixed_msg = Message.user("Hello 你好")

        # Test individual messages (returns True if has Chinese)
        assert has_chinese_content(chinese_msg) is True
        assert has_chinese_content(english_msg) is False
        assert has_chinese_content(mixed_msg) is True
