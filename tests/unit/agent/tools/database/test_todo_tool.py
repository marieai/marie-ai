"""Tests for TodoTool."""
import json
import uuid
from datetime import datetime, timezone

import pytest

from marie.agent.tools import ToolOutput
from marie.agent.tools.database.todo_tool import TodoAction, TodoTool

from .conftest import MockRecord


class TestTodoToolMetadata:
    """Tests for TodoTool metadata."""

    def test_name(self, db_config):
        """Tool should have correct name."""
        tool = TodoTool(db_config)
        assert tool.name == "todo"

    def test_description(self, db_config):
        """Tool should have description with all actions."""
        tool = TodoTool(db_config)
        for action in ["list", "add", "complete", "delete", "clear_completed"]:
            assert action in tool.description.lower()


class TestTodoToolList:
    """Tests for list action."""

    @pytest.mark.asyncio
    async def test_list_todos(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """list should return pending todos by default."""
        todo_id1 = uuid.uuid4()
        todo_id2 = uuid.uuid4()

        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({
                "id": todo_id1,
                "title": "High priority task",
                "priority": 1,
                "completed": False,
                "created_at": sample_timestamp,
            }),
            MockRecord({
                "id": todo_id2,
                "title": "Medium priority task",
                "priority": 2,
                "completed": False,
                "created_at": sample_timestamp,
            }),
        ]

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.LIST,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["count"] == 2
        assert data["pending"] == 2
        assert data["completed"] == 0
        assert data["todos"][0]["priority"] == "high"
        assert data["todos"][1]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_list_todos_with_completed(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """list with show_completed should include completed todos."""
        todo_id1 = uuid.uuid4()
        todo_id2 = uuid.uuid4()

        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({
                "id": todo_id1,
                "title": "Pending task",
                "priority": 2,
                "completed": False,
                "created_at": sample_timestamp,
            }),
            MockRecord({
                "id": todo_id2,
                "title": "Done task",
                "priority": 2,
                "completed": True,
                "created_at": sample_timestamp,
            }),
        ]

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.LIST,
            user_id=sample_user_id,
            show_completed=True
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["count"] == 2
        assert data["pending"] == 1
        assert data["completed"] == 1

    @pytest.mark.asyncio
    async def test_list_todos_empty(self, db_config, mock_pool, sample_user_id):
        """list should return empty list when no todos."""
        mock_pool._mock_fetch = lambda q, a: []

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.LIST,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["count"] == 0
        assert data["todos"] == []


class TestTodoToolAdd:
    """Tests for add action."""

    @pytest.mark.asyncio
    async def test_add_todo(self, db_config, mock_pool, sample_user_id):
        """add should create a new todo."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.ADD,
            user_id=sample_user_id,
            title="New task"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["title"] == "New task"
        assert data["priority"] == "medium"  # default
        assert "id" in data
        uuid.UUID(data["id"])  # Verify valid UUID

    @pytest.mark.asyncio
    async def test_add_todo_with_priority(self, db_config, mock_pool, sample_user_id):
        """add should accept priority parameter."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.ADD,
            user_id=sample_user_id,
            title="Urgent task",
            priority=1
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["priority"] == "high"

    @pytest.mark.asyncio
    async def test_add_todo_priority_bounds(self, db_config, mock_pool, sample_user_id):
        """add should clamp priority to valid range."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        # Test priority > 3
        result = await tool.acall(
            action=TodoAction.ADD,
            user_id=sample_user_id,
            title="Task with priority 5",
            priority=5
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["priority"] == "low"  # Clamped to 3

        # Test priority < 1
        result = await tool.acall(
            action=TodoAction.ADD,
            user_id=sample_user_id,
            title="Task with priority 0",
            priority=0
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["priority"] == "high"  # Clamped to 1

    @pytest.mark.asyncio
    async def test_add_todo_missing_title(self, db_config, mock_pool, sample_user_id):
        """add should error without title."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.ADD,
            user_id=sample_user_id
        )

        assert result.is_error is True
        assert "title is required" in result.content.lower()


class TestTodoToolComplete:
    """Tests for complete action."""

    @pytest.mark.asyncio
    async def test_complete_todo(self, db_config, mock_pool, sample_user_id):
        """complete should mark todo as done."""
        todo_id = uuid.uuid4()

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.COMPLETE,
            user_id=sample_user_id,
            todo_id=str(todo_id)
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["id"] == str(todo_id)
        assert data["completed"] is True

    @pytest.mark.asyncio
    async def test_complete_todo_not_found(self, db_config, mock_pool, sample_user_id):
        """complete should error for non-existent todo."""
        mock_pool._mock_execute = lambda q, a: "UPDATE 0"

        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.COMPLETE,
            user_id=sample_user_id,
            todo_id=str(uuid.uuid4())
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_complete_already_completed(self, db_config, mock_pool, sample_user_id):
        """complete should error if already completed."""
        todo_id = uuid.uuid4()
        mock_pool._mock_execute = lambda q, a: "UPDATE 0"

        async def mock_fetchrow(query, *args):
            return MockRecord({"completed": True})

        mock_pool.fetchrow = mock_fetchrow

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.COMPLETE,
            user_id=sample_user_id,
            todo_id=str(todo_id)
        )

        assert result.is_error is True
        assert "already completed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_complete_missing_id(self, db_config, mock_pool, sample_user_id):
        """complete should error without todo_id."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.COMPLETE,
            user_id=sample_user_id
        )

        assert result.is_error is True
        assert "todo_id is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_complete_invalid_uuid(self, db_config, mock_pool, sample_user_id):
        """complete should error with invalid UUID."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.COMPLETE,
            user_id=sample_user_id,
            todo_id="not-a-valid-uuid"
        )

        assert result.is_error is True
        assert "invalid" in result.content.lower()


class TestTodoToolDelete:
    """Tests for delete action."""

    @pytest.mark.asyncio
    async def test_delete_todo(self, db_config, mock_pool, sample_user_id):
        """delete should remove todo."""
        todo_id = uuid.uuid4()

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.DELETE,
            user_id=sample_user_id,
            todo_id=str(todo_id)
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["deleted"] is True
        assert data["id"] == str(todo_id)

    @pytest.mark.asyncio
    async def test_delete_todo_not_found(self, db_config, mock_pool, sample_user_id):
        """delete should error for non-existent todo."""
        mock_pool._mock_execute = lambda q, a: "DELETE 0"

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.DELETE,
            user_id=sample_user_id,
            todo_id=str(uuid.uuid4())
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_delete_missing_id(self, db_config, mock_pool, sample_user_id):
        """delete should error without todo_id."""
        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.DELETE,
            user_id=sample_user_id
        )

        assert result.is_error is True
        assert "todo_id is required" in result.content.lower()


class TestTodoToolClearCompleted:
    """Tests for clear_completed action."""

    @pytest.mark.asyncio
    async def test_clear_completed(self, db_config, mock_pool, sample_user_id):
        """clear_completed should remove all completed todos."""
        mock_pool._mock_execute = lambda q, a: "DELETE 5"

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.CLEAR_COMPLETED,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["cleared"] == 5

    @pytest.mark.asyncio
    async def test_clear_completed_none(self, db_config, mock_pool, sample_user_id):
        """clear_completed with no completed todos should return 0."""
        mock_pool._mock_execute = lambda q, a: "DELETE 0"

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=TodoAction.CLEAR_COMPLETED,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["cleared"] == 0


class TestTodoToolSyncCall:
    """Tests for synchronous call method."""

    def test_call_bridges_to_acall(self, db_config, mock_pool, sample_user_id):
        """call() should bridge to acall() via run_async."""
        mock_pool._mock_fetch = lambda q, a: []

        tool = TodoTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = tool.call(
            action=TodoAction.LIST,
            user_id=sample_user_id
        )

        assert isinstance(result, ToolOutput)
        assert result.is_error is False
