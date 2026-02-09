"""Tests for MemoryTool."""
import json
from datetime import datetime, timezone

import pytest

from marie.agent.tools import ToolOutput
from marie.agent.tools.database.memory_tool import MemoryAction, MemoryTool

from .conftest import MockRecord


class TestMemoryToolMetadata:
    """Tests for MemoryTool metadata."""

    def test_name(self, db_config):
        """Tool should have correct name."""
        tool = MemoryTool(db_config)
        assert tool.name == "memory"

    def test_description(self, db_config):
        """Tool should have description with all actions."""
        tool = MemoryTool(db_config)
        for action in ["view", "create", "str_replace", "insert", "delete", "rename"]:
            assert action in tool.description.lower()


class TestMemoryToolPathValidation:
    """Tests for path validation."""

    @pytest.mark.asyncio
    async def test_path_must_start_with_slash(self, db_config, mock_pool, sample_user_id):
        """Path must start with '/'."""
        tool = MemoryTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=MemoryAction.VIEW,
            user_id=sample_user_id,
            path="invalid/path"
        )

        assert result.is_error is True
        assert "must start with '/'" in result.content


class TestMemoryToolView:
    """Tests for view action."""

    @pytest.mark.asyncio
    async def test_view_file(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """view should return file content."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({
                "content": "Hello, World!",
                "is_directory": False,
                "updated_at": sample_timestamp,
            }),
        ]

        # Override fetchrow to return a single record
        async def mock_fetchrow(query, *args):
            mock_pool._execute_log.append(("fetchrow", query, args))
            results = mock_pool._mock_fetch(query, args)
            return results[0] if results else None

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True  # Skip schema creation

        result = await tool.acall(
            action=MemoryAction.VIEW,
            user_id=sample_user_id,
            path="/notes/todo.txt"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["path"] == "/notes/todo.txt"
        assert data["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_view_directory(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """view on directory should list contents."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({"path": "/notes/file1.txt", "is_directory": False, "updated_at": sample_timestamp}),
            MockRecord({"path": "/notes/file2.txt", "is_directory": False, "updated_at": sample_timestamp}),
            MockRecord({"path": "/notes/subdir/", "is_directory": True, "updated_at": sample_timestamp}),
        ]

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.VIEW,
            user_id=sample_user_id,
            path="/notes/"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["path"] == "/notes/"
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_view_file_not_found(self, db_config, mock_pool, sample_user_id):
        """view should return error for non-existent file."""
        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.VIEW,
            user_id=sample_user_id,
            path="/nonexistent.txt"
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_view_file_is_directory(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """view on directory path without trailing slash should error."""
        async def mock_fetchrow(query, *args):
            return MockRecord({
                "content": None,
                "is_directory": True,
                "updated_at": sample_timestamp,
            })

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.VIEW,
            user_id=sample_user_id,
            path="/notes"  # Missing trailing slash
        )

        assert result.is_error is True
        assert "directory" in result.content.lower()


class TestMemoryToolCreate:
    """Tests for create action."""

    @pytest.mark.asyncio
    async def test_create_file(self, db_config, mock_pool, sample_user_id):
        """create should create a new file."""
        async def mock_fetchrow(query, *args):
            return None  # File doesn't exist

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.CREATE,
            user_id=sample_user_id,
            path="/notes/new.txt",
            content="New file content"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["created"] == "/notes/new.txt"
        assert data["size"] == len("New file content")

    @pytest.mark.asyncio
    async def test_create_file_already_exists(self, db_config, mock_pool, sample_user_id):
        """create should error if file exists."""
        async def mock_fetchrow(query, *args):
            return MockRecord({"path": "/notes/existing.txt"})

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.CREATE,
            user_id=sample_user_id,
            path="/notes/existing.txt",
            content="Content"
        )

        assert result.is_error is True
        assert "already exists" in result.content.lower()

    @pytest.mark.asyncio
    async def test_create_directory(self, db_config, mock_pool, sample_user_id):
        """create with trailing slash should create directory."""
        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.CREATE,
            user_id=sample_user_id,
            path="/projects/new_project/"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["created"] == "/projects/new_project/"
        assert data["type"] == "directory"


class TestMemoryToolStrReplace:
    """Tests for str_replace action."""

    @pytest.mark.asyncio
    async def test_str_replace_success(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """str_replace should replace text in file."""
        async def mock_fetchrow(query, *args):
            return MockRecord({
                "content": "Hello World Hello",
                "is_directory": False,
            })

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.STR_REPLACE,
            user_id=sample_user_id,
            path="/test.txt",
            old_str="Hello",
            new_str="Hi"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["replacements"] == 2  # "Hello" appears twice

    @pytest.mark.asyncio
    async def test_str_replace_string_not_found(self, db_config, mock_pool, sample_user_id):
        """str_replace should error if string not found."""
        async def mock_fetchrow(query, *args):
            return MockRecord({
                "content": "Hello World",
                "is_directory": False,
            })

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.STR_REPLACE,
            user_id=sample_user_id,
            path="/test.txt",
            old_str="NotFound",
            new_str="Replacement"
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_str_replace_missing_old_str(self, db_config, mock_pool, sample_user_id):
        """str_replace should error without old_str."""
        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.STR_REPLACE,
            user_id=sample_user_id,
            path="/test.txt",
            new_str="Replacement"
        )

        assert result.is_error is True
        assert "old_str is required" in result.content.lower()


class TestMemoryToolInsert:
    """Tests for insert action."""

    @pytest.mark.asyncio
    async def test_insert_line(self, db_config, mock_pool, sample_user_id):
        """insert should insert text at specified line."""
        async def mock_fetchrow(query, *args):
            return MockRecord({
                "content": "Line 1\nLine 2\nLine 3",
                "is_directory": False,
            })

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.INSERT,
            user_id=sample_user_id,
            path="/test.txt",
            insert_line=2,
            content="New Line"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["inserted_at_line"] == 2
        assert data["new_line_count"] == 4

    @pytest.mark.asyncio
    async def test_insert_missing_line_number(self, db_config, mock_pool, sample_user_id):
        """insert should error without insert_line."""
        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.INSERT,
            user_id=sample_user_id,
            path="/test.txt",
            content="New content"
        )

        assert result.is_error is True
        assert "insert_line is required" in result.content.lower()


class TestMemoryToolDelete:
    """Tests for delete action."""

    @pytest.mark.asyncio
    async def test_delete_file(self, db_config, mock_pool, sample_user_id):
        """delete should remove file."""
        async def mock_fetchrow(query, *args):
            return MockRecord({"is_directory": False})

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.DELETE,
            user_id=sample_user_id,
            path="/test.txt"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["deleted"] == "/test.txt"

    @pytest.mark.asyncio
    async def test_delete_not_found(self, db_config, mock_pool, sample_user_id):
        """delete should error for non-existent path."""
        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.DELETE,
            user_id=sample_user_id,
            path="/nonexistent.txt"
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()


class TestMemoryToolRename:
    """Tests for rename action."""

    @pytest.mark.asyncio
    async def test_rename_file(self, db_config, mock_pool, sample_user_id):
        """rename should move file to new path."""
        call_count = [0]

        async def mock_fetchrow(query, *args):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockRecord({"is_directory": False})  # Source exists
            return None  # Destination doesn't exist

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.RENAME,
            user_id=sample_user_id,
            path="/old.txt",
            new_path="/new.txt"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["renamed"] == "/old.txt"
        assert data["to"] == "/new.txt"

    @pytest.mark.asyncio
    async def test_rename_missing_new_path(self, db_config, mock_pool, sample_user_id):
        """rename should error without new_path."""
        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.RENAME,
            user_id=sample_user_id,
            path="/old.txt"
        )

        assert result.is_error is True
        assert "new_path is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_rename_invalid_new_path(self, db_config, mock_pool, sample_user_id):
        """rename should error if new_path doesn't start with '/'."""
        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.RENAME,
            user_id=sample_user_id,
            path="/old.txt",
            new_path="invalid/path"
        )

        assert result.is_error is True
        assert "must start with '/'" in result.content

    @pytest.mark.asyncio
    async def test_rename_destination_exists(self, db_config, mock_pool, sample_user_id):
        """rename should error if destination exists."""
        async def mock_fetchrow(query, *args):
            return MockRecord({"is_directory": False, "path": "/something"})

        mock_pool.fetchrow = mock_fetchrow

        tool = MemoryTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=MemoryAction.RENAME,
            user_id=sample_user_id,
            path="/old.txt",
            new_path="/existing.txt"
        )

        assert result.is_error is True
        assert "already exists" in result.content.lower()
