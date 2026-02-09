"""Tests for NotesTool."""
import json
import uuid
from datetime import datetime, timezone

import pytest

from marie.agent.tools import ToolOutput
from marie.agent.tools.database.notes_tool import NotesAction, NotesTool

from .conftest import MockRecord


class TestNotesToolMetadata:
    """Tests for NotesTool metadata."""

    def test_name(self, db_config):
        """Tool should have correct name."""
        tool = NotesTool(db_config)
        assert tool.name == "notes"

    def test_description(self, db_config):
        """Tool should have description with all actions."""
        tool = NotesTool(db_config)
        for action in ["list", "get", "create", "update", "delete"]:
            assert action in tool.description.lower()


class TestNotesToolList:
    """Tests for list action."""

    @pytest.mark.asyncio
    async def test_list_notes(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """list should return all user notes."""
        note_id1 = uuid.uuid4()
        note_id2 = uuid.uuid4()

        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({
                "id": note_id1,
                "title": "Note 1",
                "content": "Content for note 1",
                "created_at": sample_timestamp,
                "updated_at": sample_timestamp,
            }),
            MockRecord({
                "id": note_id2,
                "title": "Note 2",
                "content": "Content for note 2 with more text to test preview truncation" * 5,
                "created_at": sample_timestamp,
                "updated_at": sample_timestamp,
            }),
        ]

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.LIST,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["count"] == 2
        assert len(data["notes"]) == 2
        assert data["notes"][0]["title"] == "Note 1"
        # Preview should be truncated with "..."
        assert "..." in data["notes"][1]["preview"]

    @pytest.mark.asyncio
    async def test_list_notes_with_search(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """list with search should filter notes."""
        note_id = uuid.uuid4()

        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({
                "id": note_id,
                "title": "Meeting Notes",
                "content": "Discussion about project",
                "created_at": sample_timestamp,
                "updated_at": sample_timestamp,
            }),
        ]

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.LIST,
            user_id=sample_user_id,
            search="meeting"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["search"] == "meeting"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_list_notes_empty(self, db_config, mock_pool, sample_user_id):
        """list should return empty list when no notes."""
        mock_pool._mock_fetch = lambda q, a: []

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.LIST,
            user_id=sample_user_id
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["count"] == 0
        assert data["notes"] == []


class TestNotesToolGet:
    """Tests for get action."""

    @pytest.mark.asyncio
    async def test_get_note(self, db_config, mock_pool, sample_user_id, sample_timestamp):
        """get should return note details."""
        note_id = uuid.uuid4()

        async def mock_fetchrow(query, *args):
            return MockRecord({
                "id": note_id,
                "title": "My Note",
                "content": "Full content of the note",
                "created_at": sample_timestamp,
                "updated_at": sample_timestamp,
            })

        mock_pool.fetchrow = mock_fetchrow

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.GET,
            user_id=sample_user_id,
            note_id=str(note_id)
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["id"] == str(note_id)
        assert data["title"] == "My Note"
        assert data["content"] == "Full content of the note"

    @pytest.mark.asyncio
    async def test_get_note_not_found(self, db_config, mock_pool, sample_user_id):
        """get should error for non-existent note."""
        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.GET,
            user_id=sample_user_id,
            note_id=str(uuid.uuid4())
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_note_missing_id(self, db_config, mock_pool, sample_user_id):
        """get should error without note_id."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.GET,
            user_id=sample_user_id
        )

        assert result.is_error is True
        assert "note_id is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_note_invalid_uuid(self, db_config, mock_pool, sample_user_id):
        """get should error with invalid UUID."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.GET,
            user_id=sample_user_id,
            note_id="not-a-valid-uuid"
        )

        assert result.is_error is True
        assert "invalid" in result.content.lower()


class TestNotesToolCreate:
    """Tests for create action."""

    @pytest.mark.asyncio
    async def test_create_note(self, db_config, mock_pool, sample_user_id):
        """create should create a new note."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.CREATE,
            user_id=sample_user_id,
            title="New Note",
            content="Note content"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["title"] == "New Note"
        assert "id" in data
        # Verify it's a valid UUID
        uuid.UUID(data["id"])

    @pytest.mark.asyncio
    async def test_create_note_missing_title(self, db_config, mock_pool, sample_user_id):
        """create should error without title."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.CREATE,
            user_id=sample_user_id,
            content="Content without title"
        )

        assert result.is_error is True
        assert "title is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_create_note_without_content(self, db_config, mock_pool, sample_user_id):
        """create should allow empty content."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.CREATE,
            user_id=sample_user_id,
            title="Title Only Note"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["title"] == "Title Only Note"


class TestNotesToolUpdate:
    """Tests for update action."""

    @pytest.mark.asyncio
    async def test_update_note_title(self, db_config, mock_pool, sample_user_id):
        """update should update note title."""
        note_id = uuid.uuid4()

        async def mock_fetchrow(query, *args):
            return MockRecord({"id": note_id})

        mock_pool.fetchrow = mock_fetchrow

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.UPDATE,
            user_id=sample_user_id,
            note_id=str(note_id),
            title="Updated Title"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["updated"] is True

    @pytest.mark.asyncio
    async def test_update_note_content(self, db_config, mock_pool, sample_user_id):
        """update should update note content."""
        note_id = uuid.uuid4()

        async def mock_fetchrow(query, *args):
            return MockRecord({"id": note_id})

        mock_pool.fetchrow = mock_fetchrow

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.UPDATE,
            user_id=sample_user_id,
            note_id=str(note_id),
            content="Updated content"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["updated"] is True

    @pytest.mark.asyncio
    async def test_update_note_not_found(self, db_config, mock_pool, sample_user_id):
        """update should error for non-existent note."""
        async def mock_fetchrow(query, *args):
            return None

        mock_pool.fetchrow = mock_fetchrow

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.UPDATE,
            user_id=sample_user_id,
            note_id=str(uuid.uuid4()),
            title="New Title"
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_update_note_no_changes(self, db_config, mock_pool, sample_user_id):
        """update should error without title or content."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.UPDATE,
            user_id=sample_user_id,
            note_id=str(uuid.uuid4())
        )

        assert result.is_error is True
        assert "at least one" in result.content.lower()


class TestNotesToolDelete:
    """Tests for delete action."""

    @pytest.mark.asyncio
    async def test_delete_note(self, db_config, mock_pool, sample_user_id):
        """delete should remove note."""
        note_id = uuid.uuid4()

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.DELETE,
            user_id=sample_user_id,
            note_id=str(note_id)
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["deleted"] is True
        assert data["id"] == str(note_id)

    @pytest.mark.asyncio
    async def test_delete_note_not_found(self, db_config, mock_pool, sample_user_id):
        """delete should error for non-existent note."""
        mock_pool._mock_execute = lambda q, a: "DELETE 0"

        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.DELETE,
            user_id=sample_user_id,
            note_id=str(uuid.uuid4())
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_delete_note_missing_id(self, db_config, mock_pool, sample_user_id):
        """delete should error without note_id."""
        tool = NotesTool(db_config, pool=mock_pool)
        tool._schema_ensured = True

        result = await tool.acall(
            action=NotesAction.DELETE,
            user_id=sample_user_id
        )

        assert result.is_error is True
        assert "note_id is required" in result.content.lower()
