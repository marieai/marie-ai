"""Notes tool for user-scoped note management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from marie.agent.tools import ToolMetadata, ToolOutput
from marie.agent.tools.database.base import AsyncDatabaseTool
from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class NotesAction(str, Enum):
    """Available actions for NotesTool."""

    LIST = "list"
    GET = "get"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class NotesInput(BaseModel):
    """Input schema for NotesTool."""

    action: NotesAction = Field(
        ...,
        description="Action: list, get, create, update, or delete",
    )
    user_id: str = Field(
        ...,
        description="User identifier for scoping notes",
    )
    note_id: Optional[str] = Field(
        None,
        description="Note UUID (required for get, update, delete)",
    )
    title: Optional[str] = Field(
        None,
        description="Note title (required for create, optional for update)",
    )
    content: Optional[str] = Field(
        None,
        description="Note content",
    )
    search: Optional[str] = Field(
        None,
        description="Search term for list action (searches title and content)",
    )


class NotesTool(AsyncDatabaseTool):
    """User-scoped notes management tool.

    Provides CRUD operations for notes with title and content.
    Each note has a UUID and is scoped to a specific user.

    Actions:
        - list: List all notes for a user (with optional search)
        - get: Get a specific note by ID
        - create: Create a new note
        - update: Update an existing note
        - delete: Delete a note
    """

    TABLE_NAME = "user_notes"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="notes",
            description=(
                "Manage user notes with CRUD operations. "
                "Actions: list, get, create, update, delete."
            ),
            fn_schema=NotesInput,
        )

    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        """Create the user_notes table."""
        await pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.TABLE_NAME} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """
        )
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_notes_user_id
            ON {self.SCHEMA}.{self.TABLE_NAME} (user_id)
        """
        )
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_notes_user_updated
            ON {self.SCHEMA}.{self.TABLE_NAME} (user_id, updated_at DESC)
        """
        )

    async def acall(
        self,
        action: NotesAction,
        user_id: str,
        note_id: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        search: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """Execute the requested action."""
        raw_input = {
            "action": action,
            "user_id": user_id,
            "note_id": note_id,
            "title": title,
            "content": content,
            "search": search,
        }

        await self._ensure_schema()
        pool = await self._get_pool()

        action_handlers = {
            NotesAction.LIST: lambda: self._list(pool, user_id, search, raw_input),
            NotesAction.GET: lambda: self._get(pool, user_id, note_id, raw_input),
            NotesAction.CREATE: lambda: self._create(
                pool, user_id, title, content, raw_input
            ),
            NotesAction.UPDATE: lambda: self._update(
                pool, user_id, note_id, title, content, raw_input
            ),
            NotesAction.DELETE: lambda: self._delete(pool, user_id, note_id, raw_input),
        }

        handler = action_handlers.get(action)
        if handler:
            return await handler()
        return self._error_output(raw_input, f"Unknown action: {action}")

    async def _list(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        search: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """List notes for a user."""
        if search:
            search_pattern = f"%{search}%"
            rows = await pool.fetch(
                f"SELECT id, title, content, created_at, updated_at "
                f"FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND (title ILIKE $2 OR content ILIKE $2) "
                "ORDER BY updated_at DESC",
                user_id,
                search_pattern,
            )
        else:
            rows = await pool.fetch(
                f"SELECT id, title, content, created_at, updated_at "
                f"FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 ORDER BY updated_at DESC",
                user_id,
            )

        notes = [
            {
                "id": str(row["id"]),
                "title": row["title"],
                "preview": (row["content"] or "")[:100]
                + ("..." if row["content"] and len(row["content"]) > 100 else ""),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

        return self._create_output(
            raw_input,
            {
                "notes": notes,
                "count": len(notes),
                "search": search,
            },
        )

    async def _get(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        note_id: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Get a specific note."""
        if not note_id:
            return self._error_output(raw_input, "note_id is required for get")

        try:
            note_uuid = uuid.UUID(note_id)
        except ValueError:
            return self._error_output(raw_input, f"Invalid note_id format: {note_id}")

        row = await pool.fetchrow(
            f"SELECT id, title, content, created_at, updated_at "
            f"FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND id = $2",
            user_id,
            note_uuid,
        )

        if not row:
            return self._error_output(raw_input, f"Note not found: {note_id}")

        return self._create_output(
            raw_input,
            {
                "id": str(row["id"]),
                "title": row["title"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            },
        )

    async def _create(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        title: Optional[str],
        content: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Create a new note."""
        if not title:
            return self._error_output(raw_input, "title is required for create")

        now = datetime.now(timezone.utc)
        note_id = uuid.uuid4()

        await pool.execute(
            f"INSERT INTO {self.SCHEMA}.{self.TABLE_NAME} "
            "(id, user_id, title, content, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4, $5, $5)",
            note_id,
            user_id,
            title,
            content or "",
            now,
        )

        return self._create_output(
            raw_input,
            {
                "id": str(note_id),
                "title": title,
                "created_at": now.isoformat(),
            },
        )

    async def _update(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        note_id: Optional[str],
        title: Optional[str],
        content: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Update an existing note."""
        if not note_id:
            return self._error_output(raw_input, "note_id is required for update")

        if title is None and content is None:
            return self._error_output(
                raw_input, "At least one of title or content is required for update"
            )

        try:
            note_uuid = uuid.UUID(note_id)
        except ValueError:
            return self._error_output(raw_input, f"Invalid note_id format: {note_id}")

        existing = await pool.fetchrow(
            f"SELECT id FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND id = $2",
            user_id,
            note_uuid,
        )

        if not existing:
            return self._error_output(raw_input, f"Note not found: {note_id}")

        now = datetime.now(timezone.utc)
        updates = ["updated_at = $1"]
        params = [now]
        param_idx = 2

        if title is not None:
            updates.append(f"title = ${param_idx}")
            params.append(title)
            param_idx += 1

        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1

        params.extend([user_id, note_uuid])

        await pool.execute(
            f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
            f"SET {', '.join(updates)} "
            f"WHERE user_id = ${param_idx} AND id = ${param_idx + 1}",
            *params,
        )

        return self._create_output(
            raw_input,
            {
                "id": note_id,
                "updated": True,
                "updated_at": now.isoformat(),
            },
        )

    async def _delete(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        note_id: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Delete a note."""
        if not note_id:
            return self._error_output(raw_input, "note_id is required for delete")

        try:
            note_uuid = uuid.UUID(note_id)
        except ValueError:
            return self._error_output(raw_input, f"Invalid note_id format: {note_id}")

        result = await pool.execute(
            f"DELETE FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND id = $2",
            user_id,
            note_uuid,
        )

        deleted = "DELETE 1" in result

        if not deleted:
            return self._error_output(raw_input, f"Note not found: {note_id}")

        return self._create_output(
            raw_input,
            {
                "id": note_id,
                "deleted": True,
            },
        )
