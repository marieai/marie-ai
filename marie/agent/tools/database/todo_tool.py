"""Todo tool for user-scoped task management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from marie.agent.tools import ToolMetadata, ToolOutput
from marie.agent.tools.database.base import AsyncDatabaseTool
from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class TodoAction(str, Enum):
    """Available actions for TodoTool."""

    LIST = "list"
    ADD = "add"
    COMPLETE = "complete"
    DELETE = "delete"
    CLEAR_COMPLETED = "clear_completed"


class TodoPriority(int, Enum):
    """Priority levels for todos."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TodoInput(BaseModel):
    """Input schema for TodoTool."""

    action: TodoAction = Field(
        ...,
        description="Action: list, add, complete, delete, or clear_completed",
    )
    user_id: str = Field(
        ...,
        description="User identifier for scoping todos",
    )
    todo_id: Optional[str] = Field(
        None,
        description="Todo UUID (required for complete, delete)",
    )
    title: Optional[str] = Field(
        None,
        description="Todo title (required for add)",
    )
    priority: Optional[int] = Field(
        2,
        description="Priority: 1 (high), 2 (medium), 3 (low). Default: 2",
        ge=1,
        le=3,
    )
    show_completed: Optional[bool] = Field(
        False,
        description="Include completed todos in list (default: False)",
    )


class TodoTool(AsyncDatabaseTool):
    """User-scoped todo/task management tool.

    Provides task list management with priorities and completion tracking.
    Each todo has a UUID and is scoped to a specific user.

    Actions:
        - list: List todos for a user (optionally include completed)
        - add: Add a new todo with optional priority
        - complete: Mark a todo as completed
        - delete: Delete a todo
        - clear_completed: Delete all completed todos

    Priority levels:
        - 1: High
        - 2: Medium (default)
        - 3: Low
    """

    TABLE_NAME = "user_todos"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="todo",
            description=(
                "Manage user todos/tasks. Actions: list, add, complete, delete, clear_completed. "
                "Priority: 1 (high), 2 (medium), 3 (low)."
            ),
            fn_schema=TodoInput,
        )

    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        """Create the user_todos table."""
        await pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.TABLE_NAME} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                completed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """
        )
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_todos_user_id
            ON {self.SCHEMA}.{self.TABLE_NAME} (user_id)
        """
        )
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_todos_user_priority
            ON {self.SCHEMA}.{self.TABLE_NAME} (user_id, completed, priority)
        """
        )

    async def acall(
        self,
        action: TodoAction,
        user_id: str,
        todo_id: Optional[str] = None,
        title: Optional[str] = None,
        priority: Optional[int] = 2,
        show_completed: Optional[bool] = False,
        **kwargs,
    ) -> ToolOutput:
        """Execute the requested action."""
        raw_input = {
            "action": action,
            "user_id": user_id,
            "todo_id": todo_id,
            "title": title,
            "priority": priority,
            "show_completed": show_completed,
        }

        await self._ensure_schema()
        pool = await self._get_pool()

        action_handlers = {
            TodoAction.LIST: lambda: self._list(
                pool, user_id, show_completed, raw_input
            ),
            TodoAction.ADD: lambda: self._add(
                pool, user_id, title, priority, raw_input
            ),
            TodoAction.COMPLETE: lambda: self._complete(
                pool, user_id, todo_id, raw_input
            ),
            TodoAction.DELETE: lambda: self._delete(pool, user_id, todo_id, raw_input),
            TodoAction.CLEAR_COMPLETED: lambda: self._clear_completed(
                pool, user_id, raw_input
            ),
        }

        handler = action_handlers.get(action)
        if handler:
            return await handler()
        return self._error_output(raw_input, f"Unknown action: {action}")

    async def _list(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        show_completed: Optional[bool],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """List todos for a user."""
        if show_completed:
            rows = await pool.fetch(
                f"SELECT id, title, priority, completed, created_at "
                f"FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 "
                "ORDER BY completed, priority, created_at",
                user_id,
            )
        else:
            rows = await pool.fetch(
                f"SELECT id, title, priority, completed, created_at "
                f"FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND completed = FALSE "
                "ORDER BY priority, created_at",
                user_id,
            )

        priority_names = {1: "high", 2: "medium", 3: "low"}
        todos = [
            {
                "id": str(row["id"]),
                "title": row["title"],
                "priority": priority_names.get(row["priority"], "medium"),
                "completed": row["completed"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

        pending_count = sum(1 for t in todos if not t["completed"])
        completed_count = sum(1 for t in todos if t["completed"])

        return self._create_output(
            raw_input,
            {
                "todos": todos,
                "count": len(todos),
                "pending": pending_count,
                "completed": completed_count,
            },
        )

    async def _add(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        title: Optional[str],
        priority: Optional[int],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Add a new todo."""
        if not title:
            return self._error_output(raw_input, "title is required for add")

        if priority is None:
            priority = 2
        priority = max(1, min(3, priority))

        now = datetime.now(timezone.utc)
        todo_id = uuid.uuid4()

        await pool.execute(
            f"INSERT INTO {self.SCHEMA}.{self.TABLE_NAME} "
            "(id, user_id, title, priority, completed, created_at) "
            "VALUES ($1, $2, $3, $4, FALSE, $5)",
            todo_id,
            user_id,
            title,
            priority,
            now,
        )

        priority_names = {1: "high", 2: "medium", 3: "low"}

        return self._create_output(
            raw_input,
            {
                "id": str(todo_id),
                "title": title,
                "priority": priority_names.get(priority, "medium"),
                "created_at": now.isoformat(),
            },
        )

    async def _complete(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        todo_id: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Mark a todo as completed."""
        if not todo_id:
            return self._error_output(raw_input, "todo_id is required for complete")

        try:
            todo_uuid = uuid.UUID(todo_id)
        except ValueError:
            return self._error_output(raw_input, f"Invalid todo_id format: {todo_id}")

        result = await pool.execute(
            f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
            "SET completed = TRUE "
            "WHERE user_id = $1 AND id = $2 AND completed = FALSE",
            user_id,
            todo_uuid,
        )

        updated = "UPDATE 1" in result

        if not updated:
            existing = await pool.fetchrow(
                f"SELECT completed FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND id = $2",
                user_id,
                todo_uuid,
            )
            if existing and existing["completed"]:
                return self._error_output(
                    raw_input, f"Todo already completed: {todo_id}"
                )
            return self._error_output(raw_input, f"Todo not found: {todo_id}")

        return self._create_output(
            raw_input,
            {
                "id": todo_id,
                "completed": True,
            },
        )

    async def _delete(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        todo_id: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Delete a todo."""
        if not todo_id:
            return self._error_output(raw_input, "todo_id is required for delete")

        try:
            todo_uuid = uuid.UUID(todo_id)
        except ValueError:
            return self._error_output(raw_input, f"Invalid todo_id format: {todo_id}")

        result = await pool.execute(
            f"DELETE FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND id = $2",
            user_id,
            todo_uuid,
        )

        deleted = "DELETE 1" in result

        if not deleted:
            return self._error_output(raw_input, f"Todo not found: {todo_id}")

        return self._create_output(
            raw_input,
            {
                "id": todo_id,
                "deleted": True,
            },
        )

    async def _clear_completed(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Delete all completed todos."""
        result = await pool.execute(
            f"DELETE FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND completed = TRUE",
            user_id,
        )

        count = 0
        if "DELETE" in result:
            parts = result.split()
            if len(parts) >= 2:
                try:
                    count = int(parts[1])
                except ValueError:
                    pass

        return self._create_output(
            raw_input,
            {
                "cleared": count,
            },
        )
