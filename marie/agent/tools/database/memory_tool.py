"""Memory tool for file-like persistent storage."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.tools import ToolMetadata, ToolOutput
from marie.agent.tools.database.base import AsyncDatabaseTool
from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class MemoryAction(str, Enum):
    """Available actions for MemoryTool."""

    VIEW = "view"
    CREATE = "create"
    STR_REPLACE = "str_replace"
    INSERT = "insert"
    DELETE = "delete"
    RENAME = "rename"


class MemoryInput(BaseModel):
    """Input schema for MemoryTool."""

    action: MemoryAction = Field(
        ...,
        description="Action: view, create, str_replace, insert, delete, or rename",
    )
    user_id: str = Field(
        ...,
        description="User identifier for scoping data",
    )
    path: str = Field(
        ...,
        description="File or directory path (e.g., '/notes/todo.txt' or '/projects/')",
    )
    content: Optional[str] = Field(
        None,
        description="Content for create action",
    )
    old_str: Optional[str] = Field(
        None,
        description="String to find for str_replace action",
    )
    new_str: Optional[str] = Field(
        None,
        description="Replacement string for str_replace action",
    )
    insert_line: Optional[int] = Field(
        None,
        description="Line number to insert at (1-based) for insert action",
    )
    new_path: Optional[str] = Field(
        None,
        description="New path for rename action",
    )


class MemoryTool(AsyncDatabaseTool):
    """File-like persistent storage tool with user scoping.

    Provides a virtual filesystem backed by PostgreSQL where each user
    has isolated storage. Supports files and directories with path-based
    navigation.

    Actions:
        - view: View file content or list directory contents
        - create: Create a new file with content
        - str_replace: Replace text in a file
        - insert: Insert text at a specific line
        - delete: Delete a file or directory
        - rename: Rename/move a file or directory

    Path format:
        - Must start with '/'
        - Directories end with '/'
        - Example: '/notes/todo.txt', '/projects/'
    """

    TABLE_NAME = "user_memory"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="memory",
            description=(
                "Persistent file-like storage. Actions: view (read file/list dir), "
                "create (new file), str_replace (edit), insert (add line), "
                "delete, rename. Paths start with '/'."
            ),
            fn_schema=MemoryInput,
        )

    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        """Create the user_memory table."""
        await pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.TABLE_NAME} (
                user_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content TEXT,
                is_directory BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, path)
            )
        """
        )
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_memory_user_path
            ON {self.SCHEMA}.{self.TABLE_NAME} (user_id, path text_pattern_ops)
        """
        )

    async def acall(
        self,
        action: MemoryAction,
        user_id: str,
        path: str,
        content: Optional[str] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        new_path: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """Execute the requested action."""
        raw_input = {
            "action": action,
            "user_id": user_id,
            "path": path,
            "content": content,
            "old_str": old_str,
            "new_str": new_str,
            "insert_line": insert_line,
            "new_path": new_path,
        }

        if not path.startswith("/"):
            return self._error_output(raw_input, "Path must start with '/'")

        await self._ensure_schema()
        pool = await self._get_pool()

        action_handlers = {
            MemoryAction.VIEW: lambda: self._view(pool, user_id, path, raw_input),
            MemoryAction.CREATE: lambda: self._create(
                pool, user_id, path, content, raw_input
            ),
            MemoryAction.STR_REPLACE: lambda: self._str_replace(
                pool, user_id, path, old_str, new_str, raw_input
            ),
            MemoryAction.INSERT: lambda: self._insert(
                pool, user_id, path, insert_line, content, raw_input
            ),
            MemoryAction.DELETE: lambda: self._delete(pool, user_id, path, raw_input),
            MemoryAction.RENAME: lambda: self._rename(
                pool, user_id, path, new_path, raw_input
            ),
        }

        handler = action_handlers.get(action)
        if handler:
            return await handler()
        return self._error_output(raw_input, f"Unknown action: {action}")

    async def _view(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """View file content or list directory."""
        if path.endswith("/"):
            return await self._list_directory(pool, user_id, path, raw_input)
        else:
            return await self._view_file(pool, user_id, path, raw_input)

    async def _view_file(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """View a file's content."""
        row = await pool.fetchrow(
            f"SELECT content, is_directory, updated_at FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if not row:
            return self._error_output(raw_input, f"File not found: {path}")

        if row["is_directory"]:
            return self._error_output(raw_input, f"Path is a directory: {path}")

        return self._create_output(
            raw_input,
            {
                "path": path,
                "content": row["content"],
                "updated_at": row["updated_at"].isoformat(),
            },
        )

    async def _list_directory(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """List directory contents."""
        pattern = path if path == "/" else path
        pattern_like = pattern + "%"

        rows = await pool.fetch(
            f"SELECT path, is_directory, updated_at FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path LIKE $2 AND path != $3",
            user_id,
            pattern_like,
            path,
        )

        entries = []
        seen = set()
        for row in rows:
            rel_path = row["path"][len(path) :]
            if "/" in rel_path:
                dir_name = rel_path.split("/")[0] + "/"
                if dir_name not in seen:
                    seen.add(dir_name)
                    entries.append({"name": dir_name, "type": "directory"})
            else:
                entry_type = "directory" if row["is_directory"] else "file"
                entries.append(
                    {
                        "name": rel_path,
                        "type": entry_type,
                        "updated_at": row["updated_at"].isoformat(),
                    }
                )

        return self._create_output(
            raw_input,
            {
                "path": path,
                "entries": sorted(
                    entries, key=lambda e: (e["type"] != "directory", e["name"])
                ),
                "count": len(entries),
            },
        )

    async def _create(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        content: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Create a new file or directory."""
        if path.endswith("/"):
            return await self._create_directory(pool, user_id, path, raw_input)

        if content is None:
            content = ""

        existing = await pool.fetchrow(
            f"SELECT path FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if existing:
            return self._error_output(raw_input, f"File already exists: {path}")

        await self._ensure_parent_dirs(pool, user_id, path)

        now = datetime.now(timezone.utc)
        await pool.execute(
            f"INSERT INTO {self.SCHEMA}.{self.TABLE_NAME} "
            "(user_id, path, content, is_directory, created_at, updated_at) "
            "VALUES ($1, $2, $3, FALSE, $4, $4)",
            user_id,
            path,
            content,
            now,
        )

        return self._create_output(
            raw_input,
            {
                "created": path,
                "size": len(content),
            },
        )

    async def _create_directory(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Create a directory."""
        existing = await pool.fetchrow(
            f"SELECT path FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if existing:
            return self._error_output(raw_input, f"Directory already exists: {path}")

        await self._ensure_parent_dirs(pool, user_id, path)

        now = datetime.now(timezone.utc)
        await pool.execute(
            f"INSERT INTO {self.SCHEMA}.{self.TABLE_NAME} "
            "(user_id, path, content, is_directory, created_at, updated_at) "
            "VALUES ($1, $2, NULL, TRUE, $3, $3)",
            user_id,
            path,
            now,
        )

        return self._create_output(raw_input, {"created": path, "type": "directory"})

    async def _ensure_parent_dirs(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
    ) -> None:
        """Ensure all parent directories exist."""
        parts = path.strip("/").split("/")
        if not path.endswith("/"):
            parts = parts[:-1]

        current = "/"
        now = datetime.now(timezone.utc)

        for part in parts:
            current = current + part + "/"
            await pool.execute(
                f"INSERT INTO {self.SCHEMA}.{self.TABLE_NAME} "
                "(user_id, path, content, is_directory, created_at, updated_at) "
                "VALUES ($1, $2, NULL, TRUE, $3, $3) "
                "ON CONFLICT (user_id, path) DO NOTHING",
                user_id,
                current,
                now,
            )

    async def _str_replace(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        old_str: Optional[str],
        new_str: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Replace text in a file."""
        if old_str is None:
            return self._error_output(raw_input, "old_str is required for str_replace")
        if new_str is None:
            new_str = ""

        row = await pool.fetchrow(
            f"SELECT content, is_directory FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if not row:
            return self._error_output(raw_input, f"File not found: {path}")
        if row["is_directory"]:
            return self._error_output(raw_input, f"Cannot edit directory: {path}")

        content = row["content"] or ""
        if old_str not in content:
            return self._error_output(
                raw_input, f"String not found in file: {old_str[:50]}..."
            )

        new_content = content.replace(old_str, new_str)
        now = datetime.now(timezone.utc)

        await pool.execute(
            f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
            "SET content = $1, updated_at = $2 "
            "WHERE user_id = $3 AND path = $4",
            new_content,
            now,
            user_id,
            path,
        )

        return self._create_output(
            raw_input,
            {
                "path": path,
                "replacements": content.count(old_str),
                "new_size": len(new_content),
            },
        )

    async def _insert(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        insert_line: Optional[int],
        content: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Insert text at a specific line."""
        if insert_line is None:
            return self._error_output(raw_input, "insert_line is required for insert")
        if content is None:
            return self._error_output(raw_input, "content is required for insert")

        row = await pool.fetchrow(
            f"SELECT content, is_directory FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if not row:
            return self._error_output(raw_input, f"File not found: {path}")
        if row["is_directory"]:
            return self._error_output(raw_input, f"Cannot edit directory: {path}")

        existing = row["content"] or ""
        lines = existing.split("\n")

        if insert_line < 1:
            insert_line = 1
        if insert_line > len(lines) + 1:
            insert_line = len(lines) + 1

        lines.insert(insert_line - 1, content)
        new_content = "\n".join(lines)
        now = datetime.now(timezone.utc)

        await pool.execute(
            f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
            "SET content = $1, updated_at = $2 "
            "WHERE user_id = $3 AND path = $4",
            new_content,
            now,
            user_id,
            path,
        )

        return self._create_output(
            raw_input,
            {
                "path": path,
                "inserted_at_line": insert_line,
                "new_line_count": len(lines),
            },
        )

    async def _delete(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Delete a file or directory."""
        row = await pool.fetchrow(
            f"SELECT is_directory FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if not row:
            return self._error_output(raw_input, f"Path not found: {path}")

        if row["is_directory"]:
            result = await pool.execute(
                f"DELETE FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND path LIKE $2",
                user_id,
                path + "%",
            )
        else:
            result = await pool.execute(
                f"DELETE FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND path = $2",
                user_id,
                path,
            )

        return self._create_output(raw_input, {"deleted": path, "status": result})

    async def _rename(
        self,
        pool: AsyncPostgresPool,
        user_id: str,
        path: str,
        new_path: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Rename or move a file/directory."""
        if not new_path:
            return self._error_output(raw_input, "new_path is required for rename")
        if not new_path.startswith("/"):
            return self._error_output(raw_input, "new_path must start with '/'")

        row = await pool.fetchrow(
            f"SELECT is_directory FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            path,
        )

        if not row:
            return self._error_output(raw_input, f"Path not found: {path}")

        existing = await pool.fetchrow(
            f"SELECT path FROM {self.SCHEMA}.{self.TABLE_NAME} "
            "WHERE user_id = $1 AND path = $2",
            user_id,
            new_path,
        )

        if existing:
            return self._error_output(
                raw_input, f"Destination already exists: {new_path}"
            )

        await self._ensure_parent_dirs(pool, user_id, new_path)

        if row["is_directory"]:
            rows = await pool.fetch(
                f"SELECT path FROM {self.SCHEMA}.{self.TABLE_NAME} "
                "WHERE user_id = $1 AND path LIKE $2",
                user_id,
                path + "%",
            )
            for r in rows:
                old_p = r["path"]
                new_p = new_path + old_p[len(path) :]
                await pool.execute(
                    f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
                    "SET path = $1 WHERE user_id = $2 AND path = $3",
                    new_p,
                    user_id,
                    old_p,
                )
        else:
            await pool.execute(
                f"UPDATE {self.SCHEMA}.{self.TABLE_NAME} "
                "SET path = $1 WHERE user_id = $2 AND path = $3",
                new_path,
                user_id,
                path,
            )

        return self._create_output(
            raw_input,
            {
                "renamed": path,
                "to": new_path,
            },
        )
