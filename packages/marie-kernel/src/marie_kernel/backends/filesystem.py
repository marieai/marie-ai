"""
FileSystemStateBackend - Read annotator results from output directories.

This backend reads from the filesystem structure used by annotators:
    {base_path}/{dag_id}/agent-output/{task_id}/

It maps task_id to output directory patterns:
- tables → agent-output/tables/
- claims → agent-output/claims/
- remarks → agent-output/remarks/

This backend is primarily intended for quick deployment scenarios.
For production, consider using PostgresStateBackend instead.
"""

import json
import os
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from marie_kernel.ref import TaskInstanceRef


class FileSystemStateBackend:
    """
    State backend that reads annotator results from output directories.

    The filesystem structure follows the pattern used by annotators:
        {base_path}/{dag_id}/agent-output/{task_id}/

    This backend supports reading JSON files from task output directories
    and aggregating results across pages.

    Example:
        ```python
        backend = FileSystemStateBackend(base_path="/path/to/working/dir")
        ti = TaskInstanceRef(
            tenant_id="default",
            dag_name="annotation",
            dag_id="job_123",
            task_id="remarks",
            try_number=1,
        )

        # Pull results from upstream 'tables' task
        tables_results = backend.pull(ti, "ANNOTATOR_RESULTS", from_tasks=["tables"])
        ```
    """

    # Standard key used for annotator results
    ANNOTATOR_RESULTS_KEY = "ANNOTATOR_RESULTS"

    def __init__(self, base_path: str) -> None:
        """
        Initialize FileSystemStateBackend.

        Args:
            base_path: Base path for the working directory. This should be
                      the root asset directory (e.g., ~/.marie/generators/{ref_type}/{ref_id})
        """
        self._base_path = os.path.expanduser(base_path)
        self._lock = Lock()
        # In-memory cache for pushed values (not persisted to disk in this version)
        self._memory_store: Dict[str, Any] = {}

    def push(
        self,
        ti: TaskInstanceRef,
        key: str,
        value: Any,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a value for the given task instance and key.

        For FileSystemStateBackend, this writes to the in-memory store.
        The actual file writes are handled by the annotators themselves.

        Args:
            ti: Task instance reference
            key: State key
            value: Value to store (should be JSON-serializable)
            metadata: Optional metadata dict
        """
        pk = self._make_key(ti, key)
        with self._lock:
            self._memory_store[pk] = {"value": value, "metadata": metadata or {}}

    def pull(
        self,
        ti: TaskInstanceRef,
        key: str,
        *,
        from_tasks: Optional[Iterable[str]] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value by key.

        If from_tasks is provided, searches those task IDs in order
        and returns the first match. For ANNOTATOR_RESULTS key, this
        reads from the filesystem.

        Args:
            ti: Task instance reference (provides dag context)
            key: State key to retrieve
            from_tasks: Optional list of task IDs to search
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found
        """
        task_ids = list(from_tasks) if from_tasks else [ti.task_id]

        for task_id in task_ids:
            # Check memory store first
            pk = self._make_key_with_task(ti, task_id, key)
            with self._lock:
                if pk in self._memory_store:
                    return self._memory_store[pk]["value"]

            # For annotator results, read from filesystem
            if key == self.ANNOTATOR_RESULTS_KEY:
                results = self._read_task_output(task_id)
                if results is not None:
                    return results

        return default

    def clear_for_task(self, ti: TaskInstanceRef) -> None:
        """
        Clear all state for a task instance.

        For filesystem backend, this clears the in-memory cache.
        It does NOT delete files from the filesystem.

        Args:
            ti: Task instance reference identifying the task to clear
        """
        prefix = f"{ti.tenant_id}:{ti.dag_name}:{ti.dag_id}:{ti.task_id}:"
        with self._lock:
            keys_to_delete = [k for k in self._memory_store if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._memory_store[k]

    def get_all_for_task(self, ti: TaskInstanceRef) -> Dict[str, Any]:
        """
        Retrieve all state for a task instance (debugging helper).

        Args:
            ti: Task instance reference

        Returns:
            Dictionary of {key: value} for all state belonging to this task
        """
        result = {}
        prefix = (
            f"{ti.tenant_id}:{ti.dag_name}:{ti.dag_id}:{ti.task_id}:{ti.try_number}:"
        )
        with self._lock:
            for pk, data in self._memory_store.items():
                if pk.startswith(prefix):
                    # Extract key from the primary key
                    key = pk[len(prefix) :]
                    result[key] = data["value"]

        # Also include filesystem results
        fs_results = self._read_task_output(ti.task_id)
        if fs_results is not None:
            result[self.ANNOTATOR_RESULTS_KEY] = fs_results

        return result

    def _make_key(self, ti: TaskInstanceRef, key: str) -> str:
        """Create a unique key for the memory store."""
        return f"{ti.tenant_id}:{ti.dag_name}:{ti.dag_id}:{ti.task_id}:{ti.try_number}:{key}"

    def _make_key_with_task(self, ti: TaskInstanceRef, task_id: str, key: str) -> str:
        """Create a unique key for the memory store with a specific task_id."""
        return (
            f"{ti.tenant_id}:{ti.dag_name}:{ti.dag_id}:{task_id}:{ti.try_number}:{key}"
        )

    def _read_task_output(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Read annotator output from the filesystem.

        The output directory structure is:
            {base_path}/agent-output/{task_id}/

        Files in this directory can be:
        - *.json - Parsed JSON results
        - *.md - Markdown results
        - *_prompt.txt - Prompt files (ignored)

        Args:
            task_id: The task ID (annotator name)

        Returns:
            Dictionary with aggregated results by page, or None if not found
        """
        output_dir = os.path.join(self._base_path, "agent-output", task_id)

        if not os.path.exists(output_dir):
            return None

        results: Dict[str, Any] = {
            "task_id": task_id,
            "output_dir": output_dir,
            "pages": {},
            "raw_files": [],
        }

        try:
            files = os.listdir(output_dir)
        except OSError:
            return None

        for filename in sorted(files):
            filepath = os.path.join(output_dir, filename)
            if not os.path.isfile(filepath):
                continue

            # Skip prompt files
            if filename.endswith("_prompt.txt"):
                continue

            # Extract page number from filename (e.g., "frame_0001.json" → 1)
            page_number = self._extract_page_number(filename)

            if filename.endswith(".json"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    if page_number is not None:
                        results["pages"][page_number] = {
                            "type": "json",
                            "data": json_data,
                            "file": filename,
                            "path": filepath,
                        }
                    results["raw_files"].append(
                        {
                            "file": filename,
                            "path": filepath,
                            "type": "json",
                            "page": page_number,
                        }
                    )
                except (json.JSONDecodeError, IOError):
                    pass

            elif filename.endswith(".md"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    if page_number is not None:
                        results["pages"][page_number] = {
                            "type": "markdown",
                            "data": md_content,
                            "file": filename,
                            "path": filepath,
                        }
                    results["raw_files"].append(
                        {
                            "file": filename,
                            "path": filepath,
                            "type": "markdown",
                            "page": page_number,
                        }
                    )
                except IOError:
                    pass

        if not results["pages"] and not results["raw_files"]:
            return None

        return results

    def _extract_page_number(self, filename: str) -> Optional[int]:
        """
        Extract page number from filename.

        Expected patterns:
        - frame_0001.json → 1
        - frame_0001.png.json → 1 (handles double extension)
        - page_1.json → 1

        Args:
            filename: The filename to parse

        Returns:
            Page number (1-indexed) or None if not found
        """
        import re

        # Remove known extensions
        base = filename
        for ext in [".json", ".md", ".png", ".tif", ".txt"]:
            if base.endswith(ext):
                base = base[: -len(ext)]

        # Try to extract number from the end
        # Match patterns like: frame_0001, page_1, 0001, etc.
        match = re.search(r"_?(\d+)$", base)
        if match:
            return int(match.group(1))

        return None

    def __len__(self) -> int:
        """Return total number of stored entries in memory."""
        with self._lock:
            return len(self._memory_store)

    def clear(self) -> None:
        """Clear all stored state in memory (useful for test cleanup)."""
        with self._lock:
            self._memory_store.clear()

    def list_annotations(self) -> List[str]:
        """
        List available annotations by scanning agent-output directory.

        Returns:
            List of annotation names (subdirectory names under agent-output/)
        """
        agent_output_dir = os.path.join(self._base_path, "agent-output")

        if not os.path.exists(agent_output_dir):
            return []

        annotations = []
        try:
            for entry in os.listdir(agent_output_dir):
                entry_path = os.path.join(agent_output_dir, entry)
                if os.path.isdir(entry_path):
                    annotations.append(entry)
        except OSError:
            return []

        return sorted(annotations)
