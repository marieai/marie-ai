"""Async file utilities."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import aiofiles


async def save_json_result(
    data: Dict[str, Any], category: str, source_file: str
) -> str:
    """
    Save JSON result to output directory.

    Args:
        data: Data to save as JSON
        category: Category subdirectory (e.g., 'document_extraction')
        source_file: Original source file path (used for naming)

    Returns:
        Path to saved file
    """
    from ..config import Config

    # Create category directory
    category_dir = Config.OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    source_name = Path(source_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{source_name}_{timestamp}.json"
    output_path = category_dir / filename

    # Save JSON
    async with aiofiles.open(output_path, "w") as f:
        await f.write(json.dumps(data, indent=2))

    return str(output_path)


async def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read JSON file asynchronously.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    async with aiofiles.open(file_path, "r") as f:
        content = await f.read()
        return json.loads(content)


async def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists (create if needed).

    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)
