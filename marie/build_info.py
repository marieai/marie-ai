from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from marie._version import __version__

BUILD_INFO_PATH_ENV = "MARIE_BUILD_INFO_PATH"
DEFAULT_BUILD_INFO_PATH = Path("/etc/marie-ai/build-info.json")
PACKAGE_BUILD_INFO_PATH = Path(__file__).with_name("build_info.json")
SOURCE_ROOT = Path(__file__).resolve().parent.parent
UNKNOWN = "unknown"


def write_build_info(
    path: str | Path,
    *,
    version: str,
    git_commit: str,
    build_time: str,
    build_number: str,
    image: str,
    image_digest: str = UNKNOWN,
) -> None:
    """Write the build identity consumed by Marie runtimes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "git_commit": git_commit,
        "build_time": build_time,
        "build_number": build_number,
        "image": image,
        "image_digest": image_digest,
    }
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_build_info(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            payload = stored
    except (FileNotFoundError, IsADirectoryError, OSError, json.JSONDecodeError):
        pass
    return payload


def _source_git_commit() -> str:
    if not (SOURCE_ROOT / ".git").exists():
        return UNKNOWN
    try:
        result = subprocess.run(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN


def _default_build_info() -> dict[str, Any]:
    for candidate in (DEFAULT_BUILD_INFO_PATH, PACKAGE_BUILD_INFO_PATH):
        payload = _read_build_info(candidate)
        if payload:
            return payload
    return {
        "version": __version__,
        "git_commit": _source_git_commit(),
        "build_number": "source",
    }


def get_build_info(path: str | Path | None = None) -> dict[str, str]:
    """Return the build identity for the running Marie installation."""
    configured_path = path or os.getenv(BUILD_INFO_PATH_ENV)
    payload = (
        _read_build_info(Path(configured_path))
        if configured_path
        else _default_build_info()
    )

    stored_version = str(payload.get("version") or UNKNOWN)
    version = __version__ if stored_version == UNKNOWN else stored_version
    commit = str(payload.get("git_commit") or UNKNOWN)
    image_digest = os.getenv("MARIE_IMAGE_DIGEST") or payload.get("image_digest")
    return {
        "version": version,
        "git_commit": commit,
        "git_commit_short": commit[:12] if commit != UNKNOWN else UNKNOWN,
        "build_time": str(payload.get("build_time") or UNKNOWN),
        "build_number": str(payload.get("build_number") or UNKNOWN),
        "image": str(payload.get("image") or UNKNOWN),
        "image_digest": str(image_digest or UNKNOWN),
    }


def format_build_identity(service: str) -> str:
    """Format a searchable one-line startup identity for a Marie service."""
    info = get_build_info()
    return (
        f"Marie-AI build service={service} version={info['version']} "
        f"commit={info['git_commit_short']} build={info['build_number']} "
        f"built_at={info['build_time']} image={info['image']} "
        f"image_digest={info['image_digest']}"
    )
