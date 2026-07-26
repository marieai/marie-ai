from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).parent
SOURCE_BUILD_INFO = ROOT / "marie" / "build_info.json"
VERSION_FILE = ROOT / "marie" / "_version.py"
UNKNOWN = "unknown"


def _stored_build_info() -> dict[str, str]:
    try:
        payload = json.loads(SOURCE_BUILD_INFO.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _package_version() -> str:
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        VERSION_FILE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Unable to read version from {VERSION_FILE}")
    return match.group(1)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN


def _first_value(*values: str | None) -> str:
    return next((value for value in values if value), UNKNOWN)


def _build_info() -> dict[str, str]:
    stored = _stored_build_info()
    return {
        "version": _first_value(
            os.getenv("MARIE_VERSION"), stored.get("version"), _package_version()
        ),
        "git_commit": _first_value(
            os.getenv("MARIE_GIT_COMMIT"),
            os.getenv("VCS_REF"),
            stored.get("git_commit"),
            _git_commit(),
        ),
        "build_time": _first_value(
            os.getenv("MARIE_BUILD_DATE"),
            os.getenv("BUILD_DATE"),
            stored.get("build_time"),
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ),
        "build_number": _first_value(
            os.getenv("MARIE_BUILD_NUMBER"),
            os.getenv("GITHUB_RUN_NUMBER"),
            stored.get("build_number"),
        ),
        "image": _first_value(
            os.getenv("MARIE_IMAGE"),
            os.getenv("IMAGE_NAME"),
            stored.get("image"),
        ),
        "image_digest": _first_value(
            os.getenv("MARIE_IMAGE_DIGEST"), stored.get("image_digest")
        ),
    }


def _write_build_info(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_build_info(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class _BuildPyWithBuildInfo(build_py):
    def run(self) -> None:
        super().run()
        _write_build_info(Path(self.build_lib) / "marie" / "build_info.json")


class _SdistWithBuildInfo(sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        _write_build_info(Path(base_dir) / "marie" / "build_info.json")


setup(
    cmdclass={
        "build_py": _BuildPyWithBuildInfo,
        "sdist": _SdistWithBuildInfo,
    }
)
