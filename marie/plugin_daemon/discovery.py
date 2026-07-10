"""Resolve how the current process reaches a marie-plugin-daemon.

Precedence: explicit URL (``MARIE_PLUGIN_DAEMON_URL``) → explicit binary
(``MARIE_PLUGIN_DAEMON_BIN``) → image binary → local dist build → ``PATH``.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from marie.constants import DEFAULT_DAEMON_BIN
from marie.plugin_daemon.frames import first_text


@dataclass(frozen=True)
class DaemonDiscovery:
    mode: str
    source: str
    url: str | None = None
    binary: str | None = None
    message: str | None = None


def discover_daemon(
    daemon_url: str | None,
    daemon_bin: str | None,
    daemon_addr: str,
    env: Mapping[str, str] | None = None,
) -> DaemonDiscovery:
    values = os.environ if env is None else env
    configured_url = first_text(daemon_url, values.get("MARIE_PLUGIN_DAEMON_URL"))
    if configured_url:
        return DaemonDiscovery(
            mode="sidecar_proxy", source="url", url=configured_url.rstrip("/")
        )

    configured_bin = first_text(daemon_bin, values.get("MARIE_PLUGIN_DAEMON_BIN"))
    if configured_bin:
        path = Path(configured_bin)
        if executable(path):
            return child_discovery(path, daemon_addr, "explicit_binary")
        return DaemonDiscovery(
            mode="unavailable",
            source="explicit_binary",
            message=f"Configured marie-plugin-daemon binary is not executable: {path}",
        )

    if executable(DEFAULT_DAEMON_BIN):
        return child_discovery(DEFAULT_DAEMON_BIN, daemon_addr, "image_binary")

    dev_binary = local_dist_binary()
    if dev_binary and executable(dev_binary):
        return child_discovery(dev_binary, daemon_addr, "local_dist")

    path_binary = shutil.which("marie-plugin-daemon", path=values.get("PATH"))
    if path_binary:
        return child_discovery(Path(path_binary), daemon_addr, "path")

    return DaemonDiscovery(
        mode="unavailable",
        source="not_found",
        message="marie-plugin-daemon is not configured and no binary was found",
    )


def child_discovery(binary: Path, daemon_addr: str, source: str) -> DaemonDiscovery:
    return DaemonDiscovery(
        mode="binary_child",
        source=source,
        url=f"http://{daemon_addr}",
        binary=str(binary),
    )


def local_dist_binary() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    dist = repo_root / "packages" / "marie-plugin-daemon" / "dist"
    # Prefer the platform-suffixed build, but fall back to a flat dist/ binary
    # (what `make build` produces: dist/marie-plugin-daemon).
    suffix = platform_suffix()
    candidates = []
    if suffix is not None:
        candidates.append(dist / suffix / "marie-plugin-daemon")
    candidates.append(dist / "marie-plugin-daemon")
    for candidate in candidates:
        if executable(candidate):
            return candidate
    return candidates[0] if candidates else None


def platform_suffix() -> str | None:
    os_name = {"linux": "linux", "darwin": "darwin"}.get(sys.platform)
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(platform.machine().lower())
    if not os_name or not arch:
        return None
    return f"{os_name}-{arch}"


def executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
