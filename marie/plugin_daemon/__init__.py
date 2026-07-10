"""Shared marie-plugin-daemon protocol: discovery, envelopes, frame parsing.

Everything needed to talk to the marie plugin daemon from any component —
executors, agent tools, gateway — without importing another subsystem.
Defaults (``DEFAULT_DAEMON_ADDR``/``DEFAULT_DAEMON_BIN``/``DEFAULT_TENANT_UUID``)
live in ``marie.constants``.
"""

from marie.plugin_daemon.discovery import DaemonDiscovery, discover_daemon
from marie.plugin_daemon.envelope import (
    DEFAULT_RUNTIME_POLICY,
    build_invocation_envelope,
    canonical_runtime_envelope,
    sign_envelope,
)
from marie.plugin_daemon.frames import parse_daemon_frames, runtime_error_frame

__all__ = [
    "DaemonDiscovery",
    "DEFAULT_RUNTIME_POLICY",
    "build_invocation_envelope",
    "canonical_runtime_envelope",
    "discover_daemon",
    "parse_daemon_frames",
    "runtime_error_frame",
    "sign_envelope",
]
