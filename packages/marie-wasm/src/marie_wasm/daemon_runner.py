"""Run a WASI-P2 `node` component as a daemon-managed plugin.

The plugin daemon spawns this module as `python -m marie_wasm.daemon_runner` in
the plugin WorkingDir (a normal `python_source` plugin from the daemon's point of
view). It speaks the daemon's stdio session protocol: a heartbeat plus `request`
envelopes in -> `stream`/`end` session messages out.

The daemon is the supervisor; this module owns component execution via Python
wasmtime, reusing `HostImplementations` (the single trusted host that enforces
permissions). This module is only the wasmtime-46 marshaling + stdio shim.

Proven recipe: packages/marie-plugin-daemon/_spike_wasm/SPIKE-FINDINGS.md.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from marie_wasm import BUILTIN_NODES_DIR, Permissions
from marie_wasm.host import DefaultHttpClient, HostImplementations

# Interface instance names are versioned by the WIT package (`marie:node@1.0.0`).
WIT_VERSION = "@1.0.0"

_emit_lock = threading.Lock()


@dataclass
class RunnerConfig:
    module_path: str
    permissions: Permissions
    credentials: dict[str, str] = field(default_factory=dict)


# ── configuration ────────────────────────────────────────────────────────────


def load_config(working_dir: str) -> RunnerConfig:
    """Read the plugin manifest to locate the .wasm component and its permissions."""
    from marie_wasm.types import BUILTIN_PERMISSIONS

    manifest = yaml.safe_load((Path(working_dir) / "marie-extension.yaml").read_text())
    runtime = manifest.get("runtime", {}) or {}
    module = runtime.get("module") or "node.wasm"
    module_path = Path(working_dir) / module
    if not module_path.exists():
        # Fall back to a built-in node referenced by name (node_type).
        node_type = runtime.get("node_type")
        if node_type:
            module_path = BUILTIN_NODES_DIR / f"{node_type}.wasm"

    preset = runtime.get("permissions")
    perms = BUILTIN_PERMISSIONS.get(preset, Permissions()) if preset else Permissions()
    return RunnerConfig(module_path=str(module_path), permissions=perms, credentials={})


# ── wasmtime wiring (proven recipe) ──────────────────────────────────────────


def _build_host(cfg: RunnerConfig) -> HostImplementations:
    return HostImplementations(
        permissions=cfg.permissions,
        credentials=cfg.credentials,
        http_client=DefaultHttpClient(),
        logger_func=lambda level, msg: _log(level, msg),
    )


def _register_host(root, host: HostImplementations):
    """Bind the five marie:node host imports onto the component linker root.

    Host funcs receive (store, *wit_args). We adapt wasmtime values <-> the
    plain-Python contract HostImplementations exposes.
    """
    from wasmtime.component import Record

    def fetch(store, req):
        # http-request record -> host dict; host {"ok":{...}}/{"error":s} -> Record/str
        request = {
            "method": getattr(req, "method", "GET"),
            "url": getattr(req, "url", ""),
            "headers": [tuple(h) for h in (getattr(req, "headers", []) or [])],
            "body": getattr(
                req, "body", None
            ),  # option<list<u8>> arrives as bytes/None
        }
        result = host.http_request(request)
        if "error" in result:
            return result["error"]  # untagged result<_, string> err arm
        ok = result["ok"]
        resp = Record()
        resp.status = ok["status"]
        resp.headers = [tuple(h) for h in ok.get("headers", [])]
        resp.body = bytes(ok.get("body", []))  # list<u8> == bytes
        return resp  # untagged ok arm

    def secrets_get(store, name):
        return host.secrets_get(name)  # option<string>: str/None

    def kv_get(store, key):
        v = host.kv_get(key)  # list[int]/None
        return bytes(v) if v is not None else None  # option<list<u8>>: bytes/None

    def kv_put(store, key, value, ttl):
        return host.kv_set(key, list(value), ttl)  # result<_, string>: None/str

    def kv_delete(store, key):
        return host.kv_delete(key)  # result<_, string>: None/str

    def log(store, level, message):
        host.log(level, message)
        return None

    def emit(store, event_type, payload):
        host.emit(event_type, payload)
        return None

    with root.add_instance(f"marie:node/http-client{WIT_VERSION}") as i:
        i.add_func("fetch", fetch)
    with root.add_instance(f"marie:node/secrets{WIT_VERSION}") as i:
        i.add_func("get", secrets_get)
    with root.add_instance(f"marie:node/kv{WIT_VERSION}") as i:
        i.add_func("get", kv_get)
        i.add_func("put", kv_put)
        i.add_func("delete", kv_delete)
    with root.add_instance(f"marie:node/console{WIT_VERSION}") as i:
        i.add_func("log", log)
    with root.add_instance(f"marie:node/events{WIT_VERSION}") as i:
        i.add_func("emit", emit)


def load_component(cfg: RunnerConfig):
    from wasmtime import Config, Engine
    from wasmtime.component import Component

    wcfg = Config()
    wcfg.wasm_component_model = True
    engine = Engine(wcfg)
    component = Component.from_file(engine, cfg.module_path)
    return engine, component


def run_execute(engine, component, cfg: RunnerConfig, payload: dict) -> dict:
    """Instantiate + call execute once. Returns {"items": [...]} or {"error": str}."""
    from wasmtime import Config, Store, WasiConfig
    from wasmtime.component import Linker, Record

    host = _build_host(cfg)
    store = Store(engine)
    store.set_wasi(WasiConfig())  # component uses WASI (serde -> wasi:random)
    linker = Linker(engine)
    linker.add_wasip2()
    with linker.root() as root:
        _register_host(root, host)

    instance = linker.instantiate(store, component)
    execute = instance.get_func(store, "execute")
    if execute is None:
        return {"error": "component does not export 'execute'"}

    # Build WIT args. item{json, binary}; env{vars}; context{workflow-id,...}.
    input_items = []
    for raw in payload.get("input") or []:
        item = Record()
        item.json = raw.get("json", "{}")
        b = raw.get("binary")
        item.binary = bytes(b) if b is not None else None
        input_items.append(item)
    env = Record()
    env.vars = payload.get("env", "{}")
    ctx_in = payload.get("ctx") or {}
    ctx = Record()
    setattr(ctx, "workflow-id", ctx_in.get("workflow_id", ""))
    setattr(ctx, "execution-id", ctx_in.get("execution_id", ""))
    setattr(ctx, "node-id", ctx_in.get("node_id", ""))
    setattr(ctx, "run-index", ctx_in.get("run_index", 0))

    result = execute(store, input_items, env, ctx)  # list (ok) or str (err) — untagged
    if isinstance(result, str):
        return {"error": result}
    items = []
    for it in result:
        binary = getattr(it, "binary", None)
        items.append(
            {
                "json": getattr(it, "json", "{}"),
                "binary": base64.b64encode(bytes(binary)).decode() if binary else None,
            }
        )
    return {"items": items}


# ── daemon stdio protocol ────────────────────────────────────────────────────


def emit(stdout, payload: dict) -> None:
    with _emit_lock:
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()


def _log(level: str, message: str) -> None:
    emit(
        sys.stdout,
        {
            "session_id": "",
            "event": "log",
            "data": {"level": str(level), "message": message},
        },
    )


def _session(stdout, session_id: str, mtype: str, data: Any) -> None:
    emit(
        stdout,
        {
            "session_id": session_id,
            "event": "session",
            "data": {"type": mtype, "data": data},
        },
    )


def _heartbeat(stdout) -> None:
    while True:
        emit(stdout, {"session_id": "", "event": "heartbeat", "data": None})
        time.sleep(2)


def main(stdin=None, stdout=None) -> None:
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    cfg = load_config(os.getcwd())
    engine, component = load_component(cfg)

    threading.Thread(target=_heartbeat, args=(stdout,), daemon=True).start()

    for line in stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _log("error", "invalid json line")
            continue
        session_id = req.get("session_id")
        if not session_id or req.get("event") != "request":
            continue
        payload = req.get("data") or {}
        try:
            result = run_execute(engine, component, cfg, payload)
        except Exception as e:  # noqa: BLE001 — surface any failure as an error frame
            _session(stdout, session_id, "error", {"message": str(e)})
            continue
        if "error" in result:
            _session(stdout, session_id, "error", {"message": result["error"]})
            continue
        for item in result["items"]:
            _session(stdout, session_id, "stream", item)
        _session(stdout, session_id, "end", {})


if __name__ == "__main__":
    main()
