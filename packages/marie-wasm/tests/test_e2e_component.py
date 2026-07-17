"""End-to-end spike: run the checked-in `node` WASI-P2 component through Python
wasmtime with working host imports, proving the daemon-runner path is viable.

This is the in-repo version of the W1.0 feasibility proof (see
packages/marie-plugin-daemon/_spike_wasm/SPIKE-FINDINGS.md). It loads
nodes/compiled/http-request.wasm, wires the marie:node host imports, calls a
local HTTP server through the host `fetch`, and asserts the component returns
status 200.

Run it yourself:
    cd packages/marie-wasm
    python -m pytest tests/test_e2e_component.py -v -s
(requires `wasmtime` installed; skipped otherwise.)
"""

import http.server
import json
import threading
import types as _t
from pathlib import Path

import pytest

# wasmtime 46+ exposes the component model in the `wasmtime.component` submodule
# (NOT the top level). Skip cleanly if wasmtime isn't installed.
wasmtime = pytest.importorskip("wasmtime")
component = pytest.importorskip("wasmtime.component")

from marie.wasm import Permissions  # noqa: E402

# The built-in node library ships inside this package: tests/ -> marie-wasm,
# so the nodes live at ../nodes/compiled/. Prefer the package constant.
try:
    from marie.wasm import BUILTIN_NODES_DIR as _BND

    FIXTURE = _BND / "http-request.wasm"
except ImportError:  # pragma: no cover
    FIXTURE = (
        Path(__file__).resolve().parents[1] / "nodes" / "compiled" / "http-request.wasm"
    )

# The compiled component imports VERSIONED interface instances. The version comes
# from the WIT package declaration `package marie:node@1.0.0;`.
IFACE_VERSION = "@1.0.0"


def _start_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


@pytest.mark.skipif(not FIXTURE.exists(), reason="compiled fixture node not present")
def test_http_request_node_runs_end_to_end():
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen

    from wasmtime import Config, Engine, Store, WasiConfig
    from wasmtime.component import Component, Linker, Record

    srv, url = _start_server()
    try:
        # Trust boundary: the host imports enforce marie.wasm Permissions.
        perms = Permissions(allow_http=True)
        calls = {"fetch": 0, "log": []}

        # marie:node/http-client.fetch : func(req: http-request) -> result<http-response, string>
        def fetch(
            store, req
        ):  # host funcs receive (store, *args); records arrive as Record
            calls["fetch"] += 1
            target = getattr(req, "url")
            if not perms.is_host_allowed(urlparse(target).hostname or ""):
                return "host not allowed"  # untagged result err arm == a str
            with urlopen(Request(target, method=getattr(req, "method", "GET"))) as r:
                body, status = r.read(), r.status
            resp = Record()
            resp.status = status
            resp.headers = []  # list<tuple<string,string>>
            resp.body = body  # list<u8> == Python bytes
            return resp  # untagged result ok arm == the Record

        def secrets_get(store, name):
            return None  # option<string> none

        def log(store, level, message):
            calls["log"].append((str(level), message))
            return None

        cfg = Config()
        cfg.wasm_component_model = True
        engine = Engine(cfg)
        store = Store(engine)
        store.set_wasi(
            WasiConfig()
        )  # required: component uses WASI (serde -> wasi:random)

        comp = Component.from_file(engine, str(FIXTURE))
        linker = Linker(engine)
        linker.add_wasip2()  # required: provides the WASI imports

        with linker.root() as root:
            with root.add_instance(f"marie:node/http-client{IFACE_VERSION}") as i:
                i.add_func("fetch", fetch)
            with root.add_instance(f"marie:node/secrets{IFACE_VERSION}") as i:
                i.add_func("get", secrets_get)
            with root.add_instance(f"marie:node/console{IFACE_VERSION}") as i:
                i.add_func("log", log)
            with root.add_instance(f"marie:node/kv{IFACE_VERSION}") as i:
                i.add_func("get", lambda store, key: None)
                i.add_func("put", lambda store, key, value, ttl: None)
                i.add_func("delete", lambda store, key: None)
            with root.add_instance(f"marie:node/events{IFACE_VERSION}") as i:
                i.add_func("emit", lambda store, event_type, payload: None)

        instance = linker.instantiate(store, comp)
        execute = instance.get_func(store, "execute")
        assert execute is not None, "component must export 'execute'"

        # execute(input: list<item>, env: env, ctx: context) -> response
        env = Record()
        env.vars = json.dumps({"method": "GET", "url": url, "headers": {}})
        ctx = Record()
        setattr(ctx, "workflow-id", "wf")
        setattr(ctx, "execution-id", "ex")
        setattr(ctx, "node-id", "n1")
        setattr(ctx, "run-index", 0)

        # response = variant ok(list<item>) | err(string) ; disjoint -> untagged
        result = execute(store, [], env, ctx)
        assert isinstance(
            result, list
        ), f"expected ok(list<item>), got {type(result)}: {result!r}"
        assert len(result) == 1
        out = json.loads(getattr(result[0], "json"))
        assert out["status"] == 200
        assert out["success"] is True
        assert calls["fetch"] == 1  # host fetch was actually exercised
        assert any(
            "HTTP Request" in m for _, m in calls["log"]
        )  # host console.log fired
    finally:
        srv.shutdown()
