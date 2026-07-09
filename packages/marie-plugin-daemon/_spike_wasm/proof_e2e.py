"""End-to-end proof: Python wasmtime 46 runs the node component with host imports + execute."""

import http.server
import json
import threading
import types as _t
from pathlib import Path

from wasmtime import Config, Engine, Store, WasiConfig
from wasmtime.component import Component, Linker, Record

FIXTURE = "/home/gbugaj/dev/marieai/marie-ai/packages/marie-wasm/nodes/compiled/http-request.wasm"


def start_server():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def main():
    srv, url = start_server()
    cfg = Config()
    cfg.wasm_component_model = True
    engine = Engine(cfg)
    store = Store(engine)
    wasi = WasiConfig()
    wasi.inherit_stderr()
    store.set_wasi(wasi)
    component = Component.from_file(engine, FIXTURE)
    linker = Linker(engine)
    linker.add_wasip2()

    # host import: marie:node/http-client.fetch  (req record) -> result<http-response, string>
    def fetch(store, req):
        print(
            "  [host] fetch called; req type:",
            type(req).__name__,
            "attrs:",
            [a for a in dir(req) if not a.startswith('_')][:8],
        )
        method = getattr(req, "method", "GET")
        u = getattr(req, "url", url)
        import urllib.request

        with urllib.request.urlopen(urllib.request.Request(u, method=method)) as r:
            body = r.read()
            status = r.status
        # http-response record: status:u16, headers:list<tuple>, body:list<u8>
        resp = Record()
        resp.status = status
        resp.headers = []
        resp.body = body
        return resp  # untagged result<http-response,string> ok -> the Record itself

    def secrets_get(store, name):
        return None  # option<string> none

    def log(store, level, message):
        print(f"  [host] console.log[{level}]: {message}")
        return None

    with linker.root() as root:
        with root.add_instance("marie:node/http-client@1.0.0") as i:
            i.add_func("fetch", fetch)
        with root.add_instance("marie:node/secrets@1.0.0") as i:
            i.add_func("get", secrets_get)
        with root.add_instance("marie:node/console@1.0.0") as i:
            i.add_func("log", log)
        with root.add_instance("marie:node/kv@1.0.0") as i:
            i.add_func("get", lambda store, key: None)
            i.add_func("put", lambda store, key, value, ttl: None)
            i.add_func("delete", lambda store, key: None)
        with root.add_instance("marie:node/events@1.0.0") as i:
            i.add_func("emit", lambda store, et, payload: None)

    instance = linker.instantiate(store, component)
    execute = instance.get_func(store, "execute")
    print("execute func:", execute)

    # execute(input: list<item>, env: env, ctx: context) -> response
    env = _t.SimpleNamespace(
        vars=json.dumps({"method": "GET", "url": url, "headers": {}})
    )
    ctx = _t.SimpleNamespace(
        **{"workflow-id": "wf", "execution-id": "ex", "node-id": "n1", "run-index": 0}
    )
    result = execute(store, [], env, ctx)
    print("RESULT type:", type(result).__name__)
    items = result if isinstance(result, list) else []
    for it in items:
        print("  item.json:", getattr(it, "json", None))
    srv.shutdown()


if __name__ == "__main__":
    main()
