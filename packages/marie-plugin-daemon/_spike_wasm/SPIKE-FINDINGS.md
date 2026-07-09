# W1.0 — Component-Model Feasibility Spike — FINDINGS (2026-06-29)

**Verdict: GATE FAIL for in-process `wasmtime-go`; GATE PASS for in-process Python `wasmtime`.**
**LOCKED decision (user): run components as daemon-managed `python_source` plugins via Python wasmtime
(`marie_wasm.daemon_runner`).** No wasmtime-go, no Rust sidecar, no daemon Go changes.

## ✅ PROVEN END-TO-END (2026-06-29, `proof_e2e.py`)

Ran the full chain in the spike venv against the checked-in fixture `nodes/compiled/http-request.wasm`:
component instantiated → it called host `console.log` AND host `http-client.fetch` → `fetch` made a **real outbound HTTP GET** to a local `http.server` → `execute` returned a list with one item:
```
[host] console.log[info]: HTTP Request: GET http://127.0.0.1:PORT/ (node: n1)
[host] fetch called; req type: Record  attrs: ['body','headers','method','url']
RESULT type: list
  item.json: {"status":200,"headers":{},"body":"{\"ok\":true}","success":true}
```
This is the blunt codex acceptance, proven outside the daemon. See `proof_e2e.py` for the exact working code.

### THE PROVEN RECIPE (exact — host.py + the plan must follow this)

1. **Engine/Config:** `cfg = Config(); cfg.wasm_component_model = True`.
2. **WASI is REQUIRED** (the fixture's `serde_json` pulls `wasi:random` via std HashMap): `linker.add_wasip2()` AND `store.set_wasi(WasiConfig())`. Without it, execute traps on `wasi:random/random@0.2.3#get-random-bytes`.
3. **Imports & loading:** `from wasmtime.component import Component, Linker, Record`; `Component.from_file(engine, path)`; `with linker.root() as root: with root.add_instance(IFACE) as i: i.add_func(FN, callable)`.
4. **Interface names are VERSIONED:** `marie:node/http-client@1.0.0`, `marie:node/secrets@1.0.0`, `marie:node/kv@1.0.0`, `marie:node/console@1.0.0`, `marie:node/events@1.0.0`. (host.py uses UNVERSIONED + wrong fn names → both must be fixed.) Function names: `fetch`, `get`, `get`/`put`/`delete`, `log`, `emit`.
5. **Host funcs take `(store, *args)`.** Args arrive marshaled: records as `wasmtime.component.Record` instances (read with `getattr`), `list<u8>` as **`bytes`**, `option<T>` as `None`/value, enums as their case string.
6. **Return value marshaling:**
   - WIT **record** → return a `Record()` instance with the field attrs set (e.g. http-response: `.status` int, `.headers` list of `(str,str)` tuples, `.body` **bytes**).
   - **`list<u8>` is `bytes`, NOT `list[int]`** (marshaler raises "expected bytes value" otherwise).
   - **`result<T,E>` with disjoint T,E is UNTAGGED** → return the ok payload (the Record) directly, or the err value (a `str`) — NOT a `Variant`, NOT a tagged dict.
   - **`option<T>`** → `None` or the value.
7. **Calling execute:** `f = instance.get_func(store, "execute")`; `f(store, input_list, env, ctx)` where `env` is a Record with `.vars` (the node-config JSON **string**), `ctx` a Record with attrs `workflow-id`/`execution-id`/`node-id`/`run-index`. **`response = ok(list<item>) | err(string)` is an untagged variant → the call returns a `list` (ok) or `str` (err) directly.** Each output `item` is a Record with `.json` (str) and `.binary` (`bytes`/`None`).

⚠️ Therefore `host.py`'s current `http_request` (returns `{"ok":...}`/`{"error":...}` dicts, `body` as `list[int]`) and `get_bindings()` (unversioned names, `request`/`set`/`logging`) are BOTH wrong for wasmtime 46. Task 1 must rework them to the recipe above (versioned `@1.0.0` interface names; `fetch`/`put`/`console`; `(store,*args)`; `Record`/`bytes` returns; untagged result).

## UPDATE — Python `wasmtime` path VERIFIED viable (the chosen path)

Installed `wasmtime` (Python) **46.0.1** in a clean venv (`uv pip install -e packages/marie-wasm`) and inspected the API:

- Component support lives in the **`wasmtime.component` submodule** (NOT top-level): `Component.from_file`, `Linker`, `Linker.root()`, `LinkerInstance.add_instance/add_func`, `Linker.instantiate`, `Instance.get_func`, `Func.__call__`, and a full value-type system (`Record`, `Variant`, `Option`, `List`, `String`, `U32`, `ResultType`, …).
- **Host imports CAN be defined**: `linker.root().add_instance("marie:node/http-client").add_func("fetch", fn)`. The C callback (`_linker.py:add_func`) invokes `fn(store, *pyargs)` where args are auto-marshaled from the component's own import types, and the return is marshaled back via the result type. → the exact thing wasmtime-go lacks.
- **Exports CAN be called**: `instance.get_func(idx)` → `Func`, which is callable.
- The checked-in fixture `nodes/compiled/http-request.wasm` loads via `Component.from_file` with `Config.wasm_component_model=True`.

**Conclusion:** Python wasmtime is capability-complete for the `node` WIT world (define 5 host imports + call `execute` with composite values). The Go blocker does not exist here.

### Two corrections this forces on the W1 plan code

1. **Use `wasmtime.component`**, not the legacy top-level `from wasmtime import Component, Linker` + `linker.define_func(iface, name, fn)`. That legacy API is what `marie/executor/wasm/wasm_executor.py` was written against — it is **broken on wasmtime 46** (ImportError + wrong linker API), reinforcing "legacy/unproven". Registration is `root().add_instance(iface)` → `add_func(name, fn)`.
2. **Host funcs take `(store, *args)`** and must return values matching the WIT result type. `host.py` methods are `method(*args) -> dict` with `{"ok":...}`/`{"error":...}` / `None` conventions designed for the OLD marshaler. A thin **adapter** is needed: drop/accept `store`, and translate host.py's return into wasmtime 46's expected `result<...>`/`option<...>` representation. **The exact result/option/variant value representation in wasmtime 46 is the ONE thing to pin down in Task 1** (write the instantiate+execute test, run it, adjust marshaling until the fixture returns `status:200`). Every capability is present; this is marshaling-shape calibration, not a feasibility risk.

`host.py get_bindings()` nested shape `{iface: {func: callable}}` still maps cleanly: iterate it with `add_instance`/`add_func` + the adapter. The WIT-name fix (Task 1) is still required.

---

## Original Go analysis (still valid — why wasmtime-go was rejected)

**Verdict: GATE FAIL for in-process `wasmtime-go` (any version), because the Go binding cannot
define host imports or call component exports — even though v46 can load/instantiate components.**

## Correction to the first pass

The first pass tested the **pinned `wasmtime-go/v33`**, which has *no component API at all*, and wrongly generalized to "wasmtime-go has no component model." Reviewer correctly flagged that **current `wasmtime-go/v46` DOES expose a component API** (`NewComponent`, `ComponentLinker`, `Config.SetWasmComponentModel`, component type inspection). Verified: `go get .../v46@latest` → `v46.0.1`, and the symbols exist.

## But v46's component API is load/instantiate/inspect ONLY — verified

Enumerating the **entire** component surface in `v46.0.1` (`component_*_feat_component_model.go`):

**Present:**
- `NewComponent` / `NewComponentDeserialize[File]` — load a component.
- `Config.SetWasmComponentModel(true)`, `SetConcurrencySupport`.
- `NewComponentLinker`, `(*ComponentLinker).Instantiate`, `DefineUnknownImportsAsTraps`, `Close`.
- Type inspection: `Component.Type`, `ComponentType.Import/ExportNth`, `ComponentItem`, `ComponentValType.Kind`, `(*ComponentInstance).GetExportIndex`.

**Absent (the parts our contract needs):**
- ❌ **No host-import function definition.** There is no `ComponentLinker.DefineFunc`/callback API — only `DefineUnknownImportsAsTraps` (stub imports so they *trap* if called). The `DefineFunc`/`Call` that exist in the package are on the **core `Linker`/`Func`** (module path), not components.
- ❌ **No exported-function call.** `ComponentInstance` exposes only `GetExportIndex`; there is no `Call` and no way to invoke `execute`.
- ❌ **No component value marshaling.** There is `ComponentValType` (type metadata) but no `ComponentVal` value type / constructors. The cgo layer does **not** bind `wasmtime_component_linker_define_func`, `wasmtime_component_func_call`, or `wasmtime_component_val_*`.

The maintainers' own tests confirm the ceiling: the only two component-linker tests are `TestComponentInstantiate` (no imports) and `TestComponentDefineUnknownImportsAsTraps` (imports satisfied by trapping). **No test defines a host function or calls an export — because those APIs don't exist.**

## Why that fails our contract

The `node` world (`marie-node.wit`) requires, end-to-end:
1. The host to **implement 5 imports the guest calls** — `http-client.fetch`, `secrets.get`, `kv.*`, `console.log`, `events.emit`. The checked-in fixture `http-request.wasm` calls `http-client.fetch`.
2. The host to **call `execute(input: list<item>, env, ctx) -> response`** and read the composite result.

v46 can instantiate the fixture only via `DefineUnknownImportsAsTraps`, which makes `http-client.fetch` **trap** on first call; and there is no API to call `execute` or marshal `list<item>`/`response`. So in-process wasmtime-go cannot run our WIT surface on the latest release. This matches the architecture doc's flagged risk ("composite-value marshaling" / host-import path) — still real in v46.

## Decision (locked, per reviewer's decision tree)

> "For Go, the preferred option is to upgrade … and use `NewComponent`/`ComponentLinker`. **If the current Go component API is not sufficient for our exported WIT surface, the fallback is a Rust Wasmtime sidecar.** Wazero remains excluded because it does not support WASI-P2 components."

Verified insufficient → **Rust Wasmtime sidecar**:

```
daemon (Go)  ──stdio/Unix socket──▶  marie-wasm-runner (Rust)
  runtime seam (W1 Tasks 2–3)            embeds wasmtime::component
  spawns + frames I/O                    implements node WIT host imports in Rust
                                         calls execute(...), returns response
```

Rust's `wasmtime` crate has the full component model: `Linker::instance().func_wrap(...)` to define host imports, typed `bindgen!`-generated calls for `execute`, and complete value marshaling.

This overrides **D3's mechanism** ("embed `wasmtime-go`") but preserves every other locked decision: the marie-wasm compiler is unchanged (D4), the daemon owns the WASM runtime, the runtime-type seam is intact, and the Python `WasmNodeExecutor` can still be retired (D3 intent). Wazero stays excluded.

## Impact on the W1 plan (delta)

- **Task 0:** complete (this finding).
- **Task 1** (add wasmtime-go + cgo): **replace** with "add the `marie-wasm-runner` Rust crate (wasmtime + component bindgen against `marie-node.wit`) + build/image profile; daemon gains a Rust toolchain stage or ships the prebuilt runner binary."
- **Tasks 2–3** (runtime seam, `runtime.type` branch): **unchanged** — mechanism-agnostic.
- **Task 4** (host imports): host imports move into the **Rust runner** (against the real WIT); the Go `wasm_runtime` becomes a **stdio client** that frames an execute request to the runner and reads result frames. The Go `permissions.go`/allow-list can stay Go-side (passed to the runner per invoke) or move to Rust — decide in re-plan.
- **Tasks 5–6** (pool wiring, `/v1/dispatch/invoke` E2E): **unchanged intent**; acceptance (fixture runs via the daemon, allow-listed HTTP via host import, streamed frame) stands — now satisfied by the Rust runner.
- **Task 7** (retire `WasmNodeExecutor`): **unchanged** (still achievable).

## Alternative still on the table

The reviewer's earlier pick — **keep the Python `WasmNodeExecutor` behind the daemon** (Python's `wasmtime` has full component support) — remains the lower-effort option, at the cost of keeping Python in the hot path and deferring the D3/D4 "retire the Python runner" goal. The Rust sidecar is preferred here per the decision tree.

`_spike_wasm/` is throwaway; keep until the fallback lands, then remove.
