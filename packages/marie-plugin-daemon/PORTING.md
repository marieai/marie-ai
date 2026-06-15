# Marie Plugin Daemon Porting Map

Upstream reference: `/home/gbugaj/dev/marieai/dify-plugin-daemon`
Upstream commit: `0e9c3cfb97ba8293a9483e1510fd1a5547f54491` (`perf: decrease mem cost (#757)`)

This daemon is a local-runtime port of Dify's plugin daemon, not a full tree copy.
The directory layout intentionally keeps Dify-derived runtime code in upstream-recognizable paths and puts Marie-specific replacements under `internal/marie`.

## Dify-Derived Runtime Surface

| Marie path | Upstream path | Status | Reason |
| --- | --- | --- | --- |
| `pkg/entities/plugin_entities/event.go` | `pkg/entities/plugin_entities/event.go` | adapted | Keeps stdout event/session message wire names, with Marie typed errors and direct parser return values. |
| `internal/core/local_runtime/setup_python_environment.go` | `internal/core/local_runtime/setup_python_environment.go` | adapted | Keeps the local `uv` venv/pip subset and drops upstream mirror/patch/version-match machinery. |
| `internal/core/local_runtime/instance.go` | `internal/core/local_runtime/{instance.go,subprocess.go,signals_instance.go}` | adapted | Keeps Dify stdio process model, heartbeat readiness, session listeners, and stdin envelope; simplifies notifier layers. |
| `internal/core/plugin_manager/manager.go` | `internal/core/plugin_manager/{manager.go,installer.go,install_entities.go}` | adapted | Keeps install/list/remove/runtime-state responsibilities; replaces DB/object-store buckets with disk state. |
| `internal/core/io_tunnel/pool.go` | `internal/core/io_tunnel/generic.go` | adapted | Keeps session stream/end/error/invoke semantics; combines Marie pool lifecycle with Dify generic invocation flow. |
| `internal/core/io_tunnel/backwards_invocation/storage.go` | `internal/core/io_tunnel/backwards_invocation/*` | replaced | Implements only the storage backwards invocation family against Marie disk KV. |
| `internal/service/base_sse.go` | `internal/service/base_sse.go` | adapted | Keeps `data: <json>\n\n` SSE framing and flush behavior using stdlib HTTP instead of Gin. |

## Marie Replacement Surface

| Marie path | Replaces | Reason |
| --- | --- | --- |
| `internal/marie/auth` | Dify API key/auth middleware | Studio signs HMAC runtime envelopes instead of using Dify API keys. |
| `internal/marie/decoder` | Dify `.difypkg` decoder/runtime package reader | Marie runtime packages are directories or ZIPs containing `marie-extension.yaml`; `.difypkg` is converter-only input. |
| `internal/marie/policy` | Dify trust/capability checks | Marie verifies signed envelope claims for trust, capabilities, credentials, package identity, tenant, and runtime policy. |
| `internal/httpapi` | Dify Gin controller/service routing | Marie exposes the daemon through stdlib HTTP endpoints consumed by Studio. |

## Omitted Upstream Surface

The following upstream surfaces are intentionally omitted from Slice 12a: CLI generators, license tooling, cluster/serverless runtime, database integrations, object-store media buckets, generated model/datasource/trigger endpoints, Dify internal API callbacks except storage backwards invocation, and broad manifest entity models not needed by the local runtime handshake.
