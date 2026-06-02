# Marie Plugin Daemon

`marie-plugin-daemon` is the decode/stub runtime service for Marie extension packages. This first milestone accepts Marie extension directories or standard ZIP archives containing `marie-extension.yaml`, expands package metadata, and exposes health/version endpoints.

Runtime execution is intentionally disabled. Tool, model, Webapp, MCP, and subprocess invocation routes must stay unsupported until signed envelopes, credential policy, and isolation work land.

## Build

```bash
go test ./...
go build -o dist/marie-plugin-daemon ./cmd/server
```

## Run

```bash
go run ./cmd/server --addr 127.0.0.1:8099
```

Endpoints:

- `GET /health`
- `GET /version`
- `POST /v1/packages/decode` with JSON body `{"path": "/path/to/package-or.zip"}`
- `POST /v1/runtime/stub-invocations` returns `501` until runtime execution is enabled

## Package Rules

- Marie packages are discovered by `marie-extension.yaml`.
- Standard ZIP archives are supported when they contain exactly one `marie-extension.yaml`.
- `.difypkg` is rejected as a runtime package format.
- The decoder reads metadata and inventories files; it does not import, evaluate, or execute package code.
