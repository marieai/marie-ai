# Marie Plugin Daemon

`marie-plugin-daemon` installs and runs Marie extension packages in isolated
Python environments. It accepts standard ZIP archives containing
`marie-extension.yaml`, verifies signed invocation envelopes, and streams plugin
results over SSE.

The shared, stdlib-only Python API for plugin entrypoints lives at
[`python_runtime`](./python_runtime/README.md). The daemon embeds and injects
this package at runtime; individual plugins do not package it as a dependency.

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
- `POST /v1/plugins/install`
- `GET /v1/plugins`
- `DELETE /v1/plugins/{packageRef}`
- `POST /v1/dispatch/invoke`

## Package Rules

- Marie packages are discovered by `marie-extension.yaml`.
- Standard ZIP archives are supported when they contain exactly one `marie-extension.yaml`.
- `.difypkg` is rejected as a runtime package format.
- The decoder reads metadata and inventories files; it does not import, evaluate, or execute package code.
