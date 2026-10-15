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

Runtime packages and Python environments default to `/mnt/data/marie-ai/plugin-runtime`.
Set `MARIE_DEFAULT_MOUNT` or `MARIE_PLUGIN_STORAGE_ROOT` to change the data root;
the daemon appends `plugin-runtime/<organization>__<workspace>/` to that root.
Studio keeps uploaded ZIPs separately in `/mnt/data/marie-ai/plugin-packages`.

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
- When `pyproject.toml` and `uv.lock` are present, installation uses `uv sync --locked --no-dev`.
- Without `uv.lock`, a non-empty `requirements.txt` supplies dependencies, including for Dify exports that also contain `pyproject.toml`. Project-only packages still require a lockfile, and an invalid lockfile fails installation rather than falling back to requirements.
