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

## Container image

Build from this package directory, not the repository root:

```bash
docker build -f docker/serverless.dockerfile -t marie-plugin-daemon:local .
MARIE_PLUGIN_DAEMON_TEST_IMAGE=marie-plugin-daemon:local \
  go test -count=1 -v ./internal/httpapi -run '^TestContainer' -timeout 3m
```

The image includes Python 3.12, uv 0.11.28, trusted CA certificates and the
daemon's embedded Python SDK. It defaults to UID/GID `10001:10001`. The build
context allowlist excludes host environments, credentials, caches and test
fixtures. A separate regression exports a synthetic build context through
Docker to verify these exclusions without accessing real credentials. These
opt-in tests require local Docker and an already-built image; without
`MARIE_PLUGIN_DAEMON_TEST_IMAGE`, the ordinary Go suite skips them.

The test starts a disposable container with a loopback-only port, a read-only
root filesystem, dropped capabilities and temporary data/cache mounts. It uses
generated signing material, verifies unsigned/invalid signatures are rejected,
then installs a plugin with a local wheel dependency and invokes it through the
real HTTP/SSE API. It verifies the dependency, SDK, virtual environment and
non-root execution, then redeploys and removes the plugin. No package registry
or Python download is needed at runtime for this test. Its container is removed
afterward on the normal path, including assertion failures. A Docker client
failure during creation can leave a container requiring manual cleanup. The
test does not verify persistence across restart or Kubernetes behavior.

For a deployment, provide a **dedicated persistent volume per installation** at
`/mnt/data/marie-ai`, writable by UID/GID 10001, and a writable temporary mount at
`/tmp` for uv's cache. The data root remains configurable with
`MARIE_PLUGIN_STORAGE_ROOT`; changing it also requires mounting writable storage
there. Configure `MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID` and
`MARIE_PLUGIN_DAEMON_SIGNING_SECRET` through the deployment's secret references.
Do not enable dev-insecure mode. No signing keys are included in the image.

The bundled interpreter supports Python 3.12 packages. Automatic interpreter
downloads are disabled with `UV_PYTHON_DOWNLOADS=never`; other interpreter
versions or native system dependencies require a separately qualified image.
Dependency installation can still use configured package indexes unless
`UV_OFFLINE=1` is set. The base image does not include CUDA, model weights or
general-purpose build tools. A passing plugin fixture does not qualify GPU
executors or arbitrary third-party packages.

Publish the resulting image by immutable digest before using it in an approved
Sandbox recipe. The build/test commands above neither publish a release nor
connect the image to the Sandbox chart; those remain separate composition and
target-qualification steps.

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
