# Marie Plugin Python Runtime

This directory contains the stdlib-only Python support package shared by
daemon-managed Marie plugins.

Plugins import the public API from `marie_plugins.runtime`:

```python
from marie_plugins.runtime import SessionFrame, run, session_frame
```

The Go daemon embeds this source, materializes it beside each plugin virtual environment, and injects that location through `PYTHONPATH`. In production the daemon-provided copy is the only runtime: plugin archives must not copy this directory, and plugins must never list `marie-plugin-runtime` in `[project.dependencies]`.

For development, this directory is also buildable as the `marie-plugin-runtime` wheel so that plugin authors outside this repository get IDE resolution and plain `pytest` runs without a monorepo checkout. Declare it in the `dev` dependency group only:

```bash
uv add --dev marie-plugin-runtime
```

Inside this repository, install it editable into whichever development environment your IDE uses as its interpreter, so `marie_plugins.runtime` resolves from site-packages:

```bash
# from packages/marie-plugin-document-extraction (or any in-repo plugin)
uv pip install --no-deps -e ../marie-plugin-daemon/python_runtime
```

`uv sync` prunes packages that are not in the lock, so re-run the install after syncing that environment.

The daemon installs plugin environments with `--no-dev`, so the published copy never enters production; at runtime the daemon's embedded copy arrives first on `PYTHONPATH` regardless. The package version lives in `marie_plugins.runtime.__version__`; a plugin developed against a newer runtime than the daemon embeds may see protocol skew, so keep the dev dependency aligned with the target daemon release.

`marie_plugins` is an implicit namespace package. Plugins contribute their own subpackages, such as `marie_plugins.document_extraction`, while the daemon contributes `marie_plugins.runtime`. Do not add `marie_plugins/__init__.py` — in this repository, in the wheel, or in a plugin archive; a regular package at that name shadows the other half of the namespace.

For in-repo development tests, add `packages/marie-plugin-daemon/python_runtime` to the test runner's Python path (the document-extraction plugin does this via `[tool.pytest.ini_options] pythonpath`). Production plugin execution receives the same path from the daemon automatically.

Packaged plugin tests should use the shared stdio test client instead of
reimplementing subprocess and session handling:

```python
import sys

from marie_plugins.runtime.testing import StdioPluginTestClient

with StdioPluginTestClient([sys.executable, '-m', 'main'], cwd=plugin_dir) as plugin:
    capabilities = plugin.invoke('capabilities')
```

The client injects the runtime path, filters frames by session, enforces a
response timeout, captures stderr diagnostics, and stops the child process.
