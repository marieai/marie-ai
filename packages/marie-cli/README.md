# Marie CLI

`marie-cli` provides the command-line frontend for `marie-ai`. It contributes
`marie.cli` to the implicit PEP 420 `marie` namespace and installs the `marie`
console script.

```bash
uv add marie-cli
marie gateway --help
python -m marie gateway --help
```

The dependency direction is intentionally one-way: `marie-cli` depends on
`marie-ai`. The server distribution retains `marie/__main__.py` as a thin
launcher so source-tree commands and `python -m marie` remain stable; it imports
`marie.cli` only when that optional entry point is invoked.
Parser-schema generation and command lookup indexes used by core modules live
under `marie.parsers`; `marie.cli.export` and `marie.cli.lookup` expose those
services to CLI consumers.

The CLI wheel contains `marie/cli`, but never `marie/__init__.py` or a second
copy of `marie/__main__.py`.

## Development

The root development environment installs this package as an editable uv path
dependency. Source changes are immediately visible after `uv sync`.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv build
```
