# marie-extension

`marie-extension` defines the Marie extension package contract. It owns YAML schema models, safe package loading, validation, and authoring-time utilities.

The distribution name is `marie-extension`; its PEP 420 import is
`marie.extension`.

V1 is metadata-only:

- packages are directories or standard ZIP archives containing exactly one `marie-extension.yaml`
- validation never executes package code
- runtime execution remains disabled until registry, credentials, daemon envelope, and audit gates are implemented

## Development

```bash
cd packages/marie-extension
uv sync --extra dev
uv run pytest
```

## CLI

```bash
marie-extension validate --path tests/fixtures/minimal-tool
```
