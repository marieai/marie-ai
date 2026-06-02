# marie-extension

`marie-extension` defines the Marie extension package contract. It owns YAML schema models, safe package loading, validation, and authoring-time utilities.

V1 is metadata-only:

- packages are directories or standard ZIP archives containing exactly one `marie-extension.yaml`
- validation never executes package code
- runtime execution remains disabled until registry, credentials, daemon envelope, and audit gates are implemented

## Development

```bash
cd packages/marie-extension
pip install -e ".[dev]"
pytest
```

## CLI

```bash
marie-extension validate --path tests/fixtures/minimal-tool
```
