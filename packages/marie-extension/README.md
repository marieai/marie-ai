# marie-extension

`marie-extension` defines the Marie extension package contract. It owns YAML schema models, safe package loading, validation, and authoring-time utilities.

The distribution name is `marie-extension`; its PEP 420 import is
`marie.extension`.

Package loading and validation only read metadata:

- packages are directories or standard ZIP archives containing exactly one `marie-extension.yaml`
- validation never executes package code
- tools, models, datasources, triggers, endpoints, and agent strategies are typed for discovery and validation
- runtime execution remains owned by the Marie plugin daemon and host-side policy

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

## Datasource page handlers

Declare a datasource under `datasources` with an exact `ref`, a
`configurationSchema` for dataset inputs and an `outputSchema` for each returned
record. An optional `providerRef` names a declared `datasource_provider` in the
same package; its existing credential requirements and installation bindings
supply credentials. A datasource without credentials can omit `providerRef`.

```yaml
providers:
  - ref: providers/wiki
    type: datasource_provider
    credentials:
      - key: api_key
        type: api_key
        required: true
datasources:
  - ref: datasources/wiki
    name: Wiki
    providerRef: providers/wiki
    configurationSchema:
      parameters:
        - key: space
          type: string
          required: true
    outputSchema:
      type: object
      required: [id, title]
      properties:
        id: {type: string}
        title: {type: string}
```

The installed plugin entrypoint binds its handler to that declaration:

```python
from pathlib import Path
import yaml

from marie.extension.datasource import DatasourcePage, DatasourceRequest
from marie.extension.manifest import ExtensionPackage
from marie_plugins.runtime import run
from marie_plugins.runtime.datasource import dispatch_datasource

package = ExtensionPackage.model_validate(
    yaml.safe_load(Path('marie-extension.yaml').read_text())
)
definition = package.datasources[0]


def fetch_page(request: DatasourceRequest) -> DatasourcePage:
    # Fetch at most request.limit records using request.configuration,
    # request.cursor and request.credentials.
    return DatasourcePage(records=[{'id': 'remote-42', 'title': 'Example'}])


run(lambda frame: dispatch_datasource(frame, definition, fetch_page))
```

Install the SDK and the Python runtime's `datasource` extra in the plugin
environment. The dispatcher reuses the existing session protocol, validates the
declared dataset inputs and record schema, and returns one `stream` page followed
by `end`. It accepts at most 100 records and 1 MiB per page. Use stored artifact
references for larger content. Records retain their declared structure; they do
not need text or chunk fields. Schema references must resolve locally.

`next_cursor` continues pagination; it does not establish source completeness or
authorize removal of unseen records. The plugin entrypoint chooses its handler;
requests cannot supply implementation paths.

Studio's `kbIndexes.testDatasource` API accepts `tenantId`, `indexId`, `sourceId`,
and optional `cursor`/`limit`. It requires KB update permission, loads the saved
binding, and invokes one page on the registered Plugin Executor using signed
admission. Credentials resolve in the worker. This operation does not enable
recurring discovery or index its returned records.
