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

from marie.extension.datasource import DatasourceResult, DatasourceRequest
from marie.extension.manifest import ExtensionPackage
from marie_plugins.runtime import run
from marie_plugins.runtime.datasource import dispatch_datasource

package = ExtensionPackage.model_validate(
    yaml.safe_load(Path('marie-extension.yaml').read_text())
)
definition = package.datasources[0]


def read_data(request: DatasourceRequest) -> DatasourceResult:
    # Use the datasource configuration and its normal provider implementation.
    return DatasourceResult(records=[{'id': 'remote-42', 'title': 'Example'}])


run(lambda frame: dispatch_datasource(frame, definition, read_data))
```

Install the SDK and the Python runtime's `datasource` extra in the plugin
environment. The dispatcher reuses the existing session protocol, validates the
declared dataset inputs and record schema, and returns a `stream` result followed
by `end`. Inline output keeps the existing 1 MiB transport bound; larger content
uses the datasource's existing stored output references. There is no page-size
or cursor contract. Records retain their declared structure and do not require
text or chunk fields. Schema references must resolve locally.

The datasource delivers data for ingestion using its normal implementation and
output/storage mechanism. Pagination of stored records belongs to query/list
APIs after ingestion. Any provider-specific request handling stays internal to
the datasource. The plugin entrypoint chooses its handler; requests cannot supply
implementation paths.

Studio's `kbIndexes.testDatasource` API accepts `tenantId`, `indexId` and `sourceId`.
It requires KB update permission, loads the saved binding, and invokes the
datasource once on the registered Plugin Executor using normal signed admission.
Credentials resolve in the worker. This test operation does not enable recurring
discovery or index its returned records.

### Index datasource document output

A datasource can opt into the existing KB document workflows by declaring
`indexingAdapter: document` alongside `outputSchema`. Other domain output stays
in its original format and needs its own compatible adapter.

Return stable record IDs and references to content already saved by the datasource:

```json
{
  "records": [{
    "id": "wiki:42",
    "uri": "s3://documents/tenants/TENANT_ID/wiki/42.md",
    "input_ref": {
      "kind": "text",
      "uri": "s3://documents/tenants/TENANT_ID/wiki/42.md",
      "media_type": "text/markdown",
      "sha256": "SHA256_OF_SAVED_BYTES"
    }
  }]
}
```

Use a real tenant UUID and lowercase 64-character SHA-256 digest. Saved text and
saved extraction references use `kb_index_content`. An original document uses
`uri` plus `sha256` and goes through `kb_indexing`. Saved artifacts must belong
to the binding's tenant; an extraction reference must match that document's
completed workflow. No output is copied into a second archive.

Unchanged records keep their indexing DAG. Changed references/digests enqueue
replacement indexing while the previous successful result remains searchable.
Return `{"id": "wiki:42", "deleted": true}` for an explicit removal. An empty
or partial delivery does not mean unseen items were deleted. Removed items need
the existing cleanup flow before reintroduction; after cleanup completes, a
reintroduced stable ID receives a fresh indexing DAG. An unchanged item whose
indexing failed also receives a fresh DAG on an explicit new sync.

Studio's Sync action and source schedule enqueue `kb_datasource_sync`. The Plugin
Executor resolves the saved binding when its scheduler job executes. It does not
queue the interactive test's expiring envelope. The existing KB item sensor then
submits indexing jobs. Configure the worker's existing `storage.psql` block and
`storage.s3.bucket_name` to match the registered KB storage connection. Output
URIs must use that bucket and the binding's tenant prefix. Share Studio's existing
extension package archive directory with the worker; the mock service template
uses `MARIE_DEFAULT_MOUNT/plugin-packages`. Match `MARIE_EXTENSION_PACKAGE_STORE`
if Studio overrides that directory, and forward the normal daemon signing key.

The full sync respects the installed package's `runtimePolicy.timeoutMs`; set it
for the provider's expected run time before qualifying a large synchronization.
The interactive datasource test retains its short preview bound. Scheduled runs
require the configured principal to remain an active organization member.

Explicit removal immediately withdraws results from search. Normal datasource
processing calls the existing backend cleanup functions for that KB-owned indexed
output before binding removal or ID reintroduction. The datasource owns cleanup
of its source records, files and saved artifacts. The
original-document `sha256` currently detects changes. Verification of those raw
bytes belongs to the original-document pipeline qualification; saved-text reads
already verify the supplied artifact digest.
