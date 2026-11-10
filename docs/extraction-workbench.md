# Deploy configured schema extraction

The `schema_extract` built-in planner runs OCR and then the existing
`DocumentAnnotatorLLMExecutor`. Each submission selects a saved `layout` and
annotator `key`. The runtime loads the current files from the extraction config
mount when the extraction node starts.

## Configure the saved definition

The source example selects `layout: workbench` and `key: extract`:

```text
config/extract/
  base/
    base-config.yml
    field-config.yml
    schema-extract.j2
  TID-workbench/annotator/
    config.yml
    schema-extract.output.json
```

Deploy those files through the same config mount used by existing annotators.
The runtime root is `marie.constants.__config_dir__/extract`; a deployment may
mount it at `/mnt/data/marie-ai/config/extract`. Studio's prompt-authoring
repository must save to the directory deployed on that runtime mount.

The example's `model_config.model_name: gpt-4o-mini` is a conventional configured
model alias. Select an alias actually available in your deployment's model
catalog/routing configuration, and configure its credentials using the existing
environment/secret configuration. Set `multimodal` and completion parameters for
that model in `TID-workbench/annotator/config.yml`. The executor service YAML does
not need another model configuration block. Existing direct or queued LLM
submission, tracing, context providers and configured output transforms apply.

For a custom profile, select its existing `prompt_path` and save the matching
`<prompt-stem>.output.json` in that profile's `annotator` directory. The runtime
uses `PromptTemplate.from_file_with_fallback`: layout prompt first, then
`extract/base/`. Schema lookup uses the same relative prompt basename, preferring
the layout's schema over the base schema. The default profile uses the shared
`base/schema-extract.j2` prompt and the layout's own schema.

Save output schemas in the existing editor format:

```json
{
  "version": "1.0",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {"payer_name": {"type": "string"}},
    "required": ["payer_name"]
  }
}
```

Legacy bare JSON Schema files also load. Extraction requires a valid saved
object schema and `model_config.expect_output: json`. `EXTRACTOR_SCHEMA` is bound
internally to the schema body alongside the existing OCR/page variables. Custom
templates retain their instructions and must return schema-shaped JSON; use an
existing configured `output_transform` when their native output requires
normalization. Missing required fields are reported in the result. Invalid
types or malformed JSON fail the extraction node.

## Route the existing executors

Deploy the OCR executor under `extract_executor` and the LLM annotator under
`annotator_llm_executor`. The planner targets:

```text
extract_executor://document/extract
  → annotator_llm_executor://annotator/schema
```

The second deployment uses the existing class:

```yaml
executors:
  - name: annotator_llm_executor
    uses:
      jtype: DocumentAnnotatorLLMExecutor
      metas:
        py_modules:
          - marie.executor.extract
      with:
        storage: <your existing executor storage configuration>
        llm_tracking: <your existing tracking configuration>
```

Reuse your deployment's valid storage/tracing settings in place of the
placeholders. The gateway's ordinary built-in planner registration includes
`schema_extract`; no additional planner package is required. Enable queued LLM
dispatch only after the existing queue/fabric/model policy is provisioned.

## Submit and read a run

Submit through the existing gateway `/api/v1/invoke` job API:

```json
{
  "action_type": "command",
  "command": "job",
  "action": "submit",
  "name": "schema_extract",
  "metadata": {
    "planner": "schema_extract",
    "project_id": "tenant-id",
    "processor_id": "processor-id",
    "ref_type": "extractor_document",
    "ref_id": "document-id.pdf",
    "uri": "s3://marie/documents/document-id.pdf",
    "policy": "allow_duplicate",
    "run_params": {"layout": "workbench", "key": "extract", "parse_mode": "ocr"}
  }
}
```

The returned `job_id` is the DAG ID. Query that DAG for status and errors. On
success read
`s3_asset_path(ref_id, "extractor_document") + "/extraction/<dag_id>.json"`.
`MARIE_S3_BUCKET` controls the bucket through the standard asset path helper.
Result pages are absolute zero-based source pages, including when existing OCR
page selection is used. Per-page values append in source order; repeated rows
and scalar occurrences remain distinct.

Per-page profiles require output for every selected source page before the run
can succeed. Per-table profiles retain provider-selected pages and multiple
units on one page. For a selected-page run, the OCR metadata does not provide
the original container's page count; `pages.total` counts the available selected
OCR pages rather than estimating the original document length.

The planner marks its OCR node with an internal `strict_result` operator flag.
OCR errors and missing metadata then fail that node, preventing extraction from
continuing with metadata from a previous run. Existing OCR jobs retain their
default response behavior.

Annotator output, refinement scratch and debug files use a temporary directory
per execution, outside the shared document asset working directory. The
configured annotator name remains `extract`. Temporary artifacts are removed
after success or failure; only the extraction result is persisted. New runs
load the newly saved files. An annotation, validation or result upload failure
raises and fails the node; successful persistence is required before returning
success.
