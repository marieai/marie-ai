# Test document extraction and OCR through the gateway

Use `document_extract.py` to submit one document to `POST /api/v1/invoke`.
The example uses the `extract` planner and defaults to `--parse-mode auto`.
The worker invokes the document extraction plugin for formats it supports,
including PDF when advertised by the plugin. OCR handles supported image inputs
and is the fallback when the plugin returns `not_extractable` for an OCR-loadable
document. Use `--parse-mode ocr` to explicitly bypass the plugin and run OCR.

The example loads the existing `examples/config.dev.json` by default. It reuses
its gateway URL, API key, queue ID, and S3 settings. No configuration changes are
required. The gateway, scheduler, extraction worker, and object storage must be
running. Run a worker from the checkout containing the changes you want to test.

## Submit an existing document

From the repository root:

```bash
.venv/bin/python examples/document_extract.py \
  --uri 's3://marie/incoming/scan.tif' \
  --pipeline default
```

Use a URI that the extraction workers can read. The gateway URL in the config is
its origin; the script appends `/api/v1/invoke`. `--mode` defaults to `sparse`;
`--timeout` controls how long HTTP submission may take. Use the final gateway
URL; the client rejects redirects.

For an OCR-only run, add `--parse-mode ocr`. `--mode` selects the OCR layout
mode and does not control plugin routing:

```bash
.venv/bin/python examples/document_extract.py \
  --input /path/to/scan.pdf --parse-mode ocr --wait
```

The shell entry point launches the same client, including when called from
another directory. Set `MARIE_PYTHON` to use your Marie environment:

```bash
export MARIE_PYTHON="$PWD/.venv/bin/python"
bash examples/srcFile-001.sh --uri 's3://marie/incoming/scan.tif'
```

No separate payload JSON file or RabbitMQ listener is needed to submit a job.
The Python client sends the current gateway command envelope:

```text
parameters.invoke_action
  action_type: command
  command: job
  action: submit
  name: extract
  api_key: configured API key
  metadata:
    project_id: configured project ID, falling back to the API key
    ref_id: input filename
    ref_type: extract
    queue_id: configured default_queue_id
    uri: your input URI
    planner: extract
    run_params: {parse_mode: auto}
    features: [{type: pipeline, name: default}]
```

As in the existing batch client, the API key supplies the project identity when
`project_id` is absent from the config. You can override it with `--project-id`
or `MARIE_PROJECT_ID`.

## Upload a local document

The same client uploads a local TIFF, PNG, JPEG, or PDF using the existing
config's `storage` section before submitting the job:

```bash
.venv/bin/python examples/document_extract.py --input /path/to/scan.tif
```

Select another existing configuration when needed:

```bash
.venv/bin/python examples/document_extract.py \
  --config /path/to/other-config.json \
  --input /path/to/scan.tif
```

The client reads `api_base_url`, `api_key`, `default_queue_id`, `storage`, and
optional `project_id` from the config. It does not change the config file.
`MARIE_API_BASE_URL`, `MARIE_API_KEY`, and `MARIE_PROJECT_ID` override configured
values; `--api-base-url` and `--project-id` take precedence over those variables.
The existing `S3_*` storage settings can also be overridden through environment
variables. A failed upload stops submission.

## Wait for extraction and download the response

Add `--wait` to receive completion through RabbitMQ and download the extraction result,
using the existing `queue` and `storage` sections of `config.dev.json`:

```bash
.venv/bin/python examples/document_extract.py \
  --input /path/to/scan.tif \
  --wait \
  --output-dir /tmp/marie-ocr-result
```

This saves the extraction response with its original S3 filename, for example
`/tmp/marie-ocr-result/scan.tif.meta.json`, with identical contents. It downloads
the files in its `assets` list under `/tmp/marie-ocr-result/assets/`.
For plugin extraction, this includes the original Markdown artifact (for example,
`assets/document-<id>.md`). The worker uploads that file before saving its URI
in the metadata's `assets` list.
The final JSON printed to the terminal includes the job ID and these local paths.
Without `--output-dir`, results go to `/tmp/marie/<job-id>/`.

The listener subscribes before submission and matches events to the returned job
ID. It uses a private queue so it can run alongside the batch client. This waits
for a new submission; it does not retrieve completion events for earlier jobs.
RabbitMQ must be reachable from the machine running the example.

`--wait-timeout` defaults to 600 seconds and is separate from the HTTP
`--timeout`. A failed job, completion timeout, or failed download exits with an
error. Interrupting the listener does not cancel the submitted job. The shell
entry point accepts the same flags.

## Check workflow assets

Without `--wait`, a successful submission prints an acceptance response:

```json
{
  "status": "ok",
  "msg": "job submitted with id <job-id>",
  "job_id": "<job-id>"
}
```

This confirms that the gateway accepted the job. Extraction completes asynchronously.
HTTP errors, gateway application errors, and responses without a job ID make
the client exit with an error.

1. Open the returned job/DAG ID in Studio's workflow details.
2. Wait for the extraction step to complete.
3. Inspect **Workflow assets** and each asset's storage link.
4. Open the stored metadata JSON and its existing `assets` list to inspect the
   generated outputs.

When testing the file-based tracking change, workflow assets should identify
stored file URIs, including the metadata JSON. The saved JSON keeps its original
`assets` list. Use the new run to check the change; existing materialization
records are not rewritten.

The current gateway `job/status` action only acknowledges the command, so it
cannot be used to prove that extraction has finished.

## Run the existing batch client

`batch_document_ocr.py` already submits through `/api/v1/invoke` using the Marie
client. It also listens for `extract.completed` events and downloads the results.
Its configuration needs the existing `storage` and `queue` sections in addition
to its API settings.

```bash
.venv/bin/python -m examples.batch_document_ocr \
  --config examples/config.dev.json \
  --pipeline default \
  --input /path/to/images \
  --output_dir /path/to/annotations
```

Batch metadata files are saved in `--output_dir`. Generated assets are cached
under `~/.marie/generators/<frame-hash>/`.

`batch_document_ocr_dir.py` runs the OCR engine locally and does not use the gateway.
