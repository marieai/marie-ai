# Reproduce the queued LLM downstream handoff

Use `mock_llm_handoff` to run:

```text
START -> mock_llm_prepare -> annotator_llm -> mock_llm_reader -> END
```

Preparation reads the submitted image/PDF and writes synthetic, empty OCR metadata.
The real `DocumentAnnotatorLLMExecutor` invokes `LLMAnnotator` with the `mock-llm`
profile. The reader runs in a separate deployment, downloads the success marker
and every page's JSON directly from S3, then writes `handoff-receipt.json` with the
consumed data, DAG ID, hostname, and PID. It fails on missing or malformed output.
Use synthetic documents, all pages, and a fresh reference ID for each DAG.

This tests separate scheduler nodes/processes. To test separate physical hosts,
run the `mock_llm_reader` deployment on the second host using the same discovery
service and storage, and omit it from the first host's deployment configuration.

## Start AIMock

From the checkout:

```bash
docker compose --env-file /dev/null \
  -f Dockerfiles/docker-compose.mock-llm-programmatic.yml up -d --build
curl --fail http://127.0.0.1:4010/health
curl --fail http://127.0.0.1:4011/fault-profile
```

The provider URL is `http://127.0.0.1:4010/v1`. The gateway's active database
policy must point there. If the gateway runs in another container or on another
host, use an address reachable from that gateway and configure AIMock's bind
address accordingly. `OPENAI_API_BASE` alone does not change the queued provider.

## Configure and restart the services

Use this source checkout or a build containing `LlmHandoffExecutorMock` and the
`mock_llm_handoff` planner. Restart the gateway as well as the mock server so both
load the new code. The mock server config is
`config/service/mock/marie-mock-scheduler-test.yml`; the mounted copy is
`/mnt/data/marie-ai/config/service/mock/marie-mock-scheduler-test.yml`.
Both contain the preparation and reader deployments.
The annotator also needs `config/extract/TID-mock-llm/annotator/config.yml` and
`mock-llm.j2` under the corresponding mounted extract directory. Copy these
test assets from the checkout if the mount predates them.

In the gateway's `with.job_scheduler_kwargs.query_planners.planners`, use
`name: mock_planners` with `py_module: marie.query_planner.mock_query_plans`.
Older mounted configs may still point to
`tests.integration.scheduler.mock_query_plans`; that module does not register
this new planner. The submit queue remains `gen5_extract`.

Supply these settings through your usual launch environment on the gateway and
annotator worker. Replace the Valkey address if it runs elsewhere:

```bash
export LLM_QUEUE_ENABLED=true
export LLM_QUEUE_FABRIC_GROUP_ID=default
export LLM_QUEUE_URL=redis://127.0.0.1:6379/0
export OPENAI_API_BASE=http://127.0.0.1:4010/v1
export OPENAI_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
```

The generated API key is a disposable test value for the annotator's client
validation; this AIMock setup does not require a real provider credential.
Keep your existing PostgreSQL, S3, discovery, and gateway authentication settings.
Use your authenticated Valkey URL when the service requires credentials. The
mock annotator's YAML forwards `LLM_QUEUE_URL`; set it in the mock server launch
environment as well as the gateway's. Restart the mock server after changing
its environment. An unresolved legacy `LLM_QUEUE_VALKEY_URL` placeholder fails
URL validation before any request reaches Valkey.

The gateway must use `llm_queue.fabric_group_id: default` and
`llm_queue.llm_dispatch.policy_source: database`. The annotator logs
`LLM request submission mode: queued-dispatch`. Both use the same Valkey and
fabric. `LLM_QUEUE_POOL_ID` is ignored by this producer path: the database policy
selects the pool. The document-size policy selects `document-small` for 1–4
effective pages. See [dispatcher setup](../../docs/docs/guides/llm-dispatcher-setup.md)
for provisioning; do not rerun initial provisioning over an existing policy.

## Submit one document

Set `GATEWAY_API_KEY` to your existing gateway credential and `HANDOFF_INPUT_URI`
to a real synthetic image or PDF already in S3. Use a one-page input for the
first run. The preparation node creates its companion metadata; OCR inference
is not required.

```bash
.venv/bin/python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.json \
  --api-key "${GATEWAY_API_KEY:?Set the gateway API key}" \
  --s3-uri "${HANDOFF_INPUT_URI:?Set a synthetic document S3 URI}" \
  --job-count 1 \
  --submit-concurrency 1 \
  --job-name gen5_extract \
  --planner mock_llm_handoff \
  --ref-type mock_llm_handoff \
  --project-id mock-llm-handoff \
  --request-template tools/stress/mock_llm_handoff.invoke.json \
  --required-executor mock_llm_prepare \
  --required-executor annotator_llm \
  --required-executor mock_llm_reader \
  --skip-preflight \
  --terminal-timeout 90 \
  --min-terminal-completion-pct 100 \
  --report /tmp/mock-llm-handoff-report.json
```

`--skip-preflight` skips optional gateway diagnostics for keys without
`runtime-observability`. Remove it when the key has that scope on `default`.

The stresser generates a fresh reference ID. If submitting through another client,
use the invoke template with a fresh `request_id`, your API key, and the input URI.
Keep the `purge_annotators` feature so retries perform new LLM work.

## Locate where the run stops

For the submitted DAG, inspect states and actual routing with this read-only SQL:

```sql
SELECT job.id, job.name, job.state, job.dependencies,
       route.pool_id, route.projection_state, route.effective_page_count
FROM marie_scheduler.job AS job
LEFT JOIN marie_scheduler.llm_job_route AS route ON route.work_unit_id = job.id
WHERE job.dag_id = '<submitted-dag-id>'::uuid
ORDER BY job.created_on, job.id;
```

Successful evidence includes all three executor nodes completing, an LLM route
projected to `document-small`, AIMock's request counter increasing, and a receipt
at `s3://<bucket>/mock_llm_handoff/<reference-prefix>/handoff-receipt.json`.
The reader's job result also returns its exact `receipt_uri`.

If preparation finishes but annotation does not, check the route projection,
gateway dispatcher, Valkey/fabric configuration, and AIMock calls. If annotation
finishes but the reader never starts, inspect scheduler dependencies and reader
discovery/capacity. If the reader starts and fails, its error identifies the
missing or invalid storage artifact. A successful annotation response alone
does not prove that the downstream handoff worked.
