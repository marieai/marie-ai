# Use one queued-dispatch contract

Status: Approved for implementation by the user on 2026-10-13

Source review: Incorporates verified Claude feedback on shared dependencies and
Studio consumers. User clarified this is a fresh install with no deployed legacy
queue state; historical namespace inspection and migration are out of scope.

## Outcome

Marie supports one durable queued-dispatch contract. `LLM_QUEUE_ENABLED` selects
queued dispatch or direct batch execution. No queue protocol version selects an
implementation, configures a deployment, labels a runtime, or names a producer.

Preserve the producer trace propagation, sampling flags, vendor trace state,
content-free execution history, and settlement safeguards from the current work.

## Confirmed inconsistencies

- `LlmQueueRuntimeConfig` accepts `v2` and `v3`, defaults to `v2`, and reads
  `LLM_QUEUE_CONTRACT_VERSION`.
- `GatewayLlmDispatchRuntime` retains legacy startup and shutdown branches beside
  `_start_v3`; a registered policy overrides the configured version.
- `BatchProcessor.uses_v3_queue` contains a legacy runtime-config exception, and
  `_get_queued_executor` can construct the legacy executor.
- `RequestStore` requires `version='v3'`; envelopes and Lua records also carry a
  string version. Fabric keys use `llm:v3:{fabric:<id>}:`.
- The legacy list path uses `list:llm:v2:requests:<pool>`,
  `list:llm:v2:replies:<producer>`, and `key:llm:v2:producer:<id>:alive`. These
  keys are not fabric-scoped. Helm, Compose, and the gateway launcher default
  enabled deployments to this path when no version is supplied.
- Runtime snapshots, metrics, stress tools, deployment configuration, and tests
  repeat the version strings. Studio's runtime projection explicitly accepts
  only `v2`, `v3`, and `mixed`, so changing the backend alone breaks that consumer.
- The etcd image tag `quay.io/coreos/etcd:v3.7.0`, Helm chart API versions, and model
  versions are unrelated and remain valid.

## Public configuration and execution

Remove `queue_contract_version`, `llm_queue_contract_version`,
`LLM_QUEUE_CONTRACT_VERSION`, and Helm's `global.llmQueue.contractVersion`.
Remove the version resolver and all version-driven runtime branches.

Use the durable producer and `RequestDispatcher` whenever the queue is enabled.
Direct batch execution remains available when it is disabled. An enabled queue
with unavailable storage, inconsistent fabric identity, missing endpoint
credentials, or invalid dispatch policy fails closed; it never falls back to
direct execution or the legacy list dispatcher.

Enabled deployments must provision registered endpoints or endpoint groups,
credential bindings, lanes, limits, and a consistent fabric identity before
starting the runtime. An empty policy is invalid; removing the version selector
does not create a valid policy automatically. This is a fresh deployment, with
no old queue state to inspect or migrate.

Retain supported URL aliases, explicit static registered dispatch policy, and
database-backed registered runtime dispatch policy. Database loading means
`load_runtime_dispatch_policy`, or `persisted_dispatch_policy` reading complete
registered policy from existing scheduler metadata. Ordinary legacy scheduler
lanes and trusted-runtime URL reuse through `validate_legacy_lane_endpoints` are
not a second surviving database execution form. Do not silently turn their URLs
into registered destinations or invent credential bindings.

When the queue is disabled, initialization does not require queue storage, a
registered dispatch policy, or queue fabric binding.

Normalize enabled startup failures, including credential resolution, into
`RuntimeFailToStart` with `_last_error='dispatch_start_failed'` and bounded public
messages. Credential resolution currently happens outside the guarded startup
block and raises a bare `ValueError`; include it in the common failure boundary.
Never include credential values or private endpoint URLs in those messages.

Remove the obsolete list request/reply contract and executor from the supported
execution path. Separate shared scheduling and client construction from the
legacy list transport before removing its modules:

- Preserve `DrrLaneConfig`, durable metadata selection/charge/refund behavior in
  `DrrLaneScheduler`, and `prepared_cost_units`, which the durable producer and
  dispatcher use. Remove or separate list-envelope selection and
  `ListQueueClient`/`MalformedQueueRequest` dependencies within that same module;
  do not delete `scheduler.py` wholesale.
- Preserve the Valkey/Redis connection factory `_build_sync_client`, currently
  imported by `RequestStore` and stress fixture inspection from `queue_io.py`.
  Move that shared factory and its callers together if retiring `queue_io.py`.
  Its purpose is shared store transport, not a generic external list API.
- Retire legacy submitter, reply-reader, dispatcher, and list-client dependencies
  only after current production/tooling imports have been moved. Port useful
  behavioral coverage to the durable path rather than retaining a selectable
  legacy implementation to satisfy old fixtures.

## Names and serialized data

Use responsibility-based names:

| Current name | Replacement |
| --- | --- |
| `V3Producer` | `QueuedDispatchProducer` |
| `uses_v3_queue` | `uses_queued_dispatch` |
| `_start_v3` | `_start_queued_dispatch` |
| `QueuedCompletionEnvelopeV3` | `QueuedCompletionEnvelope` after retiring the legacy envelope |
| `COMPLETION_QUEUE_CONTRACT_VERSION` and `COMPLETION_QUEUE_CONTRACT_VERSION_V3` | Retire string-version constants; use `COMPLETION_QUEUE_SCHEMA_VERSION = 1` for the durable data format |
| `MarieSpanAttributes.LLM_DISPATCH_CONTRACT_VERSION` / `marie.llm_dispatch.contract_version` | `MarieSpanAttributes.LLM_DISPATCH_MODE` / `marie.llm_dispatch.mode` |
| `v3_start_failed` | `dispatch_start_failed`, including backend and Studio category allowlists |
| `v2_endpoint_policy_requires_v3` | Retire with the legacy destination validator; report missing registered policy through the common startup failure boundary |
| `LlmQueueConfig` compatibility alias | Retire; use `LlmQueueRuntimeConfig` or `LlmQueueProducerConfig` according to the caller's responsibility |
| `llm-v3-*` thread names | `llm-dispatch-*` |
| `llm_v3_aimock_e2e.py` | `llm_queued_dispatch_aimock_e2e.py` |
| `test_v3_*.py` | Names describing the covered producer, lifetime, fairness, history, or reporting behavior |
| `v3_worker_fixture.py` | `queued_dispatch_worker_fixture.py` |
| `v3_qualification_evidence.py` | `queued_dispatch_qualification_evidence.py` |
| `llm-dispatch-v3.example.yml` | `llm-dispatch.example.yml` |

Preserve behavioral test coverage while updating names and fixtures. Update all
imports, module invocations, documentation links, and stress-runner references.
Land renames with dynamic imports in `llm_dispatch_failure_scenarios.py` and the
producer/startup monkeypatches in `llm_dispatch_deployment_child.py`. Update
`MarieGateway._start_gateway_background_runtimes` to start routing projection
for an enabled, successfully started durable runtime, without checking a version.

Use numeric `schema_version=1` for the durable envelope and request record.
Require an integer with the supported value when reading serialized envelopes;
reject malformed or unsupported schemas. This is an internal data-format check,
not a selectable execution version. Remove `RequestStore`'s `version` argument.
Every new durable request record, including abandoned held-reservation
tombstones, carries `schema_version=1`. Lua validates the schema for every state;
there is no abandoned-record exemption. Old schema-less tombstones stay in their
old namespace and block cutover until affirmatively settled; do not reinterpret
them as new-schema records.

Trace context stays outside the business-content digest. Re-admission retains
the original trace context, deadline, and identity. Held-reservation tombstones
retain trace identity without retaining prompt or result content.

## Runtime observations and consumers

Replace feature-level `contract_version` with `mode='queued-dispatch'` in gateway
snapshots, dispatcher health, telemetry attributes, metrics, and stress reports.
Use `mode='direct-batch'` where an execution-mode observation describes a disabled
queue. Preserve bounded categories and keep request/claim IDs out of metric labels.

The queued-dispatch registry reports `mode='queued-dispatch'`, even when no
dispatcher is registered: that describes the endpoint's scope, not evidence of
running work. In that empty case, registered/running dispatcher counts are zero,
`sampling_available=false`, and store-dependent metadata/pending counts are
unavailable rather than inferred empty. Direct mode belongs to the gateway's
execution-mode observation; direct executors are not registered as dispatchers.

All registered dispatcher rows must declare `mode='queued-dispatch'`. Missing or
conflicting modes fail snapshot projection as bounded
`runtime_mode_mismatch`; do not emit `mixed` or infer a historical version.
Add this category to observation consumers' allowlists where needed.

Each dispatcher explicitly reports sampling capability. Aggregate
`sampling_available` as `bool(dispatchers) and all(row.get('sampling_available')
is True for row in dispatchers)`; missing/false capability is unavailable. Keep
metadata-read success separate.
Never infer either from mode, schema, or the former version label.

Update the Marie Studio consumer end to end:

- `packages/api/src/server/trpc/routers/llm-dispatch-operator.ts`: project mode,
  capability, and bounded categories; remove the version allowlist.
- `packages/frontend/ui/src/types/debug.ts` and
  `packages/frontend/ui/src/components/DebugView.tsx`: update types, generated
  summary rows, the runtime-panel gate, and its displayed label. Gate the panel
  on queued-dispatch mode and the actual summary/capability, never `'v3'`.
- `packages/api/src/server/trpc/routers/llm-queue.router.ts`: remove
  `dispatchVersion`. Expose `dispatchPolicyConfigured: boolean` for registered
  policy readiness, independently of the gateway's selected execution mode.
  Scheduler initialization or default lanes alone are not ready dispatch policy.
- Check scheduler initialization/presets and associated UI tests. Preserve
  stored registered policy during ordinary scheduler edits; show incomplete
  destination registration explicitly instead of implying queue readiness.
- Update runtime projection, queue-router, event, frontend, and initialization
  fixtures/tests, plus `docs/content/monitoring/llm-dispatch-runtime.mdx` and
  `docs/content/infrastructure/llm-scheduler.mdx`. Remove default-version and
  automatic-policy-readiness claims; describe required policy provisioning.

`marie-studio/packages/m3top` currently has no `contract_version` references.
Verify its parser/rendering against the resulting snapshot; do not schedule an
unnecessary Rust rename or API change solely because it consumes the endpoint.

## Storage and initialization

Use `llm:dispatch:{fabric:<fabric-group-id>}:` for the durable store namespace.
Keep the fabric hash tag and identifier validation.

This is a fresh install. Store initialization and producer attachment operate on
the selected fabric's current policy; they do not scan historical namespaces or
require a migration/drain procedure. Do not implement compatibility state guards
for deployments that do not exist.

Keep normal validation of authoritative fabric limits, numeric record schema,
owner fencing, and remote settlement evidence. Timeouts, producer expiry,
ownership turnover, and elapsed uncertainty do not prove remote termination.

## Verification

- Configuration and startup tests prove one queue selector, direct-mode startup
  without queue dependencies, and one enabled path. Include enabled startup
  without registered policy, both database policy forms,
  missing named credentials, and normalized startup errors.
- Real Redis and Valkey tests prove admission idempotency, schema checks, fabric
  isolation, owner fencing, schema validation on abandoned tombstones,
  held-capacity cleanup, and initialization/attachment that use only current
  fabric policy. Startup must not scan unrelated namespaces.
- Actual producer/manifest/dispatcher/HTTP tests preserve the complete trace chain,
  retries, sampling, vendor state, and content privacy.
- Snapshot, Studio API/event/UI, and applicable m3top tests prove the new mode
  field works through the complete observation path. Cover no dispatchers,
  conflicting/missing mode, actual sampling capability, policy readiness after
  initialization, and runtime-panel visibility.
- Helm, Compose, shell launchers, and stress-report tests prove no queue-version
  configuration or version-dependent runner remains.
- Search changed production/configuration/tooling and test names for obsolete
  queue-version terminology, excluding unrelated dependency/protocol versions
  and this document's pre-change source snapshot.
- Run focused checks, package builds/imports, lint, formatting, and independent
  review. Report existing full-suite failures separately; do not claim live
  provider or collector qualification from owned fixture tests.

## Scope and preservation

Implementation spans Marie-AI and Marie Studio's API, frontend, and relevant
documentation. Preserve the current uncommitted telemetry work and unrelated
workspace changes. Create
no commit, deployment, live workload, or production data migration unless asked.
