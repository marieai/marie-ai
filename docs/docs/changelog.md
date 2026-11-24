---
sidebar_position: 3
---

# Release notes

Release notes are grouped by tagged release, in reverse chronological order. Each full comparison includes the complete commit history for that release. The repository [changelog](https://github.com/marieai/marie-ai/blob/develop/CHANGELOG.md) also retains earlier release notes.

## Marie AI 5.2.2

Released: 2026-11-24 · [Full changes since 5.2.1](https://github.com/marieai/marie-ai/compare/v5.2.1...v5.2.2)

### Fixes

- Register `DocList[AssetKeyDoc]` as the request schema for `/annotator/llm` and `/annotator/schema`. Workers preserve asset keys and selected pages instead of decoding these requests as `LegacyDocument`.
- Roll out the 5.2.2 image to the LLM annotation workers to apply this fix.

## Marie AI 5.2.1

Released: 2026-11-24 · [Full changes since 5.2.0](https://github.com/marieai/marie-ai/compare/v5.2.0...v5.2.1)

### Fixes

- Accept document submissions without a known page count. Direct-batch submissions skip queue routing; queued submissions use `LLM_QUEUE_DEFAULT_PAGE_COUNT`, which defaults to `1`, when the count is unknown.
- Persist requested capacity limits separately from active usage. Reservations and availability calculations honor a reduced target while existing reservations drain.
- Log each incoming job ID and queue at `INFO`. Queued LLM decisions include the selected pools and effective page count. A submission that uses the fallback count logs at `WARNING`. Routine successful gRPC event deliveries move to `DEBUG`.
- Set the first-boot PostgreSQL connection limit to 1,000 and configure `pg_stat_statements` to track nested statements and omit utility commands.
- Print the `gh release create` command for manual publishing, including how to use the notes from an existing annotated tag.

## Marie AI 5.2.0

Released: 2026-11-20 · [Full changes since 5.1.0](https://github.com/marieai/marie-ai/compare/v5.1.0...v5.2.0)

### Extraction and document outputs

- Add schema-extraction planning and persist validated extraction results per DAG. Ground values against OCR while preserving page identity and nested table-row identity.
- Improve schema-extraction configuration, prompts and OCR-mode handling. Extend LongExtractBench with source titles, source headings, page context, refinement, aggregation and recorded evaluation results.
- Propagate strict OCR errors and failed S3 writes instead of treating incomplete output as successful extraction.
- Track the actual uploaded metadata file and generated files as workflow materializations, with confirmed storage URIs and content-based versions. Replace synthetic `ocr/text`, `ocr/bboxes` and `ocr/confidence` records with deduplicated file records; skip incomplete materializations when storage operations fail.
- Preserve original plugin artifacts, including Markdown, upload them and include their URIs in extraction metadata. Scope asset listings to the current extraction.
- Add a gateway extraction CLI for local files and existing URIs, configurable extraction modes, completion waiting and result downloads. Update the OCR examples to use `/api/v1/invoke`, reuse the existing configuration and save downloaded metadata under its original stored filename.
- Use local GPT-2 tokenizer assets for TrOCR when available.

### Runtime, storage and deployment

- Add a shared registry for fabrics, gateways and storage targets. Resolve S3 credentials from registered configuration and pin targets by revision and fingerprint.
- Scope sensor polling and job routing to the configured fabric.
- Require authentication in the bundled Valkey deployment, persist its ACLs and authenticate health checks.
- Use configured PostgreSQL schemas for vector, metadata and model tables, with migration scripts for existing installations.
- Replace FerretDB with PostgreSQL-native DocumentDB in Compose and Helm. Add authenticated initialization, database creation and readiness checks; scope scheduler cron cleanup to its database.
- Align database bootstrap and Helm-bundled schemas, including sensor, knowledge-base and shared-registry definitions. Set the bootstrap database explicitly, ignore local `psql` customizations and pin the AppImage database image.
- Validate enabled gateway API-key configuration before starting listeners.
- Add SGLang deployment profiles for Qwen3.8 NVFP4 and Ansible support for replica provisioning, preflight checks, validation and lifecycle management. Update cluster provisioning and local runtime configurations.

### Operations and release tooling

- Move the standalone `m3top` terminal into Marie-AI as a top-level Rust crate, with a Linux release workflow, installation documentation and safeguards against stale gateway/navigation responses.
- Restore authorized scheduler diagnostics with independent snapshot timeouts. Add gateway metric identity and capacity-freshness reporting.
- Refresh and commit `uv.lock` before release image builds; make image builds validate the lock without modifying it. Allow retrying the same release version without amending previous commits.

### Upgrade notes

- Configure the bundled Valkey credentials and persistent ACLs before starting authenticated services.
- Update database bootstrap/schema inputs together with the runtime and review the DocumentDB deployment settings when replacing a FerretDB installation.
- Extraction materializations now describe stored files. Existing workflow records are not rewritten; inspect a new extraction run to verify file URIs.
- See the [database upgrade guide](https://github.com/marieai/marie-ai/blob/v5.2.0/docs/docs/getting-started/deployment/scheduler-database-upgrades.md) and [document extraction example](https://github.com/marieai/marie-ai/blob/v5.2.0/examples/document_ocr.md).

## Marie AI 5.1.0

Released: 2026-10-21 · [Full changes since 5.0.4](https://github.com/marieai/marie-ai/compare/v5.0.4...v5.1.0)

### LLM dispatch and routing

- Add durable Gateway-owned LLM dispatch with bounded admission, fair scheduling, cancellation, deadlines, recovery and retained execution history.
- Classify LLM work at submission, persist immutable routing policies and bind queued calls to durable routes. Select endpoint replicas at dispatch time and remove caller-selected routing/pool overrides.
- Make fairness-charge recovery durable, retain unsettled route/replica reservations until matching remote settlement and fence refunds and recovery operations.
- Consolidate queued execution on the canonical `queued-dispatch` and `direct-batch` modes, using `schema_version=1` and fabric-scoped queue keys.
- Capture raw physical provider attempts before normalization. Persist required attempt evidence before response callbacks and acknowledgement, retain results after observer failure and reject malformed completions without stranding capacity.
- Fence policy activation, reset idle bindings, reject stale routes and add actionable startup diagnostics and endpoint probes.
- Expose routing recovery diagnostics, held reservations and operator release controls. Link delayed execution traces back to Gateway submissions.
- Recover complete JSON from truncated responses and bound repetition recovery. Expand OpenAI-compatible sampling, response-format and reasoning options, including `extra_body` and `chat_template_kwargs`.

### Extraction, models and benchmarks

- Move model registration and URI handling into `marie-registry`. Load model metadata, pinned revisions, provider tokenizers and complete runtime/processor/prompt/generation profiles from an external model catalog.
- Validate annotation responses using saved completion evidence. Bind processing units, rendered prompts, configuration and source identity before submission; save prepared images and hashes before releasing memory.
- Publish validated artifacts only after evidence persistence and validation succeed. Restore annotation artifacts across workers and use mounted layout prompts and prepared disk frames.
- Add ParseBench dataset tooling, page schemas, context providers and strict response handlers. Add the document configuration benchmark library with candidate import, package validation and preflight tooling.
- Improve record-backed section population, tolerate malformed record source spans and preserve page IDs for empty table cells and derived fields.
- Preserve first/middle/last name order when a middle initial matches a census first name. Guard missing OCR lines and release image buffers after annotation failures.

### Scheduler, jobs and state

- Enforce explicit `queue_names` subscriptions across admission, dispatch, recovery and repair. Omitting `queue_names` preserves consumption from all queues; validate DAG queue identity.
- Optimize scheduler paging, indexes and throughput queries. Compute frontier SLA summaries outside the event-loop hot path.
- Preserve deferred ready entries and planner inputs, select the latest matching job by policy, and commit branch completion and skipped work atomically.
- Bound branch-condition waits and runtime shutdown; normalize branch execution results, preserve loading errors and route unhashable switch values to the default branch.
- Validate numeric configuration, tolerate malformed runtime estimates, release topology-build locks and make unsupported maintenance/reset operations explicit.
- Reset completed DAGs and jobs to a fresh schedulable state.
- Fence status writes with the claimed epoch, reconcile semaphore counters before release, retry desired-state creation after a deleted winner and tolerate optional document schema changes.
- Remove unused state writers and mutable planner defaults. Verify heartbeat/slot reuse behavior and manage job locks with context managers.
- Retain structured job failures and supervisor finalizers until terminal delivery. Clean up dead worker registrations; add terminal-rejected attempt indexing, DAG-attempt purge support and an orphan-cleanup runbook.
- Repair event-publisher startup/restart and idempotent unsubscribe. Bound subscriber execution, queue capacity and shutdown; drain callbacks reliably and report subscriber failures and queue-full drops.
- Harden load balancing and connection lifecycle: bound retries, preserve usage counts across replica changes, enforce circuit probe limits, reject acquisitions after shutdown and preserve RPC error classification through resets and retired channels.

### Plugins, observability and deployment

- Preflight embedded plugins at executor startup so missing or incorrect plugin configuration is detected before the first extraction.
- Harden plugin-daemon execution, Python environment setup, I/O tunnels, runtime ownership and agent-tool integration. Align MCP tools with the authenticated Gateway API.
- Add sandbox publication deployment, executor resource/identity safeguards, mock executors and signed plugin-runtime bindings. Improve daemon container packaging and portable runtime paths; remove obsolete plugin stubs.
- Consolidate ClickHouse telemetry and organization/workspace isolation. Provision restricted SQL-explorer readers, simplify deployment and drain active queries when a workspace reader is removed or suspended.
- Extend OpenTelemetry, ClickHouse and HyperDX integration, log facets and dispatcher/scheduler diagnostics. Add matching Compose and Helm components, including FerretDB integration.
- Add AIMock capacity/fault controls, queued/direct stress comparisons, qualification tooling and a Qwen FP8 launch script.

### Breaking changes and upgrade notes

- Model lookup requires an external `models.yaml` catalog selected through registry configuration or `MARIE_MODEL_CATALOG`. Hardcoded engine model factories/maps no longer supply complete model configuration. See the [registry configuration guide](https://github.com/marieai/marie-ai/blob/v5.1.0/packages/marie-registry/README.md).
- Legacy queue contract selectors, list transport modules and version-named queue APIs are removed. Queued deployments require the canonical schema/namespace and a complete registered routing policy; existing queue data is not migrated automatically. See the [dispatcher setup guide](https://github.com/marieai/marie-ai/blob/v5.1.0/docs/docs/guides/llm-dispatcher-setup.md).
- Scheduler schema 96 requires matching scheduler binaries and SQL. Explicit `queue_names` lists now restrict consumption. Apply the matching database upgrade before starting the new scheduler; see the [database upgrade guide](https://github.com/marieai/marie-ai/blob/v5.1.0/docs/docs/getting-started/deployment/scheduler-database-upgrades.md).

## Marie AI 5.0.4

Released: 2026-08-14 · [Full changes since 5.0.3](https://github.com/marieai/marie-ai/compare/v5.0.3...v5.0.4)

### Fixes

- Preserve datasource metadata across job retries. Copy the nested document payload before parsing so executor-parameter sanitization cannot mutate the scheduler's cached retry metadata.
