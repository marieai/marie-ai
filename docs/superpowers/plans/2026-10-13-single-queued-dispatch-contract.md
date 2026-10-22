# Single queued-dispatch contract implementation plan

> **For agentic workers:** Use superpowers:executing-plans or focused parallel workers for independently owned components. Follow red/green checks and record verification in the analysis checkpoint. Do not commit without an explicit user request.

**Goal:** Implement the approved single durable queued-dispatch contract across Marie-AI and its Studio consumers.

**Architecture:** Queue enablement selects the durable producer/dispatcher; direct batch remains available when disabled. Version selectors and legacy transport disappear from the supported path. Numeric schema checks validate current persisted work while mode/capability observations replace protocol labels.

**Tech Stack:** Python 3.12, Redis/Valkey Lua, OpenTelemetry/OpenInference, TypeScript/React/tRPC, Helm/Compose.

**Spec:** ../specs/2026-10-13-single-queued-dispatch-contract-design.md

## Global constraints

- `LLM_QUEUE_ENABLED` is the only queue/direct selector; queued mode is `queued-dispatch`, direct mode is `direct-batch`.
- Durable envelope/store schema is integer `1`; keys are `llm:dispatch:{fabric:<fabric-group-id>}:`.
- Preserve current telemetry, trace flags/state, business digest/idempotency, and affirmative remote-settlement evidence.
- Treat this as a fresh install; initialization and producer attachment use current fabric policy without historical namespace scans.
- No deployed-store access, live provider workload, automatic data migration, or commits.
- Preserve user-owned edits, especially `.run/marie gateway-FILE.run.xml`.

## Review focus

- Disabled queue startup must need neither fabric nor registered policy/store access.
- Initialization must validate current fabric limits without unrelated namespace scans.
- Abandoned tombstones must retain schema, settlement identity, and trace headers.
- Empty/mixed/missing-mode runtime observations must never imply empty stored work or sampling availability.
- Studio scheduler initialization must not imply registered destination readiness.

## Task 1: Core contract and storage

**Files:** `completion_contract.py`, `llm_queue/{config,queue_keys,store,producer,scheduler,queue_io}.py`, Lua store, `batch_processor.py`, `openai_engine.py`, `llm_ops.py`, `multimodal_ops.py`, annotator util, core tests.

**Interfaces:** `QueuedCompletionEnvelope` owns durable calls and integer `schema_version=1`; `QueuedDispatchProducer`; `BatchProcessor.uses_queued_dispatch`; `RequestStore(url, *, fabric_id, limits=None, _join_existing=False, io_timeout_seconds=2)` has no version parameter. Shared connection construction remains available in a transport-only module. Remove `LlmQueueConfig` and obsolete envelope/reply/list execution support after moving imports.

- [x] Write schema/namespace/config-selection and fabric-initialization regressions; observe expected failures.
- [x] Implement canonical names/schema, current fabric policy checks on store creation/producer attachment, and one producer path.
- [x] Preserve durable scheduling/cost helpers while removing list transport dependencies.
- [x] Port useful legacy tests to durable behavior and run core/store/producer/telemetry checks on owned Redis and Valkey.

Expected: Unsupported/bool/missing schemas fail; startup uses only current fabric policy; direct mode remains independent; durable trace/admission/settlement cases pass.

## Task 2: Gateway lifecycle

**Files:** `marie/serve/runtimes/gateway/marie/{llm_dispatch_runtime,dispatch_policy}.py`, `marie_gateway.py`, gateway/policy tests.

**Interfaces:** Consume Task 1 config/store names; use `_start_queued_dispatch` and `dispatch_start_failed`; queue-disabled startup is a no-op. Static registered policy and registered database policy survive; ordinary legacy lanes cannot become trusted destinations.

- [x] Add red tests for a single default enabled path, policy/readiness failures, disabled startup, and missing credentials.
- [x] Remove legacy startup/stop/factory branches, normalize startup failures, and update routing-projection gate.
- [x] Run focused gateway/policy tests and actual dispatcher integration after Task 1.

Expected: Enabled misconfiguration raises bounded `RuntimeFailToStart`; no fallback occurs; disabled startup works without queue dependencies.

## Task 3: Observations and telemetry

**Files:** `llm_queue/{registry,metrics,request_dispatcher}.py`, instrumentation attributes, snapshot/metrics/history/telemetry tests.

**Interfaces:** Emit `mode='queued-dispatch'`, `marie.llm_dispatch.mode`, `MarieSpanAttributes.LLM_DISPATCH_MODE`; metric execution API accepts mode instead of contract version. Dispatcher health declares boolean `sampling_available`. Registry mismatches raise `SnapshotUnavailable('runtime_mode_mismatch')`.

- [x] Add red empty/mismatch/capability and metric/span vocabulary regressions.
- [x] Replace version aggregation/defaults and label-based capability inference, preserving bounded read budgets and privacy.
- [x] Run observation/metrics tests, then actual history/HTTP propagation checks.

Expected: Empty registry has zero registered/running counts, unavailable pending metadata, and sampling false; missing/conflicting modes fail; privacy and trace chain remain intact.

## Task 4: Studio consumers

**Files:** `llm-dispatch-operator.ts`, `llm-queue.router.ts`, associated event/router tests, `types/debug.ts`, `DebugView.tsx`, scheduler readiness/UI tests, runtime/scheduler docs.

**Interfaces:** Consume `mode`, actual sampling capability, and bounded error categories; remove `dispatchVersion`, expose `dispatchPolicyConfigured: boolean` independently of execution mode.

- [x] Add red runtime projection/readiness/UI behavior tests.
- [x] Update projection/types/debug gate/display, readiness, fixtures, and docs without weakening fabric/privacy validation.
- [x] Run scoped API/frontend checks; verify m3top accepts resulting snapshots without unnecessary Rust edits.

Expected: Queued runtime panel remains visible; incomplete policy is unready; invalid modes/fabrics are rejected; stored registered policy survives normal edits.

## Task 5: Deployment and stress names

**Files:** Helm queue helpers/values, Compose queue environment, gateway launcher, example config, stress dispatch scripts, version-named worker/evidence/suite files and their imports, relevant docs/tests.

**Interfaces:** Consume canonical producer/store/startup names and mode/schema output. Remove queue version settings while preserving etcd/Helm API/model versions.

- [x] Add red deployment/no-selector and canonical-runner checks.
- [x] Rename suites/helpers by responsibility with all dynamic imports/monkeypatches; update deployment/stress configuration and documentation.
- [x] Run Helm/Compose/shell/stress reporting tests and import/CLI smoke checks.

Expected: No obsolete selectable queue version remains; registered policy prerequisites and fresh-install setup are documented; canonical tools resolve.

## Task 6: Integrate and review

- [x] Review cross-task interfaces and resolve failures with source-grounded regressions.
- [x] Run focused integration, complete relevant suites, lint/format/diff checks, package builds/imports, Studio scoped checks, and m3top compatibility.
- [x] Perform independent review and fix material findings with red/green coverage.
- [x] Update the analysis checkpoint with exact counts, baseline limitations, owned-fixture cleanup, and remaining release gates.

Expected: Approved scope implemented and reviewable; no production migration or collector/provider qualification is claimed from local fixtures.

Fresh-install correction: User clarified there is no deployed historical queue
state. Removed migration-only guards and coverage; preserve current schema, limits,
fencing, and settlement checks. See the checkpoint for follow-up verification.
