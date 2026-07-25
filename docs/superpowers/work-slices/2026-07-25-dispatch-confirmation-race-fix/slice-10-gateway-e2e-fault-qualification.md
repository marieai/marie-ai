# Slice 10: Qualify Gateway Faults End to End

**Status:** Proposed

**Depends on:** [Slices 01–09](README.md#slice-index)

**Parent task:** [Task 10](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-10-add-the-concrete-gateway-end-to-end-fault-suite)

## Scope

Add two gated, request-scoped gateway fault scenarios to the existing `mock_parallel_subgraphs` workload: delayed admission and no-replica rejection.

## Objective

Prove with a real gateway, PostgreSQL scheduler, existing eight mock executors, and per-attempt traces that a post-detach timeout reconciles safely and a known pre-RPC no-replica failure rejects before the confirmation budget.

## Touchpoints

- source and mounted mock scheduler configuration
- scheduler, JobSupervisor, GatewayJobDistributor, networking, and Marie gateway fault plumbing
- `tools/stress/gateway_e2e_stresser.py`
- `tools/stress/analyze_scheduler_trace.py`
- `marie/utils/scheduler_trace.py`
- `stress-test.sh`
- focused fault-hook and analyzer tests

## Work

- Add closed, mock-config-gated controls for admission delay and no-replica rejection.
- Require finite bounded values, exact executor match, unique `stress_run_id`, and valid timeout ordering.
- Apply delayed admission after `send_detached` and before desired-state admission.
- Inject no replicas before connection acquisition without mutating shared discovery state.
- Add stresser arguments and correlation metadata for both scenarios.
- Add per-attempt analyzer assertions for the exact event sequences in Task 10.
- Add independent `gateway-dispatch-timeout-e2e` and `gateway-no-replicas-e2e` targets.
- Refuse stale trace reuse and exit nonzero on workload or invariant failure.
- Keep the scenarios separate so one test proves ambiguous reconciliation and the other proves fast known rejection.

## Non-goals

- Do not delay mock executor processing to simulate dispatch timeout; processing begins after admission.
- Do not remove a replica globally or mutate production discovery state.
- Do not inject executor crash or restart behavior.
- Do not copy the source mock configuration into `/mnt/data` automatically during implementation or review.

## Acceptance criteria

### Delayed admission

- At least 16 affected attempts cross `send_detached`, time out as unknown, admit later, pass the worker fence, and reach exactly one terminal or recovery result.
- Timeout itself writes no dispatch-failure terminal event and releases no slot.
- No affected attempt remains unknown; no duplicate invocation or capacity overcommit occurs.

### No replicas

- At least 16 affected attempts reject as `no_available_replicas` before the confirmation budget.
- No affected attempt emits `gateway_dispatch_timeout` or `gateway_rpc_started`.
- No affected request invokes executor code, enters `dispatch_unknown`, or leaks a slot.

### Both

- No error field equals `False` or `None`.
- Analyzer failures print `job_id` and `run_attempt_id` and exit nonzero.

## Verification

First verify the gated hooks and analyzers:

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_test_hook.py \
  tests/unit/tools/stress/test_dispatch_timeout_trace_assertions.py \
  tests/unit/tools/stress/test_no_replicas_trace_assertions.py -q
~~~

After review, the operator synchronizes the source mock configuration:

~~~bash
cp config/service/mock/marie-mock-scheduler-test.yml \
  /mnt/data/marie-ai/config/service/mock/marie-mock-scheduler-test.yml
~~~

Run each qualification with a new run id and trace path:

~~~bash
./stress-test.sh gateway-dispatch-timeout-e2e
./stress-test.sh gateway-no-replicas-e2e
~~~

The full explicit `gateway_e2e_stresser.py` commands and analyzer contracts remain in Task 10 of the parent plan.
