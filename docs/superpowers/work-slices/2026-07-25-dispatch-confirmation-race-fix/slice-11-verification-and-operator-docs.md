# Slice 11: Run Verification and Update Operator Docs

**Status:** Proposed

**Depends on:** [Slice 10](slice-10-gateway-e2e-fault-qualification.md)

**Parent task:** [Task 11](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-11-full-verification-and-operator-documentation)

## Scope

Run the complete verification matrix, update correctness tooling and operator guidance, and publish evidence against the parent plan's final acceptance report.

## Objective

Make the gateway dispatch lifecycle observable and operable after the fix, with no legacy boolean error payloads, stale callback wiring, unresolved unknown attempts, or hidden database-test gaps.

## Touchpoints

- `tools/stress/scheduler-reliability.md`
- `tools/stress/README.md`
- scheduler correctness and reliability tools
- high-availability checks that enumerate attempt states
- all production and test paths changed by Slices 01–10

## Work

- Document `dispatch_unknown`, `dispatch_stalled`, and bounded reconciliation.
- Document pre-RPC rejection, post-RPC ambiguity, and executor application response as distinct outcomes.
- Document budget ordering and dynamic gateway configuration preservation.
- Update correctness queries and reports for the new temporary attempt states.
- Add the dispatch, gateway transport, stale-attempt, and capacity-adoption metrics listed in Task 11.
- Verify legacy `confirmation_event` wiring and boolean error strings are absent.
- Run focused, integration, lint, reproducer, and live gateway qualification gates.
- Produce the final acceptance counts defined in the parent plan.

## Non-goals

- Do not weaken assertions to make a live run pass.
- Do not treat a skipped PostgreSQL test as verification.
- Do not add executor-reliability metrics to this plan.

## Acceptance criteria

- All required unit and integration suites pass, with database-backed skips called out explicitly.
- Ruff passes on every touched Python area.
- The original deterministic reproducer now guards the corrected behavior.
- Both gateway E2E targets pass with independent run ids and trace files.
- Final report shows zero unresolved unknown attempts, false error strings, duplicate invocations, capacity overcommit, and no-replica timeouts.
- Operator docs explain who settles durable terminal state and who releases scheduler-owned versus worker-owned capacity.

## Verification

~~~bash
.venv/bin/python -m pytest tests/unit/scheduler/ tests/unit/job/ -q
.venv/bin/python -m pytest \
  tests/unit/serve/networking/ \
  tests/unit/serve/runtimes/gateway/ \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py -q
.venv/bin/python -m pytest tests/integration/scheduler/ -q

.venv/bin/ruff check \
  marie/scheduler/ \
  marie/job/ \
  marie/serve/networking/ \
  marie/serve/runtimes/gateway/ \
  marie/serve/runtimes/servers/marie_gateway.py \
  marie/serve/runtimes/worker/request_handling.py \
  tools/stress/ \
  tests/unit/scheduler/ \
  tests/unit/job/

./stress-test.sh reproduce-dispatch-race
./stress-test.sh gateway-dispatch-timeout-e2e
./stress-test.sh gateway-no-replicas-e2e
~~~
