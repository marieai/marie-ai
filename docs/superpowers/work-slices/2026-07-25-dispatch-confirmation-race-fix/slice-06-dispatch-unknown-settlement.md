# Slice 06: Settle Unknown Dispatches

**Status:** Proposed

**Depends on:** [Slice 04](slice-04-structured-scheduler-outcomes.md), [Slice 05](slice-05-monotonic-dispatch-audit.md)

**Parent task:** [Task 6](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-6-reconcile-post-detach-unknown-dispatches)

## Scope

Add bounded reconciliation for dispatches that crossed `send_detached` but did not produce admission or rejection before the scheduler confirmation budget.

## Objective

Keep the active attempt and its capacity safe while delivery is ambiguous, then settle once on late admission, safe pre-RPC rejection, worker adoption, terminal state, or recovery.

## Touchpoints

- `marie/scheduler/psql.py`
- `marie/scheduler/runtime.py` if task tracking is shared there
- `tests/unit/scheduler/test_dispatch_unknown_reconciliation.py`
- dispatch-race unit tests

## Work

- Start one tracked reconciler per `run_attempt_id` only for `DispatchOutcome.unknown`.
- During the bounded grace period, renew the matching scheduler ticket and run lease and await shielded admission without holding scheduler locks.
- Resolve late admission to `dispatched` and late safe pre-RPC rejection through normal fenced failure cleanup.
- Stop scheduler renewal only when worker capacity adoption is explicitly observed, the attempt is terminal, or recovery fenced it.
- On terminal state, separate durable settlement from capacity release:
  - `AttemptLifecycleService` settles job, attempt, frontier, and DAG state.
  - If worker adoption was observed, worker terminal cleanup releases the ticket.
  - If worker adoption was not observed, the reconciler releases the scheduler-owned ticket.
- On recovery or replacement, best-effort cancel local work and release any scheduler-owned ticket.
- On grace expiry, stop extending the run lease so normal recovery can fence the attempt; do not directly invent a terminal result.
- Track reconciliation tasks through scheduler shutdown and surface task failures.

## Non-goals

- Do not treat timeout itself as job failure.
- Do not use semaphore TTL as normal successful settlement.
- Do not infer worker ownership solely from a terminal event.
- Do not add a separate reconciler service or database schema.

## Acceptance criteria

- Late admission and late safe rejection each settle once.
- Unknown capacity remains occupied while scheduler-owned.
- Worker adoption transfers release responsibility exactly once.
- Terminal before adoption releases the scheduler-owned ticket exactly once.
- Terminal after adoption does not trigger scheduler release; worker terminal cleanup releases exactly once.
- Concurrent terminal and reconciliation produce one durable terminal transition and one ticket release by the recorded owner.
- Recovery and shutdown leave no reconciler task or scheduler-owned ticket behind.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_unknown_reconciliation.py \
  tests/unit/scheduler/test_dispatch_confirmation_race.py -q
~~~

