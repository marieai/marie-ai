# Slice 04: Return Structured Scheduler Outcomes

**Status:** Proposed

**Depends on:** [Slice 03](slice-03-pre-rpc-failure-propagation.md)

**Parent task:** [Task 4](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-4-return-structured-scheduler-outcomes-and-preserve-real-errors)

## Scope

Replace boolean enqueue results with structured scheduler outcomes and introduce dispatch budgets independent of candidate lease TTL.

## Objective

Classify one logical dispatch exactly once as confirmed, safely failed, or unknown, while retaining its reason and observed lifecycle stage.

## Touchpoints

- `marie/scheduler/psql.py`
- `marie/scheduler/postgres_scheduler_config.py`
- scheduler dispatch-race and dispatch-cycle unit tests

## Work

- Return `DispatchOutcome` from `enqueue()` and `_activate_and_enqueue_job()`.
- Add `dispatch_confirmation_timeout_seconds` and `pre_send_callback_timeout_seconds` with startup validation.
- Await shielded admission so timeout does not cancel later reconciliation input.
- On timeout, call `cancel_if_not_detached()` and classify from the recorded detach and RPC stage.
- Treat only pre-detach cancellation and structured pre-RPC rejection as safe immediate failures.
- Preserve post-detach and post-RPC ambiguity as `unknown`.
- Record one dispatch audit result after classification rather than from multiple exception branches.
- Pass `outcome.reason`, never a boolean or the outcome object, into failure metadata.
- Emit stage, owner, attempt, executor, and stress-run correlation fields.

## Non-goals

- Do not reconcile unknown outcomes yet.
- Do not change repository transition guards; that is Slice 05.
- Do not derive confirmation timeout from `lease_ttl_seconds`.

## Acceptance criteria

- Known setup and pre-RPC failures surface immediately with named reasons.
- Pre-detach timeout cancels and is safely failed.
- Post-detach timeout retains the attempt and slot as unknown.
- Post-RPC timeout remains unknown even when local cancellation succeeds.
- No failure metadata contains `False` or `None` as a string.
- Production budget validation preserves the ordering defined in the parent plan.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_confirmation_race.py \
  tests/unit/scheduler/test_dispatch_cycle.py -v
~~~

