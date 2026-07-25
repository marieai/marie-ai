# Slice 02: Add Dispatch Result and Handle Types

**Status:** Proposed

**Depends on:** [Slice 01](slice-01-freeze-dispatch-race-contract.md)

**Parent task:** [Task 2](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-2-add-dispatchhandle-sendfailure-and-dispatchoutcome)

## Scope

Introduce the lifecycle types used by the remaining slices without rewiring the production call path yet.

## Objective

Represent admission, failure stage, detach state, RPC start, cancellation safety, and scheduler outcome without booleans or unstructured callback tuples.

## Touchpoints

- `marie/job/dispatch_handle.py`
- `marie/job/send_failure.py`
- `marie/scheduler/dispatch_outcome.py`
- matching unit tests under `tests/unit/job/` and `tests/unit/scheduler/`

## Work

- Add immutable `DispatchAdmission` and structured `SendFailure` values.
- Add `DispatchHandle` with supervisor attachment, synchronous send detach marking, RPC-start marking, idempotent admit/reject, and atomic pre-detach cancellation.
- Preserve the admission future with `asyncio.shield` semantics after a caller timeout.
- Add `DispatchOutcome` constructors for `confirmed`, `failed`, and `unknown`.
- Keep truthiness compatibility only as a transition aid; never derive an error message from `bool(outcome)`.
- Normalize stable reasons, including `dispatch_timeout` and `no_available_replicas`.

## Non-goals

- Do not change `enqueue()` or gateway callbacks in this slice.
- Do not add reconciliation.
- Do not infer detach state from supervisor-task existence.

## Acceptance criteria

- `supervisor_started` remains reversible and distinct from `send_detached`.
- `mark_rpc_started()` permanently prevents later rejection from claiming a safe pre-RPC failure.
- Admission accepts exactly one result and remains awaitable after a shielded timeout.
- No type converts `False` or `None` into a failure reason.
- Focused unit tests cover the concurrency boundaries without production-duration sleeps.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/job/test_dispatch_handle.py \
  tests/unit/job/test_send_failure.py \
  tests/unit/scheduler/test_dispatch_outcome.py -v
~~~

