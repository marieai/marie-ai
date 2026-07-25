# Slice 05: Make Dispatch Audit Transitions Monotonic

**Status:** Proposed

**Depends on:** [Slice 04](slice-04-structured-scheduler-outcomes.md)

**Parent task:** [Task 5](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-5-make-dispatch-audit-transitions-monotonic)

## Scope

Constrain repository writes so a late dispatch result cannot overwrite a terminal, recovered, or more advanced dispatch state.

## Objective

Make PostgreSQL the authority for which lifecycle transition won and report rejected stale transitions instead of silently claiming success.

## Touchpoints

- `marie/scheduler/repository/async_job_repository.py`
- `marie/scheduler/repository/job_repository.py`
- repository unit tests
- `tests/integration/scheduler/test_job_attempt_audit.py`

## Work

- Guard dispatch-start upserts against `terminal_accepted` and `recovery_at`.
- Make dispatch-result recording outcome-aware and return whether the update applied.
- Use explicit predicates for admitted, unknown, and proven pre-RPC rejection.
- Permit only the transition table in Task 5 of the parent plan.
- Emit `job_attempt_audit_rejected` with the requested transition and attempt identity when PostgreSQL rejects a stale update.
- Keep terminal and recovery writes authoritative.

## Non-goals

- Do not add a schema migration for `dispatch_unknown`; `attempt_state` is unconstrained text.
- Do not settle capacity in repository methods.
- Do not use SQL string matching as the only proof of behavior.

## Acceptance criteria

- `dispatching -> dispatch_unknown -> dispatched` is valid.
- `dispatched -> dispatch_unknown` and `dispatched -> dispatch_failed` are rejected.
- Terminal and recovered rows cannot return to a dispatch state.
- A stale `run_attempt_id` cannot update the current attempt.
- At least one real PostgreSQL test proves the terminal or recovery race.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/repository/test_job_repository.py \
  tests/integration/scheduler/test_job_attempt_audit.py -q
~~~

Do not mark this slice complete if the required PostgreSQL test is skipped.

