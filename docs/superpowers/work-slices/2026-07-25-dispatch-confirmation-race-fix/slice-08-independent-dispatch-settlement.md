# Slice 08: Settle Dispatch Results Independently

**Status:** Proposed

**Depends on:** [Slice 04](slice-04-structured-scheduler-outcomes.md), [Slice 06](slice-06-dispatch-unknown-settlement.md)

**Parent task:** [Task 8](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-8-settle-each-dispatch-independently-and-make-terminal-logging-honest)

## Scope

Remove the batch-wide settlement barrier and align logs and counters with the durable state that won each race.

## Objective

Settle each dispatch as soon as its own wait completes so one slow gateway result cannot delay safe cleanup or unknown reconciliation for unrelated jobs.

## Touchpoints

- `marie/scheduler/psql.py`
- `marie/scheduler/services/attempt_lifecycle_service.py`
- scheduler dispatch-cycle and race tests

## Work

- Wrap each dispatch in `_await_and_settle(item)` and gather the wrappers rather than gathering raw enqueue tasks first.
- Report separate confirmed, unknown, and failed counts.
- Count unknown as occupied capacity, but not confirmed scheduled work.
- Treat rejected cleanup against an already terminal durable state as an informational race result.
- Keep a rejected transition against nonterminal state at error level.
- Use any follow-up durable-state read for diagnostics only.
- Preserve the ownership contract from Slice 06: terminal state does not by itself authorize both scheduler and worker to release capacity.

## Non-goals

- Do not redesign scheduler batching or fairness.
- Do not make unknown equivalent to success or failure.
- Do not use log-level changes to hide a state transition that genuinely failed.

## Acceptance criteria

- A fast dispatch settles before a slow sibling finishes.
- Tests use distinct job identities and prove actual ordering.
- Unknown outcomes update only the unknown count.
- Already-terminal cleanup rejection logs at `INFO`; nonterminal rejection remains `ERROR`.
- Immediate slot release occurs only for proven safe failure or an explicitly scheduler-owned terminal settlement.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_cycle.py \
  tests/unit/scheduler/test_dispatch_confirmation_race.py -q
~~~

