# Slice 01: Freeze the Dispatch-Race Contract

**Status:** Complete

**Depends on:** Existing `reproduce-dispatch-race` target

**Parent task:** [Task 1](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-1-freeze-the-current-reproduction-and-define-invariants)

## Scope

Tests and documentation only. Preserve the deterministic reproduction of the current bug and define the corrected lifecycle cases that later slices must satisfy.

## Objective

Keep one fast test that proves the existing `dispatch_error='False'` race and add named fixtures and failing contract tests for pre-detach failure, post-detach unknown delivery, safe pre-RPC rejection, no replicas, post-RPC ambiguity, and recovery fencing.

## Touchpoints

- `tests/unit/scheduler/test_dispatch_confirmation_race.py`
- `stress-test.sh`
- `tools/stress/scheduler-reliability.md`

## Work

- Preserve `./stress-test.sh reproduce-dispatch-race` and its current buggy assertions.
- Explain in the test why the literal `False` payload, timeout, cleanup error, and late success are one race signature.
- Add a state-machine fixture with distinct `job_id` and `run_attempt_id` values.
- Add the corrected-behavior test names from Task 1 without weakening the existing reproducer.
- Mark expected pre-fix failures explicitly so the baseline remains readable.

## Non-goals

- Do not change scheduler, gateway, networking, or worker production code.
- Do not make the reproducer depend on PostgreSQL or a live gateway.
- Do not add executor-health cases.

## Acceptance criteria

- The existing race reproduces deterministically.
- Every lifecycle case has a distinct test name and attempt identity.
- Test comments distinguish observed buggy behavior from the target contract.
- Later slices can flip assertions without replacing the fixture.

## Verification

~~~bash
.venv/bin/python -m pytest tests/unit/scheduler/test_dispatch_confirmation_race.py -v
./stress-test.sh reproduce-dispatch-race
~~~

Expected at this slice: the legacy reproducer passes; corrected-behavior cases are explicitly marked as not implemented or fail in the documented way.

## Implementation result

- Preserved the existing `reproduce-dispatch-race` target without modifying `stress-test.sh`.
- Added a reusable seven-case lifecycle fixture with unique job and run-attempt identifiers.
- Kept the current bug signature as one passing regression test.
- Added seven strict expected-failure tests for the corrected lifecycle contract.
- Documented the baseline in `tools/stress/scheduler-reliability.md`.

Verification on 2026-07-25:

~~~text
pytest: 1 passed, 7 xfailed
reproduce-dispatch-race: 1 passed
ruff: all checks passed
~~~
