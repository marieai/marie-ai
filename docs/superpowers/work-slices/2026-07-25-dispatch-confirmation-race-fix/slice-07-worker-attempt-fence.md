# Slice 07: Fence Worker Attempt Adoption

**Status:** Proposed

**Depends on:** [Slice 05](slice-05-monotonic-dispatch-audit.md), [Slice 06](slice-06-dispatch-unknown-settlement.md)

**Parent task:** [Task 7](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-7-fence-worker-start-and-make-capacity-adoption-mandatory)

## Scope

Add the narrow worker-side fence required to make late gateway delivery safe. This is dispatch correctness, not executor health management.

## Objective

Prevent a stale or capacity-unowned scheduler request from invoking executor code after PostgreSQL recovered or replaced its attempt.

## Touchpoints

- scheduler job repositories
- `marie/serve/runtimes/worker/request_handling.py`
- worker semaphore unit tests
- `tests/integration/scheduler/test_run_attempt_worker_fence.py`

## Work

- Add one atomic repository operation that validates job id, active state, run owner, run attempt, and unexpired run lease while extending the lease.
- Make `_sem_track` return whether capacity adoption succeeded.
- For scheduler-managed requests, enforce this order: adopt capacity, fence the attempt in PostgreSQL, record `RUNNING`, then invoke executor code.
- Reject without executor invocation when adoption fails.
- Release newly adopted capacity and reject when the PostgreSQL fence fails.
- Preserve current behavior for direct requests that carry no scheduler attempt metadata.
- Emit accepted-fence, stale-attempt rejection, and capacity-adoption failure events with attempt correlation.

## Non-goals

- Do not classify application or model failures.
- Do not add worker restart or health policy.
- Do not use lagging `JobInfo` as the authoritative attempt fence.

## Acceptance criteria

- Only the matching active, unexpired attempt reaches executor invocation.
- Expired, recovered, and replaced attempts are rejected before executor code.
- Failed adoption invokes no executor code.
- Failed fencing releases the newly adopted ticket once.
- Successful adoption provides the explicit ownership evidence used by Slice 06.
- Direct non-scheduler requests remain unchanged.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py \
  tests/integration/scheduler/test_run_attempt_worker_fence.py -q
~~~

Do not mark this slice complete if the PostgreSQL fencing test is skipped.
