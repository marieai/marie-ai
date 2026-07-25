# Slice 03: Propagate Pre-RPC Gateway Failures

**Status:** Proposed

**Depends on:** [Slice 02](slice-02-dispatch-result-and-handle.md)

**Parent task:** [Task 3](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-3-move-the-lifecycle-boundary-and-propagate-every-pre-rpc-failure)

## Scope

Wire `DispatchHandle` and `SendFailure` through JobManager, JobSupervisor, GatewayJobDistributor, and networking. This slice ends at the scheduler-facing admission result; scheduler outcome policy lands in Slice 04.

## Objective

Make every known failure before RPC creation reject the same logical dispatch immediately with its real cause, while preserving post-RPC ambiguity.

## Touchpoints

- `marie/job/job_manager.py`
- `marie/job/job_supervisor.py`
- `marie/job/job_distributor.py`
- `marie/job/gateway_job_distributor.py`
- `marie/serve/networking/__init__.py`
- focused job, networking, and supervisor tests

## Work

- Replace `confirmation_event` parameters with one `DispatchHandle`.
- Attach the reversible supervisor task without marking detach.
- Mark `send_detached` synchronously immediately after `send_nowait` returns.
- Make desired-state admission strict: write state, require an epoch, then admit.
- Prevent RPC creation when pre-send raises or exceeds its callback budget.
- Replace the callback tuple with a typed logical-send callback value.
- Invoke one final structured failure callback for no replicas, connection acquisition failure, topology failure, strict pre-send rejection, RPC failure, and retry exhaustion.
- Emit intermediate retry traces without completing the logical admission more than once.
- Run desired-state admission and acknowledgement tracking once per logical dispatch, not per retry.
- Replace misleading response traces with stage-accurate task, payload, topology, replica, connection, RPC-start, and final-send events.

## Non-goals

- Do not decide scheduler failure versus unknown policy here.
- Do not add dynamic gateway configuration repair; that is Slice 09.
- Do not add executor application-error handling.

## Acceptance criteria

- Cancellation before detach prevents send creation.
- Cancellation after detach is never reported as proof that the send stopped.
- No-replica and connection failures reject before the confirmation budget.
- Strict pre-send rejection creates no RPC task.
- A crash after RPC start remains ambiguous.
- Retries reuse one desired-state admission and finish with an explicit result, never implicit `None`.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/job/ \
  tests/unit/serve/networking/test_pre_send_callback.py \
  tests/unit/serve/networking/test_send_failure_callback.py \
  tests/integration/job/test_job_supervisor.py -q
~~~
