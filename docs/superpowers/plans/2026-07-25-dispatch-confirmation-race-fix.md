# Dispatch Confirmation Race Fix Implementation Plan

**Status:** In progress. Slice 01 is complete; review the remaining slice acceptance criteria before implementation.

**Goal:** Eliminate misleading dispatch failures, propagate known gateway failures without waiting for a timeout, prevent late dispatch from racing retry or recovery, preserve executor capacity while dispatch ownership is uncertain, and prove the gateway boundaries with deterministic mock-executor tests.

**Primary regression:** The scheduler currently turns a boolean return value into the error string "False", times out while detached dispatch work continues, then tries to fail a job that may already have completed.

**Target repository:** /home/greg/dev/marieai/marie-ai

**Runtime mock configuration:** /mnt/data/marie-ai/config/service/mock/marie-mock-scheduler-test.yml

## Required outcome

The implementation is complete only when all of these statements are true:

1. Dispatch failure metadata never contains dispatch_error="False" or error_message="False".
2. A timeout before detached send creation cancels submission, records dispatch_timeout, fails or retries the owned attempt, and releases its slot.
3. A timeout after detached send creation never immediately fails the job or releases its slot.
4. A later pre-RPC rejection resolves an unknown dispatch as a safe failure with its real reason.
5. A later admission resolves an unknown dispatch as dispatched and keeps the slot until the worker adopts it.
6. A stale run attempt cannot begin executor work after PostgreSQL has recovered or replaced that attempt.
7. A worker cannot continue when it fails to adopt or reserve executor capacity.
8. Dispatch audit state moves monotonically and cannot overwrite terminal or recovered state.
9. Each dispatch result is settled as soon as that job finishes waiting; one slow job does not delay cleanup for the rest of the batch.
10. A real gateway run using mock_parallel_subgraphs deterministically emits gateway_dispatch_timeout and then proves exactly one valid outcome for every affected attempt.
11. A pre-send callback failure or timeout prevents RPC creation and reaches the scheduler with its real cause.
12. A missing replica or connection-acquisition failure reaches the scheduler as a named pre-RPC rejection before the dispatch-confirmation budget expires.
13. dispatch_confirmation_timeout_seconds is independent of lease_ttl_seconds and is validated against the pre-send, semaphore, and run-lease budgets.
14. A dynamic gateway topology rebuild preserves configured prefetch, RPC timeout, retry policy, and circuit-breaker behavior.
15. Admission and worker-ack tracking run once per logical dispatch, not once per transport retry.
16. Ordinary executor application failures remain distinguishable from gateway transport failures.
17. The gateway test suite separately proves delayed-admission reconciliation and no-replica fast rejection.
18. An accepted terminal event stops reconciliation, and exactly one recorded capacity owner releases the ticket.

## Verified current defects

These are code-confirmed starting conditions, not hypotheses:

1. lease_ttl_seconds defaults to 5 and enqueue derives a 4-second confirmation timeout from lease_ttl_seconds - 1.
2. The mock and service configurations do not override lease_ttl_seconds.
3. No replicas causes GrpcConnectionPool.send_requests_once to return None; the topology raises, the detached task logs the crash, and nothing completes the scheduler confirmation waiter.
4. on_failure_cb is unpacked by networking but never invoked.
5. _safe_send_callback suppresses pre-send exceptions and callback timeouts, allowing RPC creation after admission setup failed.
6. The current pre-send callback signals confirmation before the desired-state write, so confirmed currently means callback entry rather than durable admission.
7. after_send_callback runs concurrently with RPC creation, receives response=None, and starts a blocking worker-ack poll in the default thread pool.
8. The dynamic Marie gateway streamer rebuild omits configured prefetch, timeout_send, retries, and circuit-breaker settings. The effective fallback is unbounded prefetch, no RPC timeout, zero retries, and no circuit breaker.
9. INTERNAL is listed as handled but not retryable; the current retry branch resets the channel and can fall through with None instead of returning an explicit failure.
10. If a circuit breaker is enabled, the current all-open fallback routes through the same unhealthy connections.

The direct dispatch-timeout mechanism is loss of a pre-RPC failure signal. Executor process health is a separate concern and is intentionally outside this plan.

## Non-goals

- Do not promise exactly-once execution for arbitrary external side effects. Enforce attempt fencing before the Marie executor begins work and preserve existing idempotency expectations.
- Do not redesign the scheduler or job queue beyond the dispatch and gateway boundaries required by this failure mode.
- Do not add a database column or enum for dispatch_unknown. job_attempt.attempt_state is unconstrained TEXT.
- Do not use mock executor processing delay as the dispatch-timeout trigger. Executor processing begins after dispatch admission and therefore cannot deterministically delay the pre-send confirmation.
- Do not add executor health or worker restart behavior. Track that work in a separate executor-reliability plan.
- Do not create commits automatically. Commit only when explicitly requested, and stage exact files rather than directories.

## Correct lifecycle model

The old plan used supervisor task creation as the handoff boundary. That is too early: the supervisor task can exist without having created detached send work.

Use the following states:

| State | Meaning | Timeout treatment |
|---|---|---|
| preparing | JobManager is writing JobInfo or preparing the supervisor | Cancel submission; safe failure |
| supervisor_started | The reversible supervisor task exists, but no detached send task exists | Cancel supervisor; safe failure |
| send_detached | GatewayJobDistributor created the independent send task | Dispatch is unknown; do not fail or release |
| admitted | Desired-state write succeeded | Record dispatched |
| rejected_pre_rpc | Pre-send setup failed before networking created the RPC | Safe failure with the real reason |
| rpc_started | Networking created the RPC task | Delivery is ambiguous; never report a safe dispatch failure from cancellation alone |
| worker_running | Worker adopted capacity and atomically fenced the active attempt | Worker owns renewal and terminal release |
| terminal | Attempt completed, failed, or stopped | AttemptLifecycleService settles durable state; the current capacity owner releases the ticket |
| recovered | Run lease recovery replaced or failed the attempt | Late worker admission must be rejected |

Important terminology:

- supervisor_started is not a handoff.
- send_detached is the ambiguity boundary. It means cancellation of the outer submit coroutine no longer reaches the independent send task.
- admitted is not executor receipt. It means durable pre-send setup succeeded.
- rpc_started is the last point at which a local send failure can no longer prove that executor code did not run.
- after_send_callback is currently invoked concurrently with RPC creation and receives response=None. It must not signal admission or be described as a response acknowledgment.
- the current networking helper swallows pre-send exceptions and callback timeouts. The corrected pre-send path must be strict: failure prevents RPC creation and propagates to DispatchHandle.
- executor application failures are terminal worker concerns; this plan changes only how the gateway classifies transport and pre-RPC failures.

## Core design

### DispatchHandle

Create one handle per scheduler dispatch attempt. It coordinates the scheduler, JobManager, JobSupervisor, and the unknown-dispatch reconciler.

Required fields:

- supervisor_task: outer JobSupervisor task, if created
- send_task: detached GatewayJobDistributor task, if created
- send_detached: asyncio.Event
- rpc_started: asyncio.Event
- admission: asyncio.Future carrying a DispatchAdmission result
- owning event loop

Required operations:

- attach_supervisor(task): record reversible supervisor work without setting send_detached
- mark_send_detached(task): synchronously record the independent send task and set send_detached
- mark_rpc_started(context): synchronously record the address, deployment, and transport attempt when networking creates the RPC task
- admit(epoch): complete admission once, with the desired-state epoch
- reject(reason, stage, safe_before_rpc): complete admission once with a structured failure; safe_before_rpc may be true only while rpc_started is false
- cancel_if_not_detached(): atomically cancel the supervisor only when send_detached is still false
- cancel_send_best_effort(): used only after the attempt is durably fenced or recovered; never treated as proof that the remote request did not execute

Use asyncio.wait_for(asyncio.shield(handle.admission), timeout). The timeout must not cancel the admission future because reconciliation continues waiting for it.

### DispatchOutcome

Replace the bool result with a truthy/falsy-compatible value that cannot lose the reason or lifecycle stage.

Required fields:

- status: confirmed, failed, or unknown
- reason: normalized string or None
- stage: pre_detach, post_detach_pre_rpc, post_rpc, admitted, or pre_rpc_rejected
- handle: DispatchHandle when reconciliation may still be required

Required behavior:

- bool(outcome) is true only for confirmed
- outcome.dispatch_unknown is true only for status=unknown
- exceptions use repr(error), not str(bool)
- dispatch_timeout is a named constant

Do not infer unknown solely from whether a supervisor task exists.

### Structured send failure

Replace the untyped callback tuple and log-only failure path with a structured SendFailure value.

Required fields:

- reason: stable machine-readable reason such as no_available_replicas, connection_acquire_failed, pre_send_timeout, pre_send_rejected, rpc_failed, or send_task_crashed
- stage: pre_detach, post_detach_pre_rpc, or post_rpc
- error: original exception repr where available
- deployment and address
- retry_index and retry_count
- rpc_started

Completion rules:

- no replicas, connection acquisition failure, topology failure before RPC creation, strict pre-send failure, and a detached-task crash before RPC creation reject admission immediately
- once rpc_started is true, a send failure is not a safe dispatch failure; retain or reconcile the attempt as unknown unless a terminal worker/job event settles it
- on_failure is invoked exactly once per logical dispatch result and is never a log-only callback
- intermediate retry failures emit transport-attempt traces but do not complete admission or invoke the final on_failure callback
- a transport retry must not rerun logical admission or create another desired-state epoch
- ordinary executor application errors remain response/terminal failures, not connection failures

### Independent dispatch budgets

Add explicit configuration rather than deriving dispatch confirmation from the soft candidate lease:

- dispatch_confirmation_timeout_seconds
- pre_send_callback_timeout_seconds
- dispatch_unknown_grace_seconds
- gateway_rpc_timeout_seconds

The job has already been promoted to an ACTIVE run attempt before enqueue, with run_ttl_seconds protection, and the scheduler semaphore has its own TTL. Validate the production relationship:

~~~text
pre_send_callback_timeout_seconds
  < dispatch_confirmation_timeout_seconds
  < semaphore_ticket_ttl_seconds
  < run_ttl_seconds
~~~

Use initial production defaults of 10 seconds for strict pre-send and 15 seconds for dispatch confirmation, subject to trace-based qualification. Keep the current 30-second semaphore TTL and 60-second run TTL unless measurement requires a separate change. If configuration permits a confirmation or unknown grace window to approach either ownership TTL, renew the run lease and semaphore while waiting rather than relying on startup validation alone.

The gated deterministic timeout test may intentionally make pre_send_callback_timeout_seconds greater than dispatch_confirmation_timeout_seconds so it can create a post-detach unknown. Production configuration must not use that inversion.

### Unknown-dispatch reconciliation

Post-detach timeout is not a terminal result. Start a bounded reconciliation task for that attempt.

While the authoritative PostgreSQL attempt remains ACTIVE, reconciliation must:

1. Keep the scheduler semaphore ticket alive.
2. Keep the run lease alive during a bounded dispatch-unknown grace period.
3. Await the shielded admission result.
4. Observe JobInfo for RUNNING or terminal state.
5. Stop scheduler renewal only after the worker has adopted the ticket, the attempt is terminal, or recovery has fenced it.

Resolution rules:

- admission accepted:
  - conditionally promote job_attempt from dispatch_unknown to dispatched
  - emit dispatch_unknown_resolved with resolution=admitted
  - continue capacity and run-lease renewal until RUNNING or terminal
- admission rejected before RPC:
  - emit dispatch_unknown_resolved with resolution=rejected_pre_rpc
  - perform normal fenced fail or retry with the real reason
  - release the scheduler ticket
- JobInfo RUNNING:
  - worker must already have adopted the same ticket
  - stop scheduler-side semaphore renewal
- accepted terminal event:
  - stop unknown-dispatch reconciliation and do not write another dispatch failure
  - let AttemptLifecycleService settle the durable job, attempt, frontier, and DAG state
  - if worker adoption was observed, let worker terminal cleanup release the ticket
  - if worker adoption was not observed, release the still scheduler-owned ticket
- attempt no longer ACTIVE with the same owner and attempt id:
  - best-effort cancel local send work
  - release any scheduler-owned ticket
  - emit dispatch_unknown_resolved with resolution=recovered_or_replaced
- grace deadline reached:
  - record dispatch_stalled diagnostics
  - stop extending the run lease so normal recovery can fence the attempt
  - continue observing until recovery or a valid worker claim occurs

Track reconciliation tasks in the scheduler runtime and drain or cancel them during shutdown. Gateway process death still falls back to run-lease recovery and semaphore TTL; TTL is a crash fallback, not successful attempt settlement.

### Worker-side attempt fence

Terminal-event fencing is too late to prevent duplicate executor side effects. Fence before invoking the executor.

For scheduler-managed requests carrying run_owner and run_attempt_id:

1. Attempt to adopt or reserve the existing semaphore ticket.
2. If capacity adoption fails, reject the request; do not invoke executor code.
3. Atomically validate and extend the PostgreSQL run lease:
   - job id matches
   - state is active
   - run_owner matches
   - run_attempt_id matches
   - run lease has not expired
4. If the update returns zero rows:
   - release the adopted ticket
   - emit executor_stale_attempt_rejected
   - return an infrastructure failure without invoking executor code
5. Only then record RUNNING and call the executor.

The atomic validation must query the scheduler job table, not only JobInfo storage. JobInfo can lag run-lease recovery.

Change worker semaphore adoption to return bool. The current log-and-continue behavior on failed adoption is not acceptable for scheduler-managed work.

Direct non-scheduler requests without run attempt metadata retain their existing behavior.

### Gateway runtime and backpressure

The dynamically rebuilt Marie gateway streamer must preserve the same runtime controls as the configured gateway. A topology rebuild must not silently fall back to GatewayStreamer defaults.

Required controls:

- prefetch: retain the configured bound; zero remains an explicit opt-out, not an accidental default
- timeout_send: require a finite executor RPC deadline for scheduler dispatch
- retries: use an explicit policy and test every retryable status
- circuit_breaker_config: retain configured circuit state across incremental address updates where possible
- load_balancer_type and gRPC channel options

Behavioral rules:

- if every circuit is open, fail fast with all_replicas_unhealthy; never fall back to routing through known-open circuits
- INTERNAL is non-retryable unless an explicit policy says otherwise; it must never fall through the retry loop as an implicit None result
- retry exhaustion returns a structured failure rather than None
- application-level executor failure does not automatically increment the transport circuit breaker
- RequestStreamer prefetch and the scheduler semaphore are complementary: prefetch bounds gateway work creation, while the semaphore bounds executor capacity

### Layer-correct deterministic fault hooks

Continue using the existing mock executors and mock_parallel_subgraphs plan. Add a narrowly gated scheduler test hook because mock processing time and failure injection occur after dispatch admission.

Use a test-only budget profile in the mock service file:

~~~yaml
job_scheduler_kwargs:
  dispatch_confirmation_timeout_seconds: 4
  pre_send_callback_timeout_seconds: 8
  dispatch_unknown_grace_seconds: 12
  test_hooks:
    allow_dispatch_admission_delay: true
    max_dispatch_admission_delay_seconds: 10
    allow_no_replica_rejection: true
~~~

Production configurations omit this block, so request metadata cannot activate the hook.

Add stress-tool options:

- --dispatch-admission-delay-seconds
- --dispatch-admission-delay-executor
- --gateway-failure-mode no_replicas
- --gateway-failure-executor

The stresser writes the requested values into scheduler test metadata together with run_id. PostgreSQLJobScheduler validates each hook against configuration and matches the job entrypoint. JobSupervisor applies the asynchronous delay after mark_send_detached and before the desired-state write. The gateway no-replica hook rejects before connection acquisition without invoking an executor.

The injected delay must satisfy:

- dispatch confirmation timeout < injected delay
- injected delay < strict pre-send callback timeout
- injected delay < dispatch_unknown_grace_seconds

For the concrete profile below, use 4-second confirmation, 6-second injected delay, 8-second strict pre-send callback, and 12-second unknown grace. Validate the actual configured values at startup instead of assuming defaults.

Emit:

- dispatch_admission_fault_injected
- gateway_dispatch_timeout with stage=post_detach_pre_rpc
- dispatch_unknown_retained
- dispatch_unknown_resolved
- gateway_dispatch_rejected with reason=no_available_replicas and stage=post_detach_pre_rpc

Never read unrestricted fault controls directly from user metadata in production code. Every fault must be gated by mock scheduler configuration and associated with a unique stress_run_id.

---

## Implementation work slices

Execute and review this plan through the work-slice set in [dispatch-confirmation-race-fix](../work-slices/2026-07-25-dispatch-confirmation-race-fix/README.md). Each numbered task below has one matching slice with explicit dependencies, acceptance criteria, and verification commands. The plan remains the design source of truth; slices define delivery and review boundaries.

| Task | Work slice |
|---|---|
| 1 | [Freeze the dispatch-race contract](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-01-freeze-dispatch-race-contract.md) |
| 2 | [Add dispatch result and handle types](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-02-dispatch-result-and-handle.md) |
| 3 | [Propagate pre-RPC gateway failures](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-03-pre-rpc-failure-propagation.md) |
| 4 | [Return structured scheduler outcomes](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-04-structured-scheduler-outcomes.md) |
| 5 | [Make dispatch audit transitions monotonic](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-05-monotonic-dispatch-audit.md) |
| 6 | [Settle unknown dispatches](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-06-dispatch-unknown-settlement.md) |
| 7 | [Fence worker attempt adoption](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-07-worker-attempt-fence.md) |
| 8 | [Settle dispatch results independently](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-08-independent-dispatch-settlement.md) |
| 9 | [Preserve gateway transport controls](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-09-gateway-transport-controls.md) |
| 10 | [Qualify gateway faults end to end](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-10-gateway-e2e-fault-qualification.md) |
| 11 | [Run verification and update operator docs](../work-slices/2026-07-25-dispatch-confirmation-race-fix/slice-11-verification-and-operator-docs.md) |

---

## Task 1: Freeze the current reproduction and define invariants

**Files**

- tests/unit/scheduler/test_dispatch_confirmation_race.py
- stress-test.sh
- tools/stress/scheduler-reliability.md

### Steps

- [x] Keep the existing deterministic regression target:

~~~bash
./stress-test.sh reproduce-dispatch-race
~~~

- [x] Record the existing buggy assertions in test comments:
  - literal dispatch_error="False"
  - timeout log
  - dispatch failure cleanup log
  - late success

- [x] Add a short state-machine fixture used by later tests. Give every fixture job a distinct job id and run_attempt_id.

- [x] Add explicit test names for the final contract:
  - test_timeout_before_detached_send_is_safe_failure
  - test_timeout_after_detached_send_is_unknown
  - test_unknown_later_admitted_becomes_dispatched
  - test_unknown_later_rejected_before_rpc_becomes_failure
  - test_no_replicas_rejects_before_confirmation_timeout
  - test_send_crash_after_rpc_start_remains_unknown
  - test_recovered_attempt_rejects_late_worker_start

### Verification

~~~bash
.venv/bin/python -m pytest tests/unit/scheduler/test_dispatch_confirmation_race.py -v
./stress-test.sh reproduce-dispatch-race
~~~

Expected before implementation: the existing bug reproducer passes; new corrected-behavior tests fail.

---

## Task 2: Add DispatchHandle, SendFailure, and DispatchOutcome

**Files**

- create marie/job/dispatch_handle.py
- create marie/job/send_failure.py
- create marie/scheduler/dispatch_outcome.py
- create tests/unit/job/test_dispatch_handle.py
- create tests/unit/job/test_send_failure.py
- create tests/unit/scheduler/test_dispatch_outcome.py

### Steps

- [ ] Implement DispatchAdmission as a frozen result:
  - accepted: bool
  - reason: str or None
  - desired_epoch: int or None
  - safe_before_rpc: bool
  - stage: str

- [ ] Implement DispatchHandle with the operations defined above.

- [ ] Implement SendFailure with the structured fields and stable reason constants defined above.

- [ ] Make completion idempotent. A late reject cannot replace an accepted admission and a late accept cannot replace a rejection.

- [ ] Implement atomic cancel_if_not_detached with no await between checking send_detached and cancelling supervisor_task.

- [ ] Implement DispatchOutcome with confirmed, failed, and unknown constructors and truthiness compatibility.

### Required tests

- [ ] Supervisor attachment does not set send_detached.
- [ ] mark_send_detached stores the independent task and sets the event.
- [ ] mark_rpc_started permanently prevents a later failure from claiming safe_before_rpc.
- [ ] cancel_if_not_detached cancels a pending supervisor.
- [ ] cancel_if_not_detached refuses after mark_send_detached.
- [ ] admission remains usable after asyncio.wait_for times out around asyncio.shield.
- [ ] admission accepts only its first terminal result.
- [ ] DispatchOutcome never stringifies False as a reason.
- [ ] SendFailure preserves the original exception repr without using a boolean or None as its reason.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/job/test_dispatch_handle.py \
  tests/unit/job/test_send_failure.py \
  tests/unit/scheduler/test_dispatch_outcome.py -v
~~~

---

## Task 3: Move the lifecycle boundary and propagate every pre-RPC failure

**Files**

- marie/job/job_manager.py
- marie/job/job_supervisor.py
- marie/job/job_distributor.py
- marie/job/gateway_job_distributor.py
- marie/serve/networking/__init__.py
- tests/integration/job/test_job_supervisor.py
- tests/unit/serve/networking/test_pre_send_callback.py
- tests/unit/serve/networking/test_send_failure_callback.py
- tests/unit/job/test_job_manager_dispatch.py

### Steps

- [ ] Replace confirmation_event parameters with DispatchHandle.

- [ ] In JobManager.submit_job:
  - attach the supervisor task after create_task
  - do not mark send_detached
  - re-raise setup failures after updating JobInfo
  - reject the handle with the actual setup error
  - check task.cancelled() before task.exception() in the done callback

- [ ] In JobSupervisor._submit_job_in_background:
  - call send_nowait
  - immediately call handle.mark_send_detached(send_task), with no await between the two operations
  - preserve the send task for reconciliation
  - attach a completion callback that classifies a crash using handle.rpc_started
  - reject a pre-RPC task crash and retain a post-RPC crash as unknown

- [ ] In pre_send_callback:
  - apply only the already validated test delay, when present
  - write desired state
  - reject if no desired epoch is returned
  - call handle.admit(epoch)
  - on error, call handle.reject(reason, stage="post_detach_pre_rpc", safe_before_rpc=True), preserving repr(error), then re-raise

- [ ] Replace the untyped three-callback tuple with a small SendCallbacks value carrying:
  - pre_send
  - rpc_started
  - on_failure
  - pre_send_timeout_seconds

- [ ] Invoke on_failure exactly once for:
  - no replicas returned by send_requests_once
  - connection acquisition failure
  - topology failure before RPC creation
  - strict pre-send exception or timeout
  - RPC failure after creation
  - retry exhaustion

- [ ] Pass a structured SendFailure to on_failure. Do not use None, False, or a generic Exception string as the reason.

- [ ] Emit a transport-attempt failure event for retryable intermediate failures; invoke on_failure only after the logical send is finally rejected or exhausted.

- [ ] Make networking run pre_send in strict mode:
  - pre-send exception propagates
  - pre-send timeout propagates as an explicit callback-timeout error
  - networking does not create rpc_task after either failure
  - the failure callback may observe the error but may not convert it to success

- [ ] Keep error-suppressing callback behavior only for genuinely observational callbacks such as rpc_started.

- [ ] Run logical admission once per dispatch:
  - desired-state write and handle.admit execute before the first RPC
  - transport retries reuse the admitted epoch
  - retries invoke rpc_started with retry_index but do not invoke pre_send again

- [ ] Remove admission signaling from after_send_callback.

- [ ] Correct the misleading trace:
  - do not emit job_supervisor_response_received when response is None
  - use gateway_rpc_started for the callback that runs at RPC task creation
  - keep actual response completion under job_supervisor_send_task_completed
  - add job_supervisor_task_started and job_supervisor_job_info_loaded
  - add gateway_payload_build_started and gateway_payload_build_completed
  - add gateway_topology_dispatch_entered
  - add gateway_replica_lookup with replica_count
  - add gateway_connection_acquire_started and gateway_connection_acquire_completed
  - add gateway_send_failed with reason, stage, rpc_started, and retry_index

### Required tests

- [ ] Cancelling between supervisor creation and send_nowait prevents send creation.
- [ ] Once send_nowait returns, cancelling the supervisor does not prove the send was cancelled.
- [ ] Desired-state failure rejects admission with its actual exception.
- [ ] Networking creates no RPC when strict pre-send raises.
- [ ] Networking creates no RPC when strict pre-send exceeds its callback budget.
- [ ] Zero replicas invokes one pre-RPC failure callback with no_available_replicas.
- [ ] Connection acquisition failure invokes one pre-RPC failure callback.
- [ ] A pre-RPC detached-task crash rejects admission immediately.
- [ ] A post-RPC detached-task crash cannot claim safe_before_rpc.
- [ ] Retry exhaustion returns a structured failure rather than None.
- [ ] INTERNAL with retries disabled returns a non-retryable structured failure.
- [ ] Logical admission and desired-state write occur once across transport retries.
- [ ] The strict pre-send callback budget can exceed the scheduler confirmation timeout for the deterministic test profile.
- [ ] A missing desired epoch is rejected.
- [ ] after_send_callback cannot complete admission.
- [ ] Cancelled supervisor tasks do not produce event-loop callback errors.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/job/ \
  tests/unit/serve/networking/test_pre_send_callback.py \
  tests/unit/serve/networking/test_send_failure_callback.py \
  tests/integration/job/test_job_supervisor.py -q
~~~

---

## Task 4: Return structured scheduler outcomes and preserve real errors

**Files**

- marie/scheduler/psql.py
- tests/unit/scheduler/test_dispatch_confirmation_race.py
- tests/unit/scheduler/test_dispatch_cycle.py

### Steps

- [ ] Change enqueue and _activate_and_enqueue_job to return DispatchOutcome.

- [ ] Add dispatch_confirmation_timeout_seconds to PostgreSQLSchedulerConfig and use it instead of lease_ttl_seconds - 1.

- [ ] Add pre_send_callback_timeout_seconds and validate all dispatch budgets at startup.

- [ ] Wait for handle.admission using asyncio.wait_for(asyncio.shield(...)).

- [ ] Handle outcomes as follows:
  - admitted before timeout: confirmed
  - rejected before RPC: failed with the rejection reason
  - timeout and cancel_if_not_detached succeeds: failed at pre_detach with dispatch_timeout
  - timeout after send_detached but before rpc_started: unknown at post_detach_pre_rpc with dispatch_timeout
  - timeout after rpc_started: unknown at post_rpc with dispatch_timeout
  - exception before detach: failed with repr(error)
  - structured safe rejection after detach but before RPC: failed at pre_rpc_rejected with its named reason
  - exception after detach without proof of pre-RPC rejection: unknown at the observed stage with repr(error)

- [ ] Do not write dispatch audit results from multiple exception branches. Classify the outcome once, then record it once.

- [ ] Pass outcome.reason into failure metadata. Never pass the outcome object or a boolean as the error argument.

- [ ] Log:
  - pre-detach timeout at ERROR
  - post-detach timeout at WARNING
  - actual rejection at ERROR with the real cause

- [ ] Include stage, run_owner, run_attempt_id, executor, and stress_run_id in trace fields.

### Required tests

- [ ] JobManager setup failure is reported immediately, not converted into dispatch_timeout.
- [ ] Desired-store rejection is reported immediately with its real exception.
- [ ] No-replica rejection is reported immediately as no_available_replicas rather than dispatch_timeout.
- [ ] Connection-acquisition rejection is reported immediately with its named reason.
- [ ] Pre-detach timeout cancels and releases.
- [ ] Post-detach timeout does not fail or release.
- [ ] Post-RPC timeout remains unknown even if local cancellation succeeds.
- [ ] Generic post-detach exception is audited unknown, not dispatch_failed.
- [ ] No failure metadata contains the strings "False" or "None".

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_confirmation_race.py \
  tests/unit/scheduler/test_dispatch_cycle.py -v
~~~

---

## Task 5: Make dispatch audit transitions monotonic

**Files**

- marie/scheduler/repository/async_job_repository.py
- marie/scheduler/repository/job_repository.py
- tests/unit/scheduler/repository/test_job_repository.py
- tests/integration/scheduler/test_job_attempt_audit.py

### Allowed transitions

| Operation | Allowed current states | New state |
|---|---|---|
| dispatch started | activated, dispatching | dispatching |
| admitted | dispatching, dispatch_unknown, dispatched | dispatched |
| timeout unknown | dispatching, dispatch_unknown | dispatch_unknown |
| proven pre-RPC rejection | dispatching, dispatch_unknown, dispatch_failed | dispatch_failed |
| terminal accepted | existing terminal guard | terminal state |
| recovered | active nonterminal attempt | recovered_retry or recovered_failed |

No operation may move dispatched back to dispatch_unknown or dispatch_failed.

### Steps

- [ ] Guard the ON CONFLICT update in record_job_attempt_dispatch_started. It must not overwrite terminal_accepted or recovery_at rows.

- [ ] Make record_job_attempt_dispatch_result outcome-aware and return bool indicating whether the update applied.

- [ ] Use separate SQL predicates for admitted, unknown, and rejected outcomes.

- [ ] Treat an unapplied audit update as an observable event:
  - emit job_attempt_audit_rejected
  - include requested transition and current attempt identity
  - never silently claim that the audit was written

- [ ] Keep terminal and recovery updates authoritative.

### Required tests

- [ ] dispatching to unknown applies.
- [ ] unknown to dispatched applies.
- [ ] dispatched to unknown is rejected.
- [ ] dispatched to failed is rejected.
- [ ] terminal accepted cannot become dispatching.
- [ ] recovered cannot become dispatching or dispatched.
- [ ] repeated identical updates are idempotent.

Use a real PostgreSQL integration test for at least the terminal/recovery race. String matching SQL is not sufficient.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/repository/test_job_repository.py \
  tests/integration/scheduler/test_job_attempt_audit.py -q
~~~

---

## Task 6: Reconcile post-detach unknown dispatches

**Files**

- marie/scheduler/psql.py
- marie/scheduler/runtime.py, if runtime task tracking is shared there
- tests/unit/scheduler/test_dispatch_unknown_reconciliation.py
- tests/unit/scheduler/test_dispatch_confirmation_race.py

### Steps

- [ ] Add a bounded _reconcile_dispatch_unknown coroutine.

- [ ] Start it only from _settle_dispatch_result when outcome.dispatch_unknown is true.

- [ ] Track it by run_attempt_id so only one reconciler exists per attempt.

- [ ] While unresolved:
  - renew the semaphore with the same executor, ticket id, and owner
  - extend the matching run lease with the existing fenced repository method
  - await admission and poll JobInfo without holding scheduler locks

- [ ] Add dispatch_unknown_grace_seconds to PostgreSQLSchedulerConfig. Validate that it is positive and greater than the deterministic mock delay used by the test profile.

- [ ] Validate the production timeout ordering against the semaphore ticket TTL and run TTL. Permit the test-only timeout inversion only when the gated admission-delay hook is enabled.

- [ ] Resolve according to the rules in Core design.

- [ ] On run-lease recovery, explicitly release the scheduler ticket instead of relying only on semaphore TTL.

- [ ] Ensure shutdown drains tracked reconciliation tasks and does not leak task exceptions.

### Required tests

- [ ] A late admission promotes unknown to dispatched.
- [ ] A late pre-RPC rejection invokes failure cleanup exactly once.
- [ ] Reconciliation renews the slot before its TTL.
- [ ] Reconciliation extends only the matching run attempt.
- [ ] RUNNING stops scheduler renewal only after worker adoption is represented.
- [ ] Recovery cancels local work best-effort and releases the ticket.
- [ ] Grace expiry permits recovery without directly failing the job.
- [ ] A terminal event after worker adoption stops reconciliation without scheduler-side ticket release; worker terminal cleanup releases it once.
- [ ] A terminal event before worker adoption settles durable state and releases the scheduler-owned ticket once.
- [ ] Concurrent terminal event and reconciliation produce one accepted terminal transition and one capacity release by the recorded owner.

Use short injected intervals or a controllable clock; do not make unit tests sleep for production TTLs.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_unknown_reconciliation.py \
  tests/unit/scheduler/test_dispatch_confirmation_race.py -q
~~~

---

## Task 7: Fence worker start and make capacity adoption mandatory

**Files**

- marie/scheduler/repository/async_job_repository.py
- marie/scheduler/repository/job_repository.py
- marie/serve/runtimes/worker/request_handling.py
- tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py
- tests/integration/scheduler/test_run_attempt_worker_fence.py

### Steps

- [ ] Add an atomic repository operation that validates and extends an ACTIVE run attempt:

~~~sql
UPDATE marie_scheduler.job
SET run_lease_expires_at = NOW() + requested_interval
WHERE id = requested_job_id
  AND state = 'active'
  AND run_owner = requested_run_owner
  AND run_attempt_id = requested_run_attempt_id
  AND run_lease_expires_at > NOW()
RETURNING id;
~~~

- [ ] Reuse the worker's configured PostgreSQL connection settings through a narrowly scoped run-attempt guard. Do not validate against JobInfo alone.

- [ ] Change _sem_track to return bool.

- [ ] For scheduler-managed work, execute this order:
  1. adopt capacity
  2. claim and extend the active run attempt
  3. record RUNNING
  4. invoke executor

- [ ] If capacity adoption fails, reject without executor invocation.

- [ ] If attempt fencing fails, release the adopted capacity and reject without executor invocation.

- [ ] Emit:
  - executor_capacity_adoption_failed
  - executor_stale_attempt_rejected
  - executor_attempt_fence_accepted

- [ ] Include job_id, run_owner, run_attempt_id, executor, and stress_run_id.

### Required tests

- [ ] Matching ACTIVE attempt is accepted and lease extended.
- [ ] Expired attempt is rejected.
- [ ] Recovered attempt is rejected.
- [ ] Replaced attempt id is rejected.
- [ ] Failed capacity adoption invokes no executor code.
- [ ] Failed attempt fence releases the newly adopted ticket.
- [ ] A late request from the recovered attempt cannot increment the mock executor invocation counter.
- [ ] Direct non-scheduler requests retain their current behavior.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py \
  tests/integration/scheduler/test_run_attempt_worker_fence.py -q
~~~

---

## Task 8: Settle each dispatch independently and make terminal logging honest

**Files**

- marie/scheduler/psql.py
- marie/scheduler/services/attempt_lifecycle_service.py
- tests/unit/scheduler/test_dispatch_cycle.py
- tests/unit/scheduler/test_dispatch_confirmation_race.py

### Steps

- [ ] Add _await_and_settle(item), which awaits one dispatch task and immediately calls _settle_dispatch_result.

- [ ] Gather these wrapper coroutines so a fast result settles before a slow dispatch returns.

- [ ] Report separate batch counts:
  - confirmed
  - unknown
  - failed

- [ ] Unknown occupies capacity for scheduling decisions but is not reported as confirmed scheduled work.

- [ ] Keep TerminalTransition as a typed truthy/falsy-compatible result if durable-state diagnostics are needed.

- [ ] Treat a rejected cleanup against an already terminal durable state as INFO.

- [ ] Treat the follow-up durable-state read as diagnostics only. Capacity release must be determined by the dispatch lifecycle, not by the later read.

- [ ] Keep the invariant explicit: only proven pre-detach or pre-RPC failures call immediate dispatch-failure cleanup.

### Required tests

- [ ] Fast dispatch settles before a slow dispatch completes.
- [ ] Tests use distinct job ids so ordering cannot pass ambiguously.
- [ ] Unknown increments the unknown count, not confirmed or failed.
- [ ] Already terminal cleanup rejection logs INFO.
- [ ] Nonterminal cleanup rejection remains ERROR.
- [ ] Slot release occurs only for safe failure outcomes.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_cycle.py \
  tests/unit/scheduler/test_dispatch_confirmation_race.py -q
~~~

---

## Task 9: Preserve gateway backpressure and make transport failure explicit

**Files**

- marie/serve/runtimes/servers/marie_gateway.py
- marie/serve/runtimes/gateway/streamer.py
- marie/serve/networking/__init__.py
- marie/job/job_supervisor.py
- marie/serve/networking/balancer/least_connection_balancer.py
- marie/serve/networking/balancer/round_robin_balancer.py
- config/service/mock/marie-mock-scheduler-test.yml
- tests/unit/serve/runtimes/gateway/test_dynamic_streamer_config.py
- tests/unit/serve/networking/test_retry_policy.py
- tests/unit/serve/networking/test_circuit_breaker_routing.py
- tests/unit/job/test_worker_ack_isolation.py

### Steps

- [ ] When MarieServerGateway recreates GatewayStreamer after discovery, explicitly pass:
  - timeout_send
  - retries
  - prefetch
  - compression
  - runtime_name and logger/metrics instrumentation
  - circuit_breaker_config
  - load_balancer_type and gRPC channel options

- [ ] Preserve these controls during incremental address updates without resetting circuit state for unchanged addresses.

- [ ] Require a finite timeout_send for scheduler dispatch. Refuse startup or emit a high-severity configuration error when scheduler dispatch uses no executor RPC deadline.

- [ ] Keep the configured mock prefetch of 4 after every full streamer rebuild. Do not silently turn it into prefetch=0.

- [ ] Make retry policy explicit:
  - retry UNAVAILABLE, DEADLINE_EXCEEDED, and NOT_FOUND only when configured
  - do not retry CANCELLED, UNKNOWN, or INTERNAL by default
  - return InternalNetworkError or SendFailure when retryable attempts are exhausted
  - never fall through the retry loop with an implicit None result

- [ ] When all circuits are open, return all_replicas_unhealthy immediately. Remove the fallback that selects an already-open connection.

- [ ] Record transport success only for a completed transport call. Keep ordinary application response failures separate from transport circuit statistics.

- [ ] Replace per-attempt default-thread-pool acknowledgement polling:
  - create one acknowledgement waiter per logical dispatch
  - prefer an asynchronous status-store watch or async polling API
  - if synchronous store APIs must remain, use dedicated bounded executors for desired-state writes and acknowledgement waits so a failed worker cannot starve admission writes
  - make acknowledgement waits cooperatively cancellable when the attempt becomes terminal, recovered, or replaced

- [ ] Add metrics and trace fields for:
  - configured prefetch, RPC timeout, and retries at streamer creation
  - replica_count and available_replica_count
  - circuit-open fast rejection
  - retry count and final transport status

### Required tests

- [ ] Full gateway rebuild retains prefetch=4 from the mock service configuration.
- [ ] Full gateway rebuild retains finite timeout_send and explicit retries.
- [ ] Incremental address update preserves circuit state for unchanged nodes.
- [ ] INTERNAL with retries=0 returns one explicit failure.
- [ ] INTERNAL with an explicitly enabled retry policy performs the configured bounded attempts and returns a failure.
- [ ] No retry branch returns None as its final result.
- [ ] All-open circuits fail fast and select no connection.
- [ ] Application failure does not open the transport circuit.
- [ ] Multiple transport attempts create one logical acknowledgement waiter.
- [ ] Saturated or timed-out acknowledgement waits cannot delay desired-state admission work in the same thread pool.
- [ ] Terminal or recovered attempts cancel their acknowledgement waiter without leaking a thread or task.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/serve/runtimes/gateway/test_dynamic_streamer_config.py \
  tests/unit/serve/networking/test_retry_policy.py \
  tests/unit/serve/networking/test_circuit_breaker_routing.py \
  tests/unit/job/test_worker_ack_isolation.py -q
~~~

---

## Task 10: Add the concrete gateway end-to-end fault suite

This task uses the existing mock_parallel_subgraphs topology for two separate gateway qualifications. Do not combine their fault controls in one run: each run proves a different lifecycle boundary.

**Files**

- config/service/mock/marie-mock-scheduler-test.yml
- /mnt/data/marie-ai/config/service/mock/marie-mock-scheduler-test.yml, synchronized manually before the run
- marie/scheduler/postgres_scheduler_config.py
- marie/scheduler/psql.py
- marie/job/job_supervisor.py
- marie/job/gateway_job_distributor.py
- marie/serve/networking/__init__.py
- marie/serve/runtimes/servers/marie_gateway.py
- tools/stress/gateway_e2e_stresser.py
- tools/stress/analyze_scheduler_trace.py
- marie/utils/scheduler_trace.py
- stress-test.sh
- tools/stress/README.md
- tests/unit/scheduler/test_dispatch_test_hook.py
- tests/unit/tools/stress/test_dispatch_timeout_trace_assertions.py
- tests/unit/tools/stress/test_no_replicas_trace_assertions.py

### Test-hook implementation

- [ ] Parse test_hooks from scheduler configuration.

- [ ] Ignore dispatch-delay metadata unless allow_dispatch_admission_delay is true.

- [ ] Ignore gateway no-replica metadata unless allow_no_replica_rejection is true.

- [ ] Validate:
  - delay is finite and positive
  - delay does not exceed max_dispatch_admission_delay_seconds
  - requested executor matches the current entrypoint
  - delay exceeds the scheduler dispatch confirmation timeout
  - delay is lower than pre_send_callback_timeout_seconds
  - delay is lower than dispatch_unknown_grace_seconds
  - gateway fault modes are from a closed allowlist
  - every fault targets the current entrypoint exactly
  - every request carries a nonempty, unique stress_run_id

- [ ] Pass the validated delay explicitly through enqueue, JobManager, and JobSupervisor.

- [ ] Apply asyncio.sleep after mark_send_detached and before desired-state write.

- [ ] Add stresser arguments:

~~~text
--dispatch-admission-delay-seconds 6
--dispatch-admission-delay-executor mock_executor_a
--gateway-failure-mode no_replicas
--gateway-failure-executor mock_executor_a
~~~

- [ ] Put the selected fault, executor match, scenario name, and run id into generated metadata.

- [ ] Add timeout, rejection, transport-stage, and reconciliation events to the compact scheduler trace profile.

### Scenario A analyzer: delayed admission

Add an assertion mode to analyze_scheduler_trace.py:

~~~bash
.venv/bin/python tools/stress/analyze_scheduler_trace.py \
  "$MARIE_SCHEDULER_TRACE_PATH" \
  --assert-scenario gateway-dispatch-timeout \
  --run-id "$MARIE_STRESS_RUN_ID" \
  --expected-executor mock_executor_a \
  --min-timeouts 16
~~~

For every affected run_attempt_id, require this sequence:

1. dispatch_admission_fault_injected
2. gateway_dispatch_timeout with stage=post_detach_pre_rpc
3. dispatch_unknown_retained
4. job_supervisor_dispatch_admitted
5. dispatch_unknown_resolved with resolution=admitted
6. executor_attempt_fence_accepted
7. executor_running_recorded
8. exactly one job_terminal_attempt_accepted or run_lease_recovered

Also require:

- no Dispatch FAILED event for the affected attempt
- no dispatch-failure terminal event for the timeout itself
- no dispatch_error or error_message equal to "False"
- no executor_stale_attempt_rejected for the eventual valid attempt
- no duplicate executor_request_received for the same run_attempt_id and node task
- no capacity count above configured executor capacity
- no affected attempt left in dispatch_unknown after the workload drains

Make assertion failures exit nonzero and print the offending job_id and run_attempt_id.

### Scenario B analyzer: no-replica fast rejection

Add:

~~~bash
.venv/bin/python tools/stress/analyze_scheduler_trace.py \
  "$MARIE_SCHEDULER_TRACE_PATH" \
  --assert-scenario gateway-no-replicas \
  --run-id "$MARIE_STRESS_RUN_ID" \
  --expected-executor mock_executor_a \
  --min-rejections 16
~~~

For every affected run_attempt_id, require:

1. gateway_dispatch_submitted
2. gateway_replica_lookup with replica_count=0
3. gateway_dispatch_rejected with reason=no_available_replicas and stage=post_detach_pre_rpc
4. dispatch failure settlement with the same named reason
5. exactly one accepted retry/failure transition according to the configured scheduler policy

Also require:

- rejection latency below dispatch_confirmation_timeout_seconds
- no gateway_dispatch_timeout for the affected attempt
- no gateway_rpc_started
- no executor_request_received
- no dispatch_unknown state
- the scheduler semaphore ticket is released exactly once
- no error field equal to False, None, or a generic dispatch failed when the named reason is known

### Runtime preparation

Synchronize the reviewed source configuration into the mounted runtime copy:

~~~bash
cp config/service/mock/marie-mock-scheduler-test.yml \
  /mnt/data/marie-ai/config/service/mock/marie-mock-scheduler-test.yml
~~~

Before starting or restarting the gateway stack:

~~~bash
export MARIE_STRESS_RUN_ID="dispatch-timeout-$(date +%Y%m%d-%H%M%S)"
export MARIE_SCHEDULER_TRACE_ENABLED=true
export MARIE_SCHEDULER_TRACE_PROFILE=full
export MARIE_SCHEDULER_TRACE_PATH="/tmp/$MARIE_STRESS_RUN_ID-scheduler.jsonl"
~~~

The gateway process must inherit those trace variables.

### Scenario A workload: delayed admission

Use the existing mock_parallel_subgraphs topology and all eight mock executors:

~~~bash
.venv/bin/python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.json \
  --api-key "$GATEWAY_API_KEY" \
  --s3-uri s3://dummy/stress.txt \
  --job-count 16 \
  --run-id "$MARIE_STRESS_RUN_ID" \
  --job-name mock_parallel_subgraphs \
  --planner mock_parallel_subgraphs \
  --required-executor mock_executor_a \
  --required-executor mock_executor_b \
  --required-executor mock_executor_c \
  --required-executor mock_executor_d \
  --required-executor mock_executor_e \
  --required-executor mock_executor_f \
  --required-executor mock_executor_g \
  --required-executor mock_executor_h \
  --mock-process-time 0.05 \
  --mock-failure-rate 0 \
  --dispatch-admission-delay-seconds 6 \
  --dispatch-admission-delay-executor mock_executor_a \
  --submit-rate 10 \
  --submit-concurrency 64 \
  --terminal-timeout 900 \
  --live-report "/tmp/$MARIE_STRESS_RUN_ID-live.html" \
  --report "/tmp/$MARIE_STRESS_RUN_ID-final.html"
~~~

The 6-second delay must exceed the configured dispatch timeout while remaining below both the strict pre-send callback timeout and the unknown-dispatch grace period. Add a startup validation that reports all four budgets and refuses an invalid scenario.

### Scenario B workload: no replicas

Use the same eight-executor topology, but inject the fault before connection acquisition for mock_executor_a:

~~~bash
.venv/bin/python tools/stress/gateway_e2e_stresser.py \
  --config tools/stress/gateway-e2e.config.json \
  --api-key "$GATEWAY_API_KEY" \
  --s3-uri s3://dummy/stress.txt \
  --job-count 16 \
  --run-id "$MARIE_STRESS_RUN_ID" \
  --job-name mock_parallel_subgraphs \
  --planner mock_parallel_subgraphs \
  --required-executor mock_executor_a \
  --required-executor mock_executor_b \
  --required-executor mock_executor_c \
  --required-executor mock_executor_d \
  --required-executor mock_executor_e \
  --required-executor mock_executor_f \
  --required-executor mock_executor_g \
  --required-executor mock_executor_h \
  --gateway-failure-mode no_replicas \
  --gateway-failure-executor mock_executor_a \
  --mock-process-time 0.05 \
  --mock-failure-rate 0 \
  --submit-rate 10 \
  --submit-concurrency 64 \
  --terminal-timeout 900 \
  --live-report "/tmp/$MARIE_STRESS_RUN_ID-live.html" \
  --report "/tmp/$MARIE_STRESS_RUN_ID-final.html"
~~~

This is a request-scoped, gated test fault. Do not remove the executor globally from the shared runtime or mutate production discovery state.

### stress-test.sh target

Add two independent targets:

~~~bash
./stress-test.sh gateway-dispatch-timeout-e2e
./stress-test.sh gateway-no-replicas-e2e
~~~

The target must:

1. require GATEWAY_API_KEY, MARIE_STRESS_RUN_ID, and MARIE_SCHEDULER_TRACE_PATH
2. run only its selected mock_parallel_subgraphs scenario
3. run the matching analyzer assertion mode
4. exit nonzero if either the workload or invariant checker fails
5. print the final report and trace paths

Do not silently delete or reuse an old trace. Require a new run id or refuse a trace containing a prior run with the same id.

Each target requires a distinct MARIE_STRESS_RUN_ID and trace file. Never combine delayed admission and no-replica rejection in the same qualification.

### Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/scheduler/test_dispatch_test_hook.py \
  tests/unit/tools/stress/test_dispatch_timeout_trace_assertions.py \
  tests/unit/tools/stress/test_no_replicas_trace_assertions.py -q

./stress-test.sh gateway-dispatch-timeout-e2e
./stress-test.sh gateway-no-replicas-e2e
~~~

Expected:

- at least 16 deterministic post-detach gateway_dispatch_timeout events
- all affected attempts resolve
- at least 16 deterministic no-replica rejections before the confirmation budget, with zero corresponding timeouts
- no false dispatch cleanup
- no stale executor invocation
- no capacity overcommit

---

## Task 11: Full verification and operator documentation

**Files**

- tools/stress/scheduler-reliability.md
- tools/stress/README.md
- config/psql/high-availability checks that classify attempt_state
- tools/stress/scheduler_correctness.py
- tools/stress/scheduler_reliability_runner.py

### Steps

- [ ] Document dispatch_unknown as a temporary ambiguity state, not a terminal result.

- [ ] Document dispatch_stalled and the bounded reconciliation grace period.

- [ ] Document the independent dispatch, pre-send, semaphore, run-lease, and executor RPC budgets and their required ordering.

- [ ] Document the distinction among pre-RPC rejection, post-RPC unknown delivery, and an executor application response.

- [ ] Document that dynamic topology rebuilds must preserve prefetch, RPC timeout, retries, and circuit-breaker settings.

- [ ] Update correctness queries and reports that enumerate attempt states.

- [ ] Add distinct metrics:
  - dispatch_confirmed_total
  - dispatch_unknown_total
  - dispatch_unknown_resolved_total by resolution
  - dispatch_pre_detach_failed_total
  - dispatch_pre_rpc_rejected_total by reason
  - gateway_no_available_replicas_total
  - gateway_all_replicas_unhealthy_total
  - gateway_transport_retry_total by status
  - gateway_transport_failure_total by final status
  - executor_stale_attempt_rejected_total
  - executor_capacity_adoption_failed_total

- [ ] Verify the old messages are absent:

~~~bash
rg -n "dispatch_error.*False|error_message.*False" marie tests tools
~~~

- [ ] Verify old confirmation wiring is removed:

~~~bash
rg -n "confirmation_event|_signal_confirmation" marie tests
~~~

### Unit and integration suites

~~~bash
.venv/bin/python -m pytest tests/unit/scheduler/ tests/unit/job/ -q
.venv/bin/python -m pytest \
  tests/unit/serve/networking/ \
  tests/unit/serve/runtimes/gateway/ \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py -q
.venv/bin/python -m pytest tests/integration/scheduler/ -q
~~~

Record database-backed skips explicitly. A skipped PostgreSQL fencing test is not proof that worker fencing works.

### Lint

~~~bash
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
~~~

### Regression targets

~~~bash
./stress-test.sh reproduce-dispatch-race
./stress-test.sh gateway-dispatch-timeout-e2e
./stress-test.sh gateway-no-replicas-e2e
~~~

## Final acceptance report

Before claiming completion, report:

- number of injected admission delays
- number of post-detach dispatch timeouts
- number resolved by admission
- number resolved by safe pre-RPC rejection
- number resolved by terminal event
- number resolved by run-lease recovery
- number of stale worker attempts rejected
- maximum observed slot occupancy per executor
- number of capacity-release ownership violations
- duplicate executor invocation count
- attempts left in dispatch_unknown
- occurrences of dispatch_error="False"
- number of no-replica faults injected
- no-replica fast rejections and maximum rejection latency
- no-replica faults incorrectly converted into dispatch timeouts
- effective gateway prefetch, RPC timeout, and retry settings after the last topology rebuild

Required final values for the deterministic happy-path qualification:

- injected delays: at least 16
- post-detach timeouts: equal to affected injected attempts
- unresolved unknown attempts: 0
- stale valid attempts rejected: 0
- capacity overcommit: 0
- capacity-release ownership violations: 0
- duplicate executor invocations: 0
- dispatch_error="False": 0

Required final values for the no-replica qualification:

- injected no-replica faults: at least 16
- named no_available_replicas rejections: equal to injected faults
- corresponding gateway_dispatch_timeout events: 0
- corresponding gateway_rpc_started events: 0
- executor invocations for affected attempts: 0
- slot leaks: 0

## Review decisions requested

Approve or revise these choices before implementation:

1. Use send_detached, not supervisor task creation, as the ambiguity boundary.
2. Cancel only before send_detached; reconcile after it.
3. Remove after_send_callback as an admission signal.
4. Add bounded unknown-dispatch reconciliation with semaphore and run-lease renewal, and release capacity according to recorded scheduler-versus-worker ownership.
5. Add authoritative worker-side PostgreSQL attempt fencing.
6. Reject worker execution when capacity adoption fails.
7. Count unknown as occupied capacity but not confirmed scheduled work.
8. Gate deterministic admission delay behind mock scheduler configuration.
9. Use the existing eight mock executors and mock_parallel_subgraphs for the gateway qualification.
10. Add an independent 15-second production dispatch-confirmation budget rather than deriving it from lease_ttl_seconds.
11. Preserve configured gateway prefetch, RPC timeout, retry policy, and circuit breaker across dynamic rebuilds.
12. Fail fast when no replicas are available or every circuit is open.
13. Run desired-state admission and worker-ack tracking once per logical dispatch, not per transport retry.
14. Keep delayed admission and no-replica rejection as separate gateway qualifications.
