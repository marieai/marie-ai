# Slice 09: Preserve Gateway Transport Controls

**Status:** Proposed

**Depends on:** [Slice 03](slice-03-pre-rpc-failure-propagation.md)

**Parent task:** [Task 9](../../plans/2026-07-25-dispatch-confirmation-race-fix.md#task-9-preserve-gateway-backpressure-and-make-transport-failure-explicit)

## Scope

Make dynamic gateway topology rebuilds preserve configured transport controls and make terminal transport failure explicit.

## Objective

Ensure a discovered or rebuilt gateway streamer retains bounded prefetch, finite RPC timeout, retry policy, circuit breaker, load balancing, and channel options instead of silently reverting to permissive defaults.

## Touchpoints

- `marie/serve/runtimes/servers/marie_gateway.py`
- `marie/serve/runtimes/gateway/streamer.py`
- `marie/serve/networking/__init__.py`
- gateway load balancers
- `marie/job/job_supervisor.py`
- source mock scheduler configuration
- focused gateway, networking, circuit, and acknowledgement tests

## Work

- Pass all configured transport controls into every full streamer rebuild.
- Preserve circuit state for unchanged addresses during incremental updates.
- Require a finite executor RPC deadline for scheduler dispatch.
- Keep the mock profile's configured prefetch bound after discovery rebuilds.
- Make retryable statuses explicit and return a structured final failure on exhaustion.
- Keep `INTERNAL` non-retryable unless configuration explicitly enables it.
- Fail fast with `all_replicas_unhealthy` when every circuit is open.
- Record transport success only for a completed transport call, separate from application response failure.
- Create acknowledgement tracking once per logical dispatch and isolate any required synchronous status-store work from admission writes.
- Emit effective transport configuration and replica/circuit/retry trace fields.

## Non-goals

- Do not implement executor health or restart policy.
- Do not classify application errors as transport failures.
- Do not use open circuits as fallback routing candidates.
- Do not update the mounted `/mnt/data` configuration as part of source implementation.

## Acceptance criteria

- Full and incremental topology updates preserve the reviewed gateway controls.
- No retry path falls through with `None`.
- All-open circuits select no connection and fail immediately.
- Application failure does not poison transport circuit statistics.
- Multiple transport attempts create one logical admission and acknowledgement flow.
- A blocked acknowledgement wait cannot starve desired-state admission.

## Verification

~~~bash
.venv/bin/python -m pytest \
  tests/unit/serve/runtimes/gateway/test_dynamic_streamer_config.py \
  tests/unit/serve/networking/test_retry_policy.py \
  tests/unit/serve/networking/test_circuit_breaker_routing.py \
  tests/unit/job/test_worker_ack_isolation.py -q
~~~

