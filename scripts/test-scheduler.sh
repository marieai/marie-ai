#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing checkout-local Python: ${python_bin}" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "rg is required for the legacy database-driver check" >&2
  exit 1
fi

if ! (exec 3<>/dev/tcp/127.0.0.1/2379) 2>/dev/null; then
  echo "Local etcd is required at 127.0.0.1:2379" >&2
  exit 1
fi

cd "${repo_root}"

run_suite() {
  local name=$1
  shift
  echo
  echo "==> ${name}"
  "${python_bin}" -m pytest -q "$@"
}

run_suite "Scheduler units and database pool" \
  tests/unit/scheduler \
  tests/unit/utils/test_scheduler_trace.py \
  tests/unit/agent/tools/database/test_postgres_pool.py \
  tests/unit/storage/test_postgres_pool_backpressure.py \
  tests/unit/executor/storage/test_postgres_handler_psycopg3.py

run_suite "Scheduler messaging and gateway submission" \
  tests/unit/messaging/test_publisher.py \
  tests/unit/serve/runtimes/gateway/test_marie_gateway_submission.py \
  tests/unit/tools/stress/test_analyze_scheduler_trace.py \
  tests/unit/tools/stress/test_gateway_e2e_stresser.py \
  tests/unit/tools/stress/test_scheduler_correctness.py

run_suite "Scheduler-adjacent runtime units" \
  tests/unit/job/test_event_publisher_trace.py \
  tests/unit/job/test_gateway_job_distributor.py \
  tests/unit/job/test_job_manager_monitor_trace.py \
  tests/unit/job/test_job_supervisor_trace.py \
  tests/integration/job/test_job_supervisor.py \
  tests/unit/query_planner/test_guardrail_evaluator.py \
  tests/unit/query_planner/test_guardrail_mapper.py \
  tests/unit/query_planner/test_kb_indexing_params.py \
  tests/sensors \
  tests/unit/utils/test_sensor_storage_init.py \
  tests/unit/serve/runtimes/gateway/test_llm_dispatch_runtime.py \
  tests/unit/serve/runtimes/gateway/test_setup_server_sensor.py \
  tests/unit/serve/test_capacity_from_nodes.py \
  tests/unit/serve/test_discovery_lease_params.py \
  tests/unit/serve/test_timeout_utils.py \
  tests/unit/serve/runtimes/worker/test_status_lease_timings.py \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_failure_reporting.py

run_suite "LLM queue units" \
  tests/unit/engine/llm_queue

echo
echo "Note: stale DocArray v1, JSONPath, and discovery contract tests are excluded."
echo "Process-spawning worker runtime tests are not part of this deterministic runner."

echo
echo "Note: test_job_scheduler_core.py is excluded because it wipes the default local scheduler database."
run_suite "Scheduler integrations" \
  tests/integration/scheduler \
  --ignore=tests/integration/scheduler/test_job_scheduler_core.py

run_suite "Etcd units" \
  tests/unit/job/test_lease_cache.py \
  tests/unit/serve/test_etcd_client_reconnect.py \
  tests/unit/serve/test_etcd_monitor_failed_recovery.py \
  tests/unit/serve/test_registry_reregistration.py \
  tests/unit/serve/test_watch_callback_cancelled.py \
  tests/unit/serve/runtimes/gateway/test_gateway_reconcile.py

run_suite "Etcd integrations" \
  tests/integration/serve/test_etcd_client.py \
  tests/integration/serve/test_etcd_client_transaction.py \
  tests/integration/serve/test_etcd_client_txn_api.py \
  tests/integration/serve/test_semaphore_store.py \
  tests/integration/serve/test_state_store.py

echo
echo "==> Legacy PostgreSQL driver scan"
if rg -n "psycopg2|asyncpg" \
  marie/scheduler/psql.py \
  marie/scheduler/repository \
  marie/storage/database/postgres_pool.py; then
  echo "Legacy PostgreSQL driver reference found" >&2
  exit 1
fi

echo "No psycopg2 or asyncpg references found."
echo
echo "Scheduler and etcd verification passed."
