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

declare -A pytest_totals=(
  [passed]=0
  [failed]=0
  [skipped]=0
  [xfailed]=0
  [xpassed]=0
  [errors]=0
  [deselected]=0
  [warnings]=0
)
suite_count=0
summary_dir=$(mktemp -d)
trap 'rm -rf -- "${summary_dir}"' EXIT

run_suite() {
  local name=$1
  shift
  echo
  echo "==> ${name}"

  local suite_log="${summary_dir}/suite-${suite_count}.log"
  set +e
  "${python_bin}" -m pytest -q "$@" 2>&1 | tee "${suite_log}"
  local pipeline_status=("${PIPESTATUS[@]}")
  set -e

  if ((pipeline_status[0] != 0)); then
    return "${pipeline_status[0]}"
  fi
  if ((pipeline_status[1] != 0)); then
    return "${pipeline_status[1]}"
  fi

  local summary_line
  summary_line=$(rg -N '[0-9]+ (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?).* in [0-9.]+s' "${suite_log}" | tail -n 1 || true)
  if [[ -z "${summary_line}" ]]; then
    echo "Unable to read pytest summary for: ${name}" >&2
    return 1
  fi

  local count outcome
  while read -r count outcome; do
    case "${outcome}" in
      error | errors) outcome=errors ;;
      warning | warnings) outcome=warnings ;;
    esac
    pytest_totals["${outcome}"]=$((pytest_totals["${outcome}"] + count))
  done < <(printf '%s\n' "${summary_line}" | rg -o '[0-9]+ (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?)')

  suite_count=$((suite_count + 1))
}

print_summary() {
  echo
  echo "==> Verification totals"
  echo "Pytest suites: ${suite_count} passed"

  printf 'Pytest totals:'
  local separator=' '
  local outcome count
  for outcome in passed xfailed skipped xpassed failed errors deselected warnings; do
    count=${pytest_totals["${outcome}"]}
    if ((count > 0)); then
      printf '%s%s %s' "${separator}" "${count}" "${outcome}"
      separator=', '
    fi
  done
  printf '\n'
  echo "Legacy PostgreSQL driver scan: passed"
}

run_suite "Scheduler units, SLA configuration, and database pool" \
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
  tests/unit/serve/runtimes/worker/test_worker_request_handler_failure_reporting.py \
  tests/unit/serve/runtimes/worker/test_worker_request_handler_semaphore.py

run_suite "LLM queue units" \
  tests/unit/engine/llm_queue

echo
echo "Note: stale DocArray v1, JSONPath, and discovery contract tests are excluded."
echo "Process-spawning worker runtime tests are not part of this deterministic runner."

echo
echo "Note: test_job_scheduler_core.py is excluded because it wipes the default local scheduler database."
run_suite "Scheduler integrations, SLA ordering, and persistence" \
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
print_summary
echo
echo "Scheduler and etcd verification passed."
