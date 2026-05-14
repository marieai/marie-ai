#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/stress/run_gateway_benchmark_matrix.sh --executor-count <1|2> [options]

Required:
  --executor-count <n>     Logical extract executor slots under test.
  --input-dir <path>       Local input directory for upload-mode runs.

Optional:
  --config <path>          Stress config file.
  --job-name <name>        Job name submitted to the gateway.
  --planner <name>         Planner name.
  --run-time <duration>    Run duration per rate.
  --soft-sla-seconds <n>   Soft SLA deadline from submit start.
  --hard-sla-seconds <n>   Hard SLA deadline from submit start.
  --min-soft-sla <pct>     Required soft SLA compliance.
  --min-hard-sla <pct>     Required hard SLA compliance.
  --submit-concurrency <n> Submit concurrency.
  --debug-sample-interval <n>
                           Gateway debug sample interval.
  --progress-interval <n>  Live progress and live-report refresh cadence.
  --report-dir <path>      Output directory for reports.
  --rates "a b c"          Override the default rate sweep.

Defaults:
  --config tools/stress/gateway-e2e.config.example.json
  --job-name extract
  --planner extract
  --run-time 2m
  --soft-sla-seconds 15
  --hard-sla-seconds 45
  --min-soft-sla 95
  --min-hard-sla 99
  --submit-concurrency 10
  --debug-sample-interval 5
  --progress-interval 5
  --report-dir /tmp/gateway-benchmark-matrix

Default rate sweeps:
  executor-count=1 -> 0.6 0.7 0.8 0.9 1.0
  executor-count=2 -> 1.0 1.2 1.4 1.6 1.8 2.0
EOF
}

executor_count=""
config="tools/stress/gateway-e2e.config.example.json"
input_dir=""
job_name="extract"
planner="extract"
run_time="2m"
soft_sla_seconds="15"
hard_sla_seconds="45"
min_soft_sla="95"
min_hard_sla="99"
submit_concurrency="10"
debug_sample_interval="5"
progress_interval="5"
report_dir="/tmp/gateway-benchmark-matrix"
rates_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --executor-count)
      executor_count="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --input-dir)
      input_dir="$2"
      shift 2
      ;;
    --job-name)
      job_name="$2"
      shift 2
      ;;
    --planner)
      planner="$2"
      shift 2
      ;;
    --run-time)
      run_time="$2"
      shift 2
      ;;
    --soft-sla-seconds)
      soft_sla_seconds="$2"
      shift 2
      ;;
    --hard-sla-seconds)
      hard_sla_seconds="$2"
      shift 2
      ;;
    --min-soft-sla)
      min_soft_sla="$2"
      shift 2
      ;;
    --min-hard-sla)
      min_hard_sla="$2"
      shift 2
      ;;
    --submit-concurrency)
      submit_concurrency="$2"
      shift 2
      ;;
    --debug-sample-interval)
      debug_sample_interval="$2"
      shift 2
      ;;
    --progress-interval)
      progress_interval="$2"
      shift 2
      ;;
    --report-dir)
      report_dir="$2"
      shift 2
      ;;
    --rates)
      rates_override="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$executor_count" || -z "$input_dir" ]]; then
  usage >&2
  exit 1
fi

case "$executor_count" in
  1)
    default_rates=(0.6 0.7 0.8 0.9 1.0)
    ;;
  2)
    default_rates=(1.0 1.2 1.4 1.6 1.8 2.0)
    ;;
  *)
    echo "Unsupported executor count: $executor_count" >&2
    exit 1
    ;;
esac

if [[ -n "$rates_override" ]]; then
  read -r -a rates <<<"$rates_override"
else
  rates=("${default_rates[@]}")
fi

mkdir -p "$report_dir"

summary_file="$report_dir/summary.txt"
current_live_path="$report_dir/current-live.html"
: >"$summary_file"

for rate in "${rates[@]}"; do
  safe_rate="${rate//./_}"
  report_path="$report_dir/executors-${executor_count}-rate-${safe_rate}.html"
  live_report_path="$report_dir/executors-${executor_count}-rate-${safe_rate}-live.html"
  log_path="$report_dir/executors-${executor_count}-rate-${safe_rate}.log"

  ln -sfn "$(basename "$live_report_path")" "$current_live_path"

  echo "=== rate=${rate} executors=${executor_count} ===" | tee -a "$summary_file"
  echo "report=${report_path}" | tee -a "$summary_file"
  echo "live_report=${live_report_path}" | tee -a "$summary_file"
  echo "current_live=${current_live_path}" | tee -a "$summary_file"

  if python tools/stress/gateway_e2e_stresser.py \
    --config "$config" \
    --input-dir "$input_dir" \
    --job-name "$job_name" \
    --planner "$planner" \
    --submit-concurrency "$submit_concurrency" \
    --progress-interval "$progress_interval" \
    --live-report "$live_report_path" \
    --report "$report_path" \
    --run-time "$run_time" \
    --submit-rate "$rate" \
    --debug-sample-interval "$debug_sample_interval" \
    --soft-sla-seconds "$soft_sla_seconds" \
    --hard-sla-seconds "$hard_sla_seconds" \
    --min-soft-sla-compliance-pct "$min_soft_sla" \
    --min-hard-sla-compliance-pct "$min_hard_sla" \
    >"$log_path" 2>&1; then
    status="PASS"
  else
    status="FAIL"
  fi

  echo "status=${status}" | tee -a "$summary_file"
  echo "log=${log_path}" | tee -a "$summary_file"
  echo | tee -a "$summary_file"
done

echo "Wrote benchmark matrix outputs to ${report_dir}"
