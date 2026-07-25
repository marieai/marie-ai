#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
  MARIE_STRESS_PYTHON=.venv/bin/python
else
  MARIE_STRESS_PYTHON=python3
fi

case "${1:-}" in
  reproduce-dispatch-race)
    exec "${MARIE_STRESS_PYTHON}" -m pytest -vv \
      tests/unit/scheduler/test_dispatch_confirmation_race.py::test_late_confirmation_reproduces_false_dispatch_cleanup_race
    ;;
  gateway-e2e)
    : "${GATEWAY_API_KEY:?Set GATEWAY_API_KEY before running the live stress test}"
    exec "${MARIE_STRESS_PYTHON}" tools/stress/gateway_e2e_stresser.py \
      --config tools/stress/gateway-e2e.config.json \
      --api-key "${GATEWAY_API_KEY}" \
      --input-dir "${MARIE_STRESS_INPUT_DIR:-${HOME}/.marie/generators}" \
      --job-count "${MARIE_STRESS_JOB_COUNT:-1000}" \
      --job-name gen5_extract \
      --planner mock_annotator_llm \
      --llm-pool-id document-small \
      --ref-type stress \
      --project-id mock-annotator-llm-stress \
      --request-template tools/stress/mock_annotator_llm.invoke.json \
      --submit-rate "${MARIE_STRESS_SUBMIT_RATE:-10}" \
      --submit-concurrency "${MARIE_STRESS_SUBMIT_CONCURRENCY:-64}" \
      --terminal-timeout "${MARIE_STRESS_TERMINAL_TIMEOUT:-14400}" \
      --live-report "${MARIE_STRESS_LIVE_REPORT:-${HOME}/tmp/gateway-e2e-live.html}" \
      --report "${MARIE_STRESS_FINAL_REPORT:-${HOME}/tmp/gateway-e2e-final.html}"
    ;;
  *)
    printf 'Usage: %s {reproduce-dispatch-race|gateway-e2e}\n' "$0" >&2
    exit 2
    ;;
esac
