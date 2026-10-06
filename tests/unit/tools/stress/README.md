
  - Normal: 60/60 completed; exactly 20 jobs in each small, medium, and large pool.
  - Five-second delay: 6/6 completed; zero lost or open jobs.
  - Error discovery: AIMock produced ambiguous disconnects instead of HTTP errors, leaving four outcome_unknown reservations. AIMock now returns an explicit HTTP 503.
  - Verification: 61 Python tests passed; AIMock unit test passed; rebuilt image returned HTTP 503 with provider_unavailable.

  Before testing continues, manually stop the gateway and annotator. Then clear the contaminated local queue state:

  docker exec marie-valkey sh -lc \
    'valkey-cli --scan --pattern "llm:v3:{fabric:default}:*" |
     while IFS= read -r key; do
       valkey-cli UNLINK "$key" >/dev/null
     done'

  Use your existing scheduler database clearing procedure, then activate the rebuilt AIMock image:

  cd /home/gbugaj/dev/marieai/marie-ai/Dockerfiles

  docker compose \
    -f docker-compose.mock-llm-programmatic.yml \
    up -d --no-build --force-recreate aimock-programmatic

  Start the gateway and annotator manually, then tell me done. I’ll run the explicit HTTP-error and chaos phases through the new replay suite.

  The reusable command is:

  cd /home/gbugaj/dev/marieai/marie-ai

  .venv/bin/python tools/stress/llm_dispatch_stress_suite.py \
    --runtime-mode queued-dispatch \
    --suite-id local-baseline

  Later, after starting the annotator in direct-batch mode, use the same workload:

  .venv/bin/python tools/stress/llm_dispatch_stress_suite.py \
    --runtime-mode direct-batch \
    --suite-id local-baseline

  Artifacts are written under ~/tmp/llm-dispatch-stress/, including phase reports and comparison JSON/Markdown.
