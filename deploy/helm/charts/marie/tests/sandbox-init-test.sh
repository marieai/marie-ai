#!/usr/bin/env bash
# Helm render tests for sandbox system-init correctness.
#
# Verifies that when sandbox.enabled=true the chart renders:
#   - sandbox-pg-init Job in wave 0 (mem0 DB + pgvector init)
#   - sandbox-seed-defaults Job in wave 1 with correct PG secret wiring
#   - MinIO provisioning Job with sync-wave "0"
#   - Server Deployment with AWS_MQ stub env vars
#   - PostgreSQL initdb SQL that creates the mem0 database
#   - Correct gateway-to-dep connection env vars
#
# All tests run entirely with `helm template` — no cluster required.
#
# Usage:
#   ./deploy/helm/charts/marie/tests/sandbox-init-test.sh
#
# Exit code: 0 = all tests passed, non-zero = at least one test failed.

set -uo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="test-sbx"
PASS=0
FAIL=0

# ---- helpers ---------------------------------------------------------------

pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

assert_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "${haystack}" | grep -qF "${needle}"; then
        pass "${label}"
    else
        fail "${label} — expected to find: ${needle}"
        echo "  Snippet of rendered output:"
        echo "${haystack}" | head -30
    fi
}

assert_not_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "${haystack}" | grep -qF "${needle}"; then
        fail "${label} — expected NOT to find: ${needle}"
        echo "  Snippet of rendered output:"
        echo "${haystack}" | head -20
    else
        pass "${label}"
    fi
}

# Full sandbox render (sandbox.enabled=true, sandbox.secrets.enabled=true with dummy storeRef).
# Using || true so empty templates don't abort the test run.
render_sandbox() {
    helm template "${RELEASE}" "${CHART_DIR}" \
        -f "${CHART_DIR}/values.yaml" \
        -f "${CHART_DIR}/values-sandbox.yaml" \
        --set "sandbox.enabled=true" \
        --set "sandbox.secrets.storeRef.name=test-store" \
        "$@" 2>&1 || true
}

render_template() {
    local tpl="$1"; shift
    helm template "${RELEASE}" "${CHART_DIR}" \
        -f "${CHART_DIR}/values.yaml" \
        -f "${CHART_DIR}/values-sandbox.yaml" \
        --set "sandbox.enabled=true" \
        --set "sandbox.secrets.storeRef.name=test-store" \
        --show-only "${tpl}" \
        "$@" 2>&1 || true
}

# Expected release prefix used by subchart secrets (<release>-<subchart>)
RELEASE_PG="${RELEASE}-postgresql"       # correct PG secret name (subchart)
WRONG_PG="${RELEASE}-marie-postgresql"   # would be wrong (umbrella fullname + postgresql)
FULLNAME="${RELEASE}-marie"              # umbrella chart fullname

# ---- T1: helm lint ----------------------------------------------------------
echo ""
echo "=== T1: helm lint ==="
if helm lint "${CHART_DIR}" \
        -f "${CHART_DIR}/values.yaml" \
        -f "${CHART_DIR}/values-sandbox.yaml" \
        --set "sandbox.secrets.enabled=true" \
        --set "sandbox.secrets.storeRef.name=test-store" \
        --quiet 2>&1; then
    pass "helm lint passes"
else
    fail "helm lint failed"
fi

# ---- T2: sandbox-pg-init Job renders when sandbox.enabled=true --------------
echo ""
echo "=== T2: sandbox-pg-init Job renders ==="
pg_init=$(render_template "templates/sandbox-pg-init.yaml")

assert_contains "pg-init: kind is Job" \
    "kind: Job" "${pg_init}"

assert_contains "pg-init: name = ${FULLNAME}-sandbox-pg-init" \
    "name: ${FULLNAME}-sandbox-pg-init" "${pg_init}"

# ---- T3: sandbox-pg-init Job is in sync-wave "0" ----------------------------
echo ""
echo "=== T3: sandbox-pg-init Job annotated with sync-wave 0 ==="
assert_contains "pg-init: sync-wave annotation present" \
    "argocd.argoproj.io/sync-wave: \"0\"" "${pg_init}"

# ---- T4: sandbox-pg-init Job references correct PG secret -------------------
echo ""
echo "=== T4: sandbox-pg-init references correct PostgreSQL secret ==="
assert_contains "pg-init: secretKeyRef name = ${RELEASE_PG}" \
    "name: ${RELEASE_PG}" "${pg_init}"

assert_not_contains "pg-init: does NOT reference wrong secret (${WRONG_PG})" \
    "name: ${WRONG_PG}" "${pg_init}"

assert_contains "pg-init: secretKeyRef key = postgres-password" \
    "key: postgres-password" "${pg_init}"

# ---- T5: sandbox-pg-init Job creates mem0 DB and enables pgvector -----------
echo ""
echo "=== T5: sandbox-pg-init command creates mem0 + pgvector ==="
assert_contains "pg-init: creates mem0 database" \
    "CREATE DATABASE mem0" "${pg_init}"

assert_contains "pg-init: enables vector extension in mem0" \
    "CREATE EXTENSION IF NOT EXISTS vector" "${pg_init}"

# ---- T6: sandbox-pg-init does NOT render when sandbox.enabled=false ---------
echo ""
echo "=== T6: sandbox-pg-init absent when sandbox.enabled=false ==="
disabled=$(helm template "${RELEASE}" "${CHART_DIR}" \
    -f "${CHART_DIR}/values.yaml" \
    -f "${CHART_DIR}/values-sandbox.yaml" \
    --set "sandbox.enabled=false" \
    --show-only templates/sandbox-pg-init.yaml \
    2>&1 || true)
assert_not_contains "pg-init: not rendered when sandbox.enabled=false" \
    "kind: Job" "${disabled}"

# ---- T7: sandbox-seed-defaults references correct PG secret name -----------
echo ""
echo "=== T7: sandbox-seed-defaults uses correct PG secret name ==="
seed=$(render_template "templates/sandbox-seed-defaults.yaml")

assert_contains "seed-defaults: renders" \
    "kind: Job" "${seed}"

assert_contains "seed-defaults: secretKeyRef name = ${RELEASE_PG}" \
    "name: ${RELEASE_PG}" "${seed}"

assert_not_contains "seed-defaults: does NOT reference wrong secret" \
    "name: ${WRONG_PG}" "${seed}"

# ---- T8: sandbox-seed-defaults uses postgres-password key -------------------
echo ""
echo "=== T8: sandbox-seed-defaults uses postgres-password key ==="
assert_contains "seed-defaults: key = postgres-password" \
    "key: postgres-password" "${seed}"

# ---- T9: sandbox-seed-defaults is in sync-wave "1" -------------------------
echo ""
echo "=== T9: sandbox-seed-defaults annotated with sync-wave 1 ==="
assert_contains "seed-defaults: sync-wave annotation = 1" \
    "argocd.argoproj.io/sync-wave: \"1\"" "${seed}"

# ---- T10: MinIO provisioning Job has sync-wave "0" -------------------------
echo ""
echo "=== T10: MinIO provisioning Job sync-wave 0 ==="
minio_job=$(render_template "charts/minio/templates/provisioning-job.yaml")
assert_contains "minio-provision: sync-wave annotation = 0" \
    "argocd.argoproj.io/sync-wave: \"0\"" "${minio_job}"

# ---- T11: PostgreSQL initdb SQL creates mem0 database -----------------------
echo ""
echo "=== T11: postgresql initdb SQL creates mem0 DB ==="
pg_cm=$(render_template "charts/postgresql/templates/configmap-initdb.yaml")
assert_contains "postgresql initdb: CREATE DATABASE mem0" \
    "CREATE DATABASE mem0" "${pg_cm}"

# ---- T12: postgresql initdb SQL creates pgvector extension ------------------
echo ""
echo "=== T12: postgresql initdb SQL enables pgvector in postgres DB ==="
assert_contains "postgresql initdb: CREATE EXTENSION IF NOT EXISTS vector" \
    "CREATE EXTENSION IF NOT EXISTS vector" "${pg_cm}"

# ---- T13: Server Deployment has AWS_MQ stub env vars -----------------------
echo ""
echo "=== T13: Server Deployment has AWS_MQ stub env vars ==="
server_deploy=$(render_template "charts/server/templates/deployment.yaml")
assert_contains "server: AWS_MQ_HOSTNAME stub" \
    "AWS_MQ_HOSTNAME" "${server_deploy}"
assert_contains "server: AWS_MQ_USERNAME stub" \
    "AWS_MQ_USERNAME" "${server_deploy}"
assert_contains "server: AWS_MQ_PASSWORD stub" \
    "AWS_MQ_PASSWORD" "${server_deploy}"

# ---- T14: Server Deployment wires PostgreSQL connection ---------------------
echo ""
echo "=== T14: Server Deployment wires PostgreSQL ==="
assert_contains "server: DATABASE_HOST set to ${RELEASE}-postgresql" \
    "value: \"${RELEASE}-postgresql\"" "${server_deploy}"

assert_contains "server: PG secretKeyRef name = ${RELEASE}-postgresql" \
    "name: ${RELEASE}-postgresql" "${server_deploy}"

# ---- T15: Server Deployment wires RabbitMQ connection ----------------------
echo ""
echo "=== T15: Server Deployment wires RabbitMQ ==="
assert_contains "server: RABBIT_MQ_HOSTNAME set" \
    "RABBIT_MQ_HOSTNAME" "${server_deploy}"
assert_contains "server: RabbitMQ secretKeyRef name = ${RELEASE}-rabbitmq" \
    "name: ${RELEASE}-rabbitmq" "${server_deploy}"

# ---- T16: Server Deployment wires MinIO S3 connection ----------------------
echo ""
echo "=== T16: Server Deployment wires MinIO/S3 ==="
assert_contains "server: S3_ENDPOINT_URL set" \
    "S3_ENDPOINT_URL" "${server_deploy}"
assert_contains "server: MinIO secretKeyRef name = ${RELEASE}-minio" \
    "name: ${RELEASE}-minio" "${server_deploy}"

# ---- T17: Server Deployment wires etcd discovery ---------------------------
echo ""
echo "=== T17: Server Deployment wires etcd ==="
assert_contains "server: ETCD_ENDPOINTS set" \
    "ETCD_ENDPOINTS" "${server_deploy}"
assert_contains "server: etcd host = ${RELEASE}-etcd" \
    "${RELEASE}-etcd" "${server_deploy}"

# ---- T18: Server Deployment wires Valkey LLM queue -------------------------
echo ""
echo "=== T18: Server Deployment wires Valkey (LLM queue) ==="
assert_contains "server: LLM_QUEUE_VALKEY_URL set" \
    "LLM_QUEUE_VALKEY_URL" "${server_deploy}"
assert_contains "server: Valkey URL points to ${RELEASE}-valkey" \
    "${RELEASE}-valkey" "${server_deploy}"

# ---- T19: wave ordering: full render contains both wave annotations --------
echo ""
echo "=== T19: Wave ordering — both wave annotations present in full render ==="
# T3 verifies pg-init is wave 0; T9 verifies seed-defaults is wave 1.
# This test confirms both coexist in the same full render without template errors.
all=$(render_sandbox)

wave0_count=$(echo "${all}" | grep -c 'sync-wave: "0"' || true)
wave1_count=$(echo "${all}" | grep -c 'sync-wave: "1"' || true)

if [ "${wave0_count}" -ge 2 ]; then
    pass "full render contains at least 2 wave-0 resources (minio-provision + pg-init; got ${wave0_count})"
else
    fail "expected >= 2 wave-0 resources, got ${wave0_count}"
fi
if [ "${wave1_count}" -ge 1 ]; then
    pass "full render contains at least 1 wave-1 resource (seed-defaults; got ${wave1_count})"
else
    fail "expected >= 1 wave-1 resource, got ${wave1_count}"
fi

# ---- T20: full render has no helm template errors --------------------------
echo ""
echo "=== T20: Full sandbox render exits cleanly ==="
full_render_output=$(helm template "${RELEASE}" "${CHART_DIR}" \
    -f "${CHART_DIR}/values.yaml" \
    -f "${CHART_DIR}/values-sandbox.yaml" \
    --set "sandbox.enabled=true" \
    --set "sandbox.secrets.storeRef.name=test-store" \
    2>&1)
full_render_exit=$?
if [ ${full_render_exit} -eq 0 ]; then
    pass "helm template --set sandbox.enabled=true exits 0"
else
    fail "helm template exited ${full_render_exit}"
    echo "${full_render_output}" | head -20
fi

# ---- summary ----------------------------------------------------------------
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
