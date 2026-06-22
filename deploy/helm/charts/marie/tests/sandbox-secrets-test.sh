#!/usr/bin/env bash
# Helm render tests for templates/sandbox-secrets.yaml (Slice 6 — ESO delivery).
#
# Tests run entirely with `helm template` — no cluster required.
# Uses --show-only to isolate the sandbox-secrets template from the rest of the chart.
#
# Note: helm template --show-only returns exit=1 when the selected template renders
# nothing (empty output).  The render_secrets() helper captures output regardless of
# exit code so negative assertions (ExternalSecrets absent when disabled) still work.
#
# Usage:
#   ./deploy/helm/charts/marie/tests/sandbox-secrets-test.sh
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

# Render only sandbox-secrets.yaml with arbitrary extra --set flags.
# Returns empty string (and exit 0) when the template renders nothing — helm
# itself returns exit 1 in that case, which we swallow deliberately here.
render_secrets() {
    helm template "${RELEASE}" "${CHART_DIR}" \
        -f "${CHART_DIR}/values.yaml" \
        -f "${CHART_DIR}/values-sandbox.yaml" \
        --show-only templates/sandbox-secrets.yaml \
        "$@" 2>&1 || true
}

# Assert that the rendered output contains a string.
assert_contains() {
    local label="$1"
    local needle="$2"
    local haystack="$3"
    if echo "${haystack}" | grep -qF "${needle}"; then
        pass "${label}"
    else
        fail "${label} — expected to find: ${needle}"
        echo "  Rendered output:"
        echo "${haystack}" | head -40
    fi
}

# Assert that the rendered output does NOT contain a string.
assert_not_contains() {
    local label="$1"
    local needle="$2"
    local haystack="$3"
    if echo "${haystack}" | grep -qF "${needle}"; then
        fail "${label} — expected NOT to find: ${needle}"
        echo "  Rendered output:"
        echo "${haystack}" | head -20
    else
        pass "${label}"
    fi
}

# ---- Test fixture values ----------------------------------------------------
# Minimal set of --set flags enabling all four ExternalSecrets.
STORE_NAME="vault-backend"
STORE_KIND="ClusterSecretStore"
REFRESH="5m"
ADMIN_KEY="sandboxes/sbx-test-a1b2c3/admin"
DB_KEY="sandboxes/sbx-test-a1b2c3/db"
STORAGE_KEY="sandboxes/sbx-test-a1b2c3/storage"
RUNNER_KEY="sandboxes/sbx-test-a1b2c3/runner"

ENABLED_FLAGS=(
    --set "sandbox.secrets.enabled=true"
    --set "sandbox.secrets.storeRef.name=${STORE_NAME}"
    --set "sandbox.secrets.storeRef.kind=${STORE_KIND}"
    --set "sandbox.secrets.refreshInterval=${REFRESH}"
    --set "sandbox.secrets.secrets.adminApiKey.remoteKey=${ADMIN_KEY}"
    --set "sandbox.secrets.secrets.dbPassword.remoteKey=${DB_KEY}"
    --set "sandbox.secrets.secrets.storageCredentials.remoteKey=${STORAGE_KEY}"
    --set "sandbox.secrets.secrets.runnerSecret.remoteKey=${RUNNER_KEY}"
)

# Expected fullname = <release>-marie (release "test-sbx" does not contain "marie")
FULLNAME="${RELEASE}-marie"

# ---- T1: helm lint passes ---------------------------------------------------
echo ""
echo "=== T1: helm lint ==="
if helm lint "${CHART_DIR}" \
        -f "${CHART_DIR}/values.yaml" \
        -f "${CHART_DIR}/values-sandbox.yaml" \
        --set "sandbox.secrets.enabled=true" \
        --set "sandbox.secrets.storeRef.name=${STORE_NAME}" \
        --quiet 2>&1; then
    pass "helm lint passes"
else
    fail "helm lint failed"
fi

# ---- T2: ExternalSecrets render when enabled --------------------------------
echo ""
echo "=== T2: ExternalSecrets render when sandbox.enabled=true + sandbox.secrets.enabled=true ==="
output=$(render_secrets "${ENABLED_FLAGS[@]}")

assert_contains "renders kind: ExternalSecret" \
    "kind: ExternalSecret" "${output}"

COUNT=$(echo "${output}" | grep -c "kind: ExternalSecret" || true)
if [ "${COUNT}" -eq 4 ]; then
    pass "renders exactly 4 ExternalSecret resources (got ${COUNT})"
else
    fail "expected 4 ExternalSecret resources, got ${COUNT}"
fi

# ---- T3: Admin-key ExternalSecret — correct target Secret name and key ------
echo ""
echo "=== T3: Admin-key ExternalSecret produces correct target Secret name and key ==="

# spec.target.name must equal <fullname>-sandbox-admin
assert_contains "admin ExternalSecret target name = ${FULLNAME}-sandbox-admin" \
    "name: ${FULLNAME}-sandbox-admin" "${output}"

# data[].secretKey must be api_key
assert_contains "admin ExternalSecret secretKey = api_key" \
    "secretKey: api_key" "${output}"

# ---- T4: Admin-key ExternalSecret — correct store reference -----------------
echo ""
echo "=== T4: Admin-key ExternalSecret references the configured store ==="
assert_contains "storeRef name = ${STORE_NAME}" \
    "name: \"${STORE_NAME}\"" "${output}"
assert_contains "storeRef kind = ${STORE_KIND}" \
    "kind: ${STORE_KIND}" "${output}"

# ---- T5: Admin-key ExternalSecret — correct remoteKey ----------------------
echo ""
echo "=== T5: Admin-key ExternalSecret references the configured remoteKey ==="
assert_contains "admin remoteKey = ${ADMIN_KEY}" \
    "key: \"${ADMIN_KEY}\"" "${output}"

# ---- T6: refreshInterval is propagated --------------------------------------
echo ""
echo "=== T6: refreshInterval is propagated to all ExternalSecrets ==="
assert_contains "refreshInterval = ${REFRESH}" \
    "refreshInterval: \"${REFRESH}\"" "${output}"

# ---- T7: db ExternalSecret target name and key ------------------------------
echo ""
echo "=== T7: DB password ExternalSecret ==="
assert_contains "db ExternalSecret target name = ${FULLNAME}-sandbox-db" \
    "name: ${FULLNAME}-sandbox-db" "${output}"
assert_contains "db ExternalSecret secretKey = password" \
    "secretKey: password" "${output}"
assert_contains "db remoteKey = ${DB_KEY}" \
    "key: \"${DB_KEY}\"" "${output}"

# ---- T8: Storage ExternalSecret has two data entries (access-key + secret-key)
echo ""
echo "=== T8: Storage ExternalSecret has both access-key and secret-key entries ==="
assert_contains "storage ExternalSecret target name = ${FULLNAME}-sandbox-storage" \
    "name: ${FULLNAME}-sandbox-storage" "${output}"
assert_contains "storage ExternalSecret secretKey = access-key" \
    "secretKey: access-key" "${output}"
assert_contains "storage ExternalSecret secretKey = secret-key" \
    "secretKey: secret-key" "${output}"

# ---- T9: Runner ExternalSecret target name and key --------------------------
echo ""
echo "=== T9: Runner ExternalSecret ==="
assert_contains "runner ExternalSecret target name = ${FULLNAME}-sandbox-runner" \
    "name: ${FULLNAME}-sandbox-runner" "${output}"
assert_contains "runner ExternalSecret secretKey = token" \
    "secretKey: token" "${output}"

# ---- T10: No ExternalSecrets when sandbox.enabled=false --------------------
echo ""
echo "=== T10: No ExternalSecrets when sandbox.enabled=false ==="
disabled_output=$(render_secrets \
    --set "sandbox.enabled=false" \
    --set "sandbox.secrets.enabled=true" \
    --set "sandbox.secrets.storeRef.name=${STORE_NAME}" \
    --set "sandbox.secrets.secrets.adminApiKey.remoteKey=${ADMIN_KEY}" \
    --set "sandbox.secrets.secrets.dbPassword.remoteKey=${DB_KEY}" \
    --set "sandbox.secrets.secrets.storageCredentials.remoteKey=${STORAGE_KEY}" \
    --set "sandbox.secrets.secrets.runnerSecret.remoteKey=${RUNNER_KEY}")
assert_not_contains "no ExternalSecret when sandbox.enabled=false" \
    "kind: ExternalSecret" "${disabled_output}"

# ---- T11: No ExternalSecrets when sandbox.secrets.enabled=false -------------
echo ""
echo "=== T11: No ExternalSecrets when sandbox.secrets.enabled=false ==="
secrets_disabled_output=$(render_secrets \
    --set "sandbox.secrets.enabled=false" \
    --set "sandbox.secrets.storeRef.name=${STORE_NAME}" \
    --set "sandbox.secrets.secrets.adminApiKey.remoteKey=${ADMIN_KEY}")
assert_not_contains "no ExternalSecret when sandbox.secrets.enabled=false" \
    "kind: ExternalSecret" "${secrets_disabled_output}"

# ---- T12: No ExternalSecret for entries with empty remoteKey ----------------
echo ""
echo "=== T12: Entry with empty remoteKey is skipped ==="
# Only adminApiKey has a remoteKey; db/storage/runner use defaults (empty remoteKey)
partial_output=$(render_secrets \
    --set "sandbox.secrets.enabled=true" \
    --set "sandbox.secrets.storeRef.name=${STORE_NAME}" \
    --set "sandbox.secrets.secrets.adminApiKey.remoteKey=${ADMIN_KEY}")
assert_contains "admin ExternalSecret still renders" \
    "kind: ExternalSecret" "${partial_output}"
assert_not_contains "db ExternalSecret skipped when remoteKey empty" \
    "${FULLNAME}-sandbox-db" "${partial_output}"
assert_not_contains "storage ExternalSecret skipped when remoteKey empty" \
    "${FULLNAME}-sandbox-storage" "${partial_output}"
assert_not_contains "runner ExternalSecret skipped when remoteKey empty" \
    "${FULLNAME}-sandbox-runner" "${partial_output}"

# ---- T13: Custom secretName override ----------------------------------------
echo ""
echo "=== T13: Custom secretName override ==="
custom_name="custom-admin-secret"
custom_output=$(render_secrets \
    "${ENABLED_FLAGS[@]}" \
    --set "sandbox.secrets.secrets.adminApiKey.secretName=${custom_name}")
assert_contains "custom admin secretName is used as target.name" \
    "name: ${custom_name}" "${custom_output}"
assert_not_contains "default admin secret name not used when overridden" \
    "name: ${FULLNAME}-sandbox-admin" "${custom_output}"

# ---- T14: ESO apiVersion is external-secrets.io/v1 -------------------------
echo ""
echo "=== T14: ESO apiVersion is external-secrets.io/v1 ==="
assert_contains "apiVersion = external-secrets.io/v1" \
    "apiVersion: external-secrets.io/v1" "${output}"

# ---- T15: admin-key secret name matches seed-defaults Job expectation -------
echo ""
echo "=== T15: Admin-key secret name matches seed-defaults Job default ==="
# The seed-defaults Job defaults to <fullname>-sandbox-admin (key: api_key).
# Verify the ExternalSecret target name equals that value when secretName is empty.
assert_contains "admin target = <fullname>-sandbox-admin (matches seed-defaults Job)" \
    "name: ${FULLNAME}-sandbox-admin" "${output}"

# ---- summary ----------------------------------------------------------------
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
