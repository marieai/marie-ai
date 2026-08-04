#!/usr/bin/env bash
set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
helm="${HELM:-helm}"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "${temporary_dir}"' EXIT

replicas_for() {
  local manifest="$1"
  local resource_name="$2"
  awk -v wanted="${resource_name}" '
    /^---$/ { name = ""; in_metadata = 0; in_spec = 0 }
    /^metadata:$/ { in_metadata = 1; next }
    in_metadata && /^  name: / { name = $2; in_metadata = 0 }
    /^spec:$/ { in_spec = 1; next }
    in_spec && /^  replicas: / && name == wanted { print $2; exit }
  ' "${manifest}"
}

workloads=(lifecycle-proof-server lifecycle-proof-executor-cpu-local)
infrastructure=(
  lifecycle-proof-gitea
  lifecycle-proof-postgresql
  lifecycle-proof-rabbitmq
  lifecycle-proof-etcd
  lifecycle-proof-valkey
  lifecycle-proof-minio
  lifecycle-proof-clickhouse
)

for phase in infrastructure migrating active suspended; do
  rendered="${temporary_dir}/${phase}.yaml"
  "${helm}" template lifecycle-proof "${chart_dir}" \
    --namespace m3-lifecycle-proof \
    --values "${chart_dir}/values-appimage.yaml" \
    --values "${chart_dir}/values-appimage-amd64.yaml" \
    --set "global.lifecycle.phase=${phase}" >"${rendered}"

  workload_replicas=0
  infrastructure_replicas=1
  if [[ "${phase}" == active ]]; then
    workload_replicas=1
  elif [[ "${phase}" == suspended ]]; then
    infrastructure_replicas=0
  fi

  for resource in "${workloads[@]}"; do
    test "$(replicas_for "${rendered}" "${resource}")" = "${workload_replicas}"
  done
  for resource in "${infrastructure[@]}"; do
    test "$(replicas_for "${rendered}" "${resource}")" = "${infrastructure_replicas}"
  done

  grep -q 'name: fix-erlang-cookie-permissions' "${rendered}"
  grep -q 'chmod 0600 /var/lib/rabbitmq/.erlang.cookie' "${rendered}"
  grep -q 'fsGroupChangePolicy: OnRootMismatch' "${rendered}"

  provisioning_jobs="$(grep -c '^kind: Job$' "${rendered}" || true)"
  if [[ "${phase}" == suspended ]]; then
    test "${provisioning_jobs}" -eq 0
  elif [[ "${phase}" == migrating ]]; then
    test "${provisioning_jobs}" -eq 2
  else
    test "${provisioning_jobs}" -eq 1
  fi
done

for phase in active suspended; do
  rendered="${temporary_dir}/autoscaling-${phase}.yaml"
  "${helm}" template autoscaling-proof "${chart_dir}" \
    --namespace m3-lifecycle-proof \
    --values "${chart_dir}/values-appimage.yaml" \
    --values "${chart_dir}/values-appimage-amd64.yaml" \
    --set "global.lifecycle.phase=${phase}" \
    --set server.autoscaling.enabled=true >"${rendered}"

  hpas="$(grep -c '^kind: HorizontalPodAutoscaler$' "${rendered}" || true)"
  if [[ "${phase}" == active ]]; then
    test "${hpas}" -eq 1
  else
    test "${hpas}" -eq 0
    test "$(replicas_for "${rendered}" autoscaling-proof-server)" = 0
  fi
done

if "${helm}" template invalid-lifecycle "${chart_dir}" \
  --values "${chart_dir}/values-appimage.yaml" \
  --values "${chart_dir}/values-appimage-amd64.yaml" \
  --set global.lifecycle.phase=invalid >/dev/null 2>&1; then
  echo "invalid lifecycle phase rendered successfully" >&2
  exit 1
fi

echo "Verified Marie lifecycle renders"
