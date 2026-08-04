#!/usr/bin/env bash

set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${chart_dir}/../../../.." && pwd)"
helm="${HELM:-helm}"

verify_scheduler_sql() {
  local source_dir packaged_dir expected actual path
  source_dir="${repo_dir}/config/psql"
  packaged_dir="${chart_dir}/charts/server/files/psql"
  expected="$(mktemp)"
  actual="$(mktemp)"

  {
    printf '%s\n' cron_job_init.sql
    find "${source_dir}/schema" -maxdepth 1 -type f -name '[0-9][0-9][0-9]_*.sql' -printf 'schema/%f\n'
    find "${source_dir}/schema/lease" -maxdepth 1 -type f -name '*.sql' -printf 'schema/lease/%f\n'
    printf '%s\n' schema/monitoring/throughput_analysis.sql
  } | sort >"${expected}"
  find "${packaged_dir}" -type f -printf '%P\n' | sort >"${actual}"

  diff -u "${expected}" "${actual}"
  while IFS= read -r path; do
    cmp "${source_dir}/${path}" "${packaged_dir}/${path}"
  done <"${expected}"

  rm -f "${expected}" "${actual}"
}

verify_platform() {
  local platform="$1"
  local rendered actual expected
  rendered="$(mktemp)"
  actual="$(mktemp)"
  expected="$(mktemp)"

  "$helm" lint "$chart_dir" \
    --values "$chart_dir/values-appimage.yaml" \
    --values "$chart_dir/values-appimage-${platform}.yaml"
  "$helm" template "appimage-${platform}" "$chart_dir" --namespace m3-default \
    --values "$chart_dir/values-appimage.yaml" \
    --values "$chart_dir/values-appimage-${platform}.yaml" >"$rendered"

  awk '$1 == "image:" { gsub(/"/, "", $2); print $2 }' "$rendered" | sort -u >"$actual"
  cat >"$expected" <<'EOF'
docker.io/clickhouse/clickhouse-server@sha256:c67cd26ea87301f3115e5fa7822905bcbb89cbd81e52bdd1ab7a938d1d5b77d8
docker.io/gitea/gitea@sha256:fd917399b5bbde18348d52eda18b3690d75ae1c108630c6dc3a2bf10a3e0c353
docker.io/library/rabbitmq@sha256:3c498e636fd64462480c5f9ff842eb224ab84160a8ada1ded5375e9569e9230c
docker.io/marieai/marie-gateway@sha256:e7d59c0591ed6a6ed3ddf4233dafca6ca740fc1565cdf6c80b916a734b68cf72
docker.io/marieai/marie@sha256:26342bd0c1db5e94f1da8d611aee3a7f3bc13aceb41dfd616e7962f1dab915a0
docker.io/minio/mc@sha256:eb4ea9884b77704230e2423e9004d2fa738dc272876b9cc41a297d29443b8780
docker.io/minio/minio@sha256:a1a8bd4ac40ad7881a245bab97323e18f971e4d4cba2c2007ec1bedd21cbaba2
docker.io/valkey/valkey@sha256:3fe38a705227d29534a199e876b38d5474dec4d3baca980ac6894df539416562
ghcr.io/ferretdb/postgres-documentdb@sha256:c2bd151a4ba2227d2f2a4c50406a104def04aef8780495505e6e1fcf6b8b2d8e
quay.io/coreos/etcd@sha256:893a99e64e181fede58348cc824cdfae956ea8b64e0e008f7105e950d9cb3f33
EOF

  diff -u "$expected" "$actual"
  if grep -Eq 'imagePullPolicy: (Always|Never)' "$rendered"; then
    echo "Online AppImage dependencies must pull once and reuse the local cache" >&2
    exit 1
  fi
  if [[ "$(grep -c '^  progressDeadlineSeconds: 1800$' "$rendered" || true)" -ne 4 ]]; then
    echo "AppImage Deployments must allow the pinned images 30 minutes to download and start" >&2
    exit 1
  fi
  if ! awk '
    $1 == "-" && $2 == "name:" && $3 == "gitea" { in_gitea = 1 }
    in_gitea && $1 == "runAsUser:" && $2 == "0" { found = 1; exit }
    END { exit !found }
  ' "$rendered"; then
    echo "The standard Gitea image must initialize its data directory as root" >&2
    exit 1
  fi
  if ! awk '
    $1 == "gpu_monitor_enabled:" && $2 == "false" { found = 1; exit }
    END { exit !found }
  ' "$rendered"; then
    echo "AppImage CPU executor must disable GPU health monitoring" >&2
    exit 1
  fi
  if ! grep -q '^            - name: MARIE_PORT$' "$rendered" \
    || ! grep -q '^              value: "52010"$' "$rendered" \
    || ! grep -q 'port:.*ENV.MARIE_PORT' "$rendered"; then
    echo "Executor runtime and Service port are not wired to the same value" >&2
    exit 1
  fi
  grpc_probes="$(grep -c '^            grpc:$' "$rendered" || true)"
  if [[ "$grpc_probes" -lt 3 ]]; then
    echo "Executor must use native gRPC health probes on its gRPC port" >&2
    exit 1
  fi
  if ! awk '
    $1 == "llm_tracking:" { in_tracking = 1; next }
    in_tracking && $1 == "enabled:" { if ($2 == "false") disabled += 1; in_tracking = 0 }
    END { exit disabled < 2 }
  ' "$rendered"; then
    echo "Pinned gateway and executor images require optional LLM media tracking to be disabled" >&2
    exit 1
  fi
  if ! grep -q '^                name: appimage-amd64-config$' "$rendered"; then
    echo "Executor must reference the parent Marie ConfigMap under an umbrella release" >&2
    exit 1
  fi
  if ! grep -q '^              mountPath: /marie/config/psql$' "$rendered" \
    || ! grep -q '^                path: schema/monitoring/throughput_analysis.sql$' "$rendered" \
    || ! grep -q '^                path: cron_job_init.sql$' "$rendered"; then
    echo "Gateway scheduler SQL fixtures are not mounted at their runtime paths" >&2
    exit 1
  fi
  rm -f "$rendered" "$actual" "$expected"
}

verify_scheduler_sql
verify_platform amd64
