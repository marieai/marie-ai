#!/usr/bin/env bash
set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
helm="${HELM:-helm}"
rendered="$(mktemp)"
trap 'rm -f "${rendered}"' EXIT

"${helm}" template gpu-proof "${chart_dir}" \
  --namespace m3-gpu-proof \
  --values "${chart_dir}/values-appimage.yaml" \
  --values "${chart_dir}/values-appimage-amd64.yaml" \
  --values "${chart_dir}/tests/values-gpu-executor.yaml" >"${rendered}"

grep -q '^  name: gpu-proof-executor-gpu-local$' "${rendered}"
grep -q '^      runtimeClassName: "nvidia"$' "${rendered}"
grep -q '^        marie.ai/gpu: "true"$' "${rendered}"
test "$(grep -c '^              nvidia.com/gpu: 1$' "${rendered}")" -eq 2
grep -q '^          key: nvidia.com/gpu$' "${rendered}"

echo "Verified NVIDIA executor scheduling render"
