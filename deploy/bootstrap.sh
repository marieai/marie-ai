#!/bin/bash
# Bootstrap Marie-AI on a local Kubernetes cluster using the Helm chart.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $0 [k3d|kind]

Defaults:
  provider      k3d
  cluster       marie-helm-smoke
  namespace     marie
  release       marie

Environment overrides:
  PROVIDER, CLUSTER_NAME, NAMESPACE, RELEASE
  MARIE_IMAGE, GATEWAY_IMAGE
  K3D_AGENTS, KIND_WORKERS
  LOAD_LOCAL_IMAGES, SKIP_OPTIONAL, WAIT_TIMEOUT
EOF
}

case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

PROVIDER="${1:-${PROVIDER:-k3d}}"

export CLUSTER_NAME="${CLUSTER_NAME:-marie-helm-smoke}"
export NAMESPACE="${NAMESPACE:-marie}"
export RELEASE="${RELEASE:-marie}"
export K3D_AGENTS="${K3D_AGENTS:-2}"
export KIND_WORKERS="${KIND_WORKERS:-2}"

exec "${SCRIPT_DIR}/smoke-marie-helm.sh" "${PROVIDER}"
