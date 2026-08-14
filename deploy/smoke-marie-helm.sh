#!/bin/bash
# Smoke test the Marie-AI Helm chart on a local kind or k3d cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROVIDER="${1:-${PROVIDER:-k3d}}"
CLUSTER_NAME="${CLUSTER_NAME:-marie-helm-smoke}"
NAMESPACE="${NAMESPACE:-marie}"
RELEASE="${RELEASE:-marie}"
MARIE_IMAGE="${MARIE_IMAGE:-marieai/marie:5.0.3-cuda}"
GATEWAY_IMAGE="${GATEWAY_IMAGE:-marieai/marie-gateway:5.0.3-cpu}"
LOAD_LOCAL_IMAGES="${LOAD_LOCAL_IMAGES:-true}"
SKIP_OPTIONAL="${SKIP_OPTIONAL:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-20m}"
K3D_IMAGE_IMPORT_MODE="${K3D_IMAGE_IMPORT_MODE:-direct}"
K3D_AGENTS="${K3D_AGENTS:-2}"
KIND_WORKERS="${KIND_WORKERS:-2}"
# Optional: install Argo CD (control plane for the sandbox/snapshot feature). Off by default.
INSTALL_ARGOCD="${INSTALL_ARGOCD:-false}"
ARGOCD_NAMESPACE="${ARGOCD_NAMESPACE:-argocd}"
ARGOCD_VERSION="${ARGOCD_VERSION:-stable}"

usage() {
    cat <<EOF
Usage: $0 [k3d|kind]

Environment:
  CLUSTER_NAME       Local cluster name (default: marie-helm-smoke)
  NAMESPACE          Kubernetes namespace (default: marie)
  RELEASE            Helm release name (default: marie)
  MARIE_IMAGE        Executor image (default: marieai/marie:5.0.3-cuda)
  GATEWAY_IMAGE      Gateway/server image (default: marieai/marie-gateway:5.0.3-cpu)
  LOAD_LOCAL_IMAGES  Load local Docker images into the cluster if present (default: true)
  SKIP_OPTIONAL      Disable ClickHouse for smoke speed (default: false)
  WAIT_TIMEOUT       Helm/kubectl wait timeout (default: 20m)
  K3D_IMAGE_IMPORT_MODE k3d image import mode (default: direct)
  K3D_AGENTS         k3d agent count for new smoke clusters (default: 2)
  KIND_WORKERS       kind worker count for new smoke clusters (default: 2)
  INSTALL_ARGOCD     Install Argo CD (sandbox/snapshot control plane) into the cluster (default: false)
  ARGOCD_NAMESPACE   Namespace for Argo CD (default: argocd)
  ARGOCD_VERSION     Argo CD install manifest channel/tag, e.g. stable or v2.13.0 (default: stable)
  HTTP_HOST_PORT     Marie HTTP host port passed to setup-local-k8s.sh
  GRPC_HOST_PORT     Marie gRPC host port passed to setup-local-k8s.sh
EOF
}

log() {
    echo "[smoke] $*"
}

image_repo() {
    echo "${1%:*}"
}

image_tag() {
    echo "${1##*:}"
}

load_image() {
    local image="$1"

    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        log "Local image ${image} not found; cluster will pull it if needed"
        return
    fi

    case "${PROVIDER}" in
        kind)
            kind load docker-image "${image}" --name "${CLUSTER_NAME}"
            ;;
        k3d)
            k3d image import -m "${K3D_IMAGE_IMPORT_MODE}" "${image}" -c "${CLUSTER_NAME}"
            ;;
    esac
}

check_executor_image() {
    if ! docker image inspect "${MARIE_IMAGE}" >/dev/null 2>&1; then
        log "Local image ${MARIE_IMAGE} not found; skipping local pkg_resources check"
        return
    fi

    if ! docker run --rm --entrypoint python "${MARIE_IMAGE}" -c "import pkg_resources" >/dev/null 2>&1; then
        echo "Executor image ${MARIE_IMAGE} is missing pkg_resources." >&2
        echo "Rebuild/publish the image with setuptools<81 before running the Helm smoke test." >&2
        exit 1
    fi
}

install_argocd() {
    # Argo CD is the control plane for the Marie sandbox/snapshot feature.
    # Platform-level, installed once per cluster; not part of the Marie Helm release.
    # Ref: https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/
    log "Installing Argo CD (${ARGOCD_VERSION}) into namespace ${ARGOCD_NAMESPACE}"
    kubectl create namespace "${ARGOCD_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -n "${ARGOCD_NAMESPACE}" \
        -f "https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml"
    kubectl -n "${ARGOCD_NAMESPACE}" rollout status deploy/argocd-server --timeout="${WAIT_TIMEOUT}"
    log "Argo CD ready. API/UI: kubectl -n ${ARGOCD_NAMESPACE} port-forward svc/argocd-server 8080:443"
}

cluster_exists() {
    case "${PROVIDER}" in
        kind)
            kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"
            ;;
        k3d)
            k3d cluster list 2>/dev/null | grep -q "^${CLUSTER_NAME}[[:space:]]"
            ;;
    esac
}

case "${PROVIDER}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    kind|k3d)
        ;;
    *)
        echo "Unsupported provider: ${PROVIDER}" >&2
        usage
        exit 1
        ;;
esac

check_executor_image

if cluster_exists; then
    log "Using existing ${PROVIDER} cluster ${CLUSTER_NAME}"
    kubectl cluster-info
else
    log "Ensuring ${PROVIDER} cluster ${CLUSTER_NAME}"
    CLUSTER_NAME="${CLUSTER_NAME}" EXPOSE_OPTIONAL_PORTS=false K3D_AGENTS="${K3D_AGENTS}" KIND_WORKERS="${KIND_WORKERS}" "${SCRIPT_DIR}/setup-local-k8s.sh" "${PROVIDER}"
fi

if [[ "${LOAD_LOCAL_IMAGES}" == "true" ]]; then
    log "Loading local Marie images when available"
    load_image "${MARIE_IMAGE}"
    load_image "${GATEWAY_IMAGE}"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

chart_dir="${tmp_dir}/marie"
cp -a "${PROJECT_ROOT}/deploy/helm/charts/marie" "${chart_dir}"

log "Building Helm dependencies"
helm dependency build "${chart_dir}"

helm_args=(
    upgrade --install "${RELEASE}" "${chart_dir}"
    -n "${NAMESPACE}" --create-namespace
    -f "${chart_dir}/values-local.yaml"
    --set "global.marie.image.repository=$(image_repo "${MARIE_IMAGE}")"
    --set "global.marie.image.tag=$(image_tag "${MARIE_IMAGE}")"
    --set "marie.image.repository=$(image_repo "${MARIE_IMAGE}")"
    --set "marie.image.tag=$(image_tag "${MARIE_IMAGE}")"
    --set "server.image.repository=$(image_repo "${GATEWAY_IMAGE}")"
    --set "server.image.tag=$(image_tag "${GATEWAY_IMAGE}")"
    --set gitea.enabled=false
    --wait --timeout "${WAIT_TIMEOUT}"
)

if [[ "${SKIP_OPTIONAL}" == "true" ]]; then
    helm_args+=(--set clickhouse.enabled=false)
fi

log "Installing Marie-AI Helm release ${RELEASE} in namespace ${NAMESPACE}"
helm "${helm_args[@]}"

if [[ "${INSTALL_ARGOCD}" == "true" ]]; then
    install_argocd
fi

log "Waiting for core workloads"
kubectl -n "${NAMESPACE}" rollout status "deploy/${RELEASE}-server" --timeout="${WAIT_TIMEOUT}"
kubectl -n "${NAMESPACE}" rollout status "deploy/${RELEASE}-executor-cpu-local" --timeout="${WAIT_TIMEOUT}"
kubectl -n "${NAMESPACE}" rollout status "statefulset/${RELEASE}-valkey" --timeout="${WAIT_TIMEOUT}"

log "Current pods"
kubectl -n "${NAMESPACE}" get pods -o wide

log "Marie-AI Helm smoke test completed"
log "For API health: kubectl -n ${NAMESPACE} port-forward svc/${RELEASE}-server 51000:51000"
