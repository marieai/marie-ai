#!/bin/bash
# setup-local-k8s.sh
# Script to quickly set up a local Kubernetes cluster for Marie Operator development
#
# Usage:
#   ./setup-local-k8s.sh k3d       # Fastest startup
#   ./setup-local-k8s.sh kind      # Recommended for operator development
#   ./setup-local-k8s.sh minikube  # Most production-like

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-marie-dev}"
K8S_VERSION="${K8S_VERSION:-v1.29.0}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
KIND_VERSION="${KIND_VERSION:-v0.24.0}"
K3D_VERSION="${K3D_VERSION:-v5.7.5}"
KIND_WORKERS="${KIND_WORKERS:-2}"
K3D_AGENTS="${K3D_AGENTS:-2}"
EXPOSE_OPTIONAL_PORTS="${EXPOSE_OPTIONAL_PORTS:-true}"
HTTP_HOST_PORT="${HTTP_HOST_PORT:-8080}"
GRPC_HOST_PORT="${GRPC_HOST_PORT:-52000}"
METRICS_HOST_PORT="${METRICS_HOST_PORT:-9090}"
CLICKHOUSE_HTTP_HOST_PORT="${CLICKHOUSE_HTTP_HOST_PORT:-8123}"
CLICKHOUSE_NATIVE_HOST_PORT="${CLICKHOUSE_NATIVE_HOST_PORT:-9001}"
GITEA_HTTP_HOST_PORT="${GITEA_HTTP_HOST_PORT:-3001}"
GITEA_SSH_HOST_PORT="${GITEA_SSH_HOST_PORT:-2222}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    log_info "Docker is available"
}

check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl is not installed in PATH."
        log_info "Will use 'minikube kubectl' if using minikube."
        return 1
    fi
    log_info "kubectl is available"
    return 0
}

ensure_local_bin() {
    mkdir -p "${LOCAL_BIN}"
    case ":${PATH}:" in
        *":${LOCAL_BIN}:"*) ;;
        *)
            export PATH="${LOCAL_BIN}:${PATH}"
            log_warn "${LOCAL_BIN} was not in PATH for this shell; added it for this run"
            ;;
    esac
}

install_kind() {
    ensure_local_bin
    local os
    local arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    case "${arch}" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)
            log_error "Unsupported architecture for kind: ${arch}"
            exit 1
            ;;
    esac

    log_info "Installing kind ${KIND_VERSION} to ${LOCAL_BIN}/kind"
    curl -fsSL -o "${LOCAL_BIN}/kind" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${os}-${arch}"
    chmod +x "${LOCAL_BIN}/kind"
}

install_k3d() {
    ensure_local_bin
    local os
    local arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    case "${arch}" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)
            log_error "Unsupported architecture for k3d: ${arch}"
            exit 1
            ;;
    esac

    log_info "Installing k3d ${K3D_VERSION} to ${LOCAL_BIN}/k3d"
    curl -fsSL -o "${LOCAL_BIN}/k3d" "https://github.com/k3d-io/k3d/releases/download/${K3D_VERSION}/k3d-${os}-${arch}"
    chmod +x "${LOCAL_BIN}/k3d"
}

# Wrapper to use kubectl or minikube kubectl
run_kubectl() {
    if command -v kubectl &> /dev/null; then
        kubectl "$@"
    elif command -v minikube &> /dev/null; then
        minikube kubectl -- "$@"
    else
        log_error "Neither kubectl nor minikube found. Please install kubectl."
        exit 1
    fi
}

setup_kind() {
    check_docker
    if ! check_kubectl; then
        log_error "kubectl is required for kind. Please install kubectl first."
        exit 1
    fi

    if ! command -v kind &> /dev/null; then
        install_kind
    fi

    # Check if cluster already exists
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        log_warn "Cluster '${CLUSTER_NAME}' already exists"
        read -p "Delete and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kind delete cluster --name ${CLUSTER_NAME}
        else
            log_info "Using existing cluster"
            run_kubectl cluster-info --context kind-${CLUSTER_NAME}
            return
        fi
    fi

    log_info "Creating kind cluster '${CLUSTER_NAME}'..."

    local kind_config
    kind_config="$(mktemp)"

    cat > "${kind_config}" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${CLUSTER_NAME}
nodes:
  # Control plane node with port mappings
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      # HTTP ingress
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      # HTTPS ingress
      - containerPort: 443
        hostPort: 443
        protocol: TCP
      # Marie HTTP API (NodePort 30080 -> localhost:8080)
      - containerPort: 30080
        hostPort: ${HTTP_HOST_PORT}
        protocol: TCP
      # Marie gRPC API (NodePort 30052 -> localhost:52000)
      - containerPort: 30052
        hostPort: ${GRPC_HOST_PORT}
        protocol: TCP
EOF

    if [[ "${EXPOSE_OPTIONAL_PORTS}" == "true" ]]; then
        cat >> "${kind_config}" <<EOF
      # Metrics (NodePort 30090 -> localhost:9090)
      - containerPort: 30090
        hostPort: ${METRICS_HOST_PORT}
        protocol: TCP
      # ClickHouse HTTP API (NodePort 30123 -> localhost:8123)
      - containerPort: 30123
        hostPort: ${CLICKHOUSE_HTTP_HOST_PORT}
        protocol: TCP
      # ClickHouse Native (NodePort 30900 -> localhost:9001)
      - containerPort: 30900
        hostPort: ${CLICKHOUSE_NATIVE_HOST_PORT}
        protocol: TCP
      # Gitea Web UI (NodePort 30300 -> localhost:3001)
      - containerPort: 30300
        hostPort: ${GITEA_HTTP_HOST_PORT}
        protocol: TCP
      # Gitea SSH (NodePort 30222 -> localhost:2222)
      - containerPort: 30222
        hostPort: ${GITEA_SSH_HOST_PORT}
        protocol: TCP
EOF
    fi

    if [[ "${KIND_WORKERS}" -gt 0 ]]; then
        echo "  # Worker nodes for executor pools" >> "${kind_config}"
        for ((i = 0; i < KIND_WORKERS; i++)); do
            echo "  - role: worker" >> "${kind_config}"
        done
    fi

    kind create cluster --image "kindest/node:${K8S_VERSION}" --config="${kind_config}"
    rm -f "${kind_config}"

    log_info "Kind cluster '${CLUSTER_NAME}' created successfully!"
}

setup_k3d() {
    check_docker
    if ! check_kubectl; then
        log_error "kubectl is required for k3d. Please install kubectl first."
        exit 1
    fi

    if ! command -v k3d &> /dev/null; then
        install_k3d
    fi

    # Check if cluster already exists
    if k3d cluster list 2>/dev/null | grep -q "${CLUSTER_NAME}"; then
        log_warn "Cluster '${CLUSTER_NAME}' already exists"
        read -p "Delete and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            k3d cluster delete ${CLUSTER_NAME}
        else
            log_info "Using existing cluster"
            run_kubectl cluster-info
            return
        fi
    fi

    log_info "Creating k3d cluster '${CLUSTER_NAME}'..."

    local k3d_args=(
        cluster create "${CLUSTER_NAME}"
        --servers 1
        --agents "${K3D_AGENTS}"
        --image "rancher/k3s:${K8S_VERSION}-k3s1"
        --port "${HTTP_HOST_PORT}:30080@server:0"
        --port "${GRPC_HOST_PORT}:30052@server:0"
        --k3s-arg "--disable=traefik@server:0"
        --wait
    )

    if [[ "${EXPOSE_OPTIONAL_PORTS}" == "true" ]]; then
        k3d_args+=(
            --port "${METRICS_HOST_PORT}:30090@server:0"
            --port "${CLICKHOUSE_HTTP_HOST_PORT}:30123@server:0"
            --port "${CLICKHOUSE_NATIVE_HOST_PORT}:30900@server:0"
            --port "${GITEA_HTTP_HOST_PORT}:30300@server:0"
            --port "${GITEA_SSH_HOST_PORT}:30222@server:0"
        )
    fi

    k3d "${k3d_args[@]}"

    log_info "k3d cluster '${CLUSTER_NAME}' created successfully!"
}

setup_minikube() {
    check_docker
    # minikube has built-in kubectl, so we just warn if kubectl is not in PATH
    check_kubectl || log_info "Using minikube's built-in kubectl"

    if ! command -v minikube &> /dev/null; then
        log_info "Installing minikube..."
        curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
        sudo install minikube-linux-amd64 /usr/local/bin/minikube
        rm minikube-linux-amd64
    fi

    # Check if profile already exists
    if minikube profile list 2>/dev/null | grep -q "${CLUSTER_NAME}"; then
        log_warn "Profile '${CLUSTER_NAME}' already exists"
        read -p "Delete and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            minikube delete -p ${CLUSTER_NAME}
        else
            log_info "Using existing profile"
            minikube profile ${CLUSTER_NAME}
            run_kubectl cluster-info
            return
        fi
    fi

    log_info "Creating minikube cluster '${CLUSTER_NAME}'..."

    minikube start \
        --cpus=4 \
        --memory=8192 \
        --disk-size=50g \
        --driver=docker \
        --kubernetes-version=${K8S_VERSION} \
        --profile=${CLUSTER_NAME}

    # Enable useful addons
    log_info "Enabling minikube addons..."
    minikube addons enable metrics-server -p ${CLUSTER_NAME}
    minikube addons enable ingress -p ${CLUSTER_NAME}

    # Set the profile as active so minikube commands work without -p flag
    minikube profile ${CLUSTER_NAME}

    log_info "Minikube cluster '${CLUSTER_NAME}' created successfully!"
    log_info "Profile '${CLUSTER_NAME}' is now active."
    log_info ""
    log_info "Useful commands:"
    log_info "  minikube dashboard           # Open Kubernetes dashboard"
    log_info "  minikube tunnel              # Expose LoadBalancer services"
    log_info "  minikube service <name>      # Get URL for a service"
    log_info "  minikube ssh                 # SSH into the node"
}

delete_cluster() {
    case "$1" in
        kind)
            if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
                log_info "Deleting kind cluster '${CLUSTER_NAME}'..."
                kind delete cluster --name ${CLUSTER_NAME}
            else
                log_warn "Kind cluster '${CLUSTER_NAME}' does not exist"
            fi
            ;;
        k3d)
            if k3d cluster list 2>/dev/null | grep -q "${CLUSTER_NAME}"; then
                log_info "Deleting k3d cluster '${CLUSTER_NAME}'..."
                k3d cluster delete ${CLUSTER_NAME}
            else
                log_warn "k3d cluster '${CLUSTER_NAME}' does not exist"
            fi
            ;;
        minikube)
            if minikube profile list 2>/dev/null | grep -q "${CLUSTER_NAME}"; then
                log_info "Deleting minikube profile '${CLUSTER_NAME}'..."
                minikube delete -p ${CLUSTER_NAME}
            else
                log_warn "Minikube profile '${CLUSTER_NAME}' does not exist"
            fi
            ;;
        *)
            log_error "Unknown cluster type: $1"
            exit 1
            ;;
    esac
}

print_usage() {
    echo "Marie Operator - Local Kubernetes Setup Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  k3d        Create a k3d cluster (fastest startup)"
    echo "  kind       Create a kind cluster (recommended for operator development)"
    echo "  minikube   Create a minikube cluster (most production-like)"
    echo "  delete     Delete the cluster (requires cluster type as argument)"
    echo "  status     Show cluster status"
    echo ""
    echo "Options:"
    echo "  CLUSTER_NAME=<name>   Set cluster name (default: marie-dev)"
    echo "  K8S_VERSION=<version> Set Kubernetes version (default: v1.29.0)"
    echo "  LOCAL_BIN=<path>      Install kind/k3d here if missing (default: ~/.local/bin)"
    echo "  KIND_VERSION=<ver>    kind version to install if missing (default: v0.24.0)"
    echo "  K3D_VERSION=<ver>     k3d version to install if missing (default: v5.7.5)"
    echo "  EXPOSE_OPTIONAL_PORTS=true|false"
    echo "  HTTP_HOST_PORT=<port> Marie HTTP host port (default: 8080)"
    echo "  GRPC_HOST_PORT=<port> Marie gRPC host port (default: 52000)"
    echo ""
    echo "Examples:"
    echo "  $0 kind                           # Create kind cluster"
    echo "  $0 k3d                            # Create k3d cluster"
    echo "  $0 minikube                       # Create minikube cluster"
    echo "  $0 delete kind                    # Delete kind cluster"
    echo "  CLUSTER_NAME=test $0 kind         # Create cluster named 'test'"
    echo ""
}

show_status() {
    echo "=== Local Kubernetes Cluster Status ==="
    echo ""

    echo "Kind clusters:"
    if command -v kind &> /dev/null; then
        kind get clusters 2>/dev/null || echo "  (none)"
    else
        echo "  (kind not installed)"
    fi
    echo ""

    echo "k3d clusters:"
    if command -v k3d &> /dev/null; then
        k3d cluster list 2>/dev/null || echo "  (none)"
    else
        echo "  (k3d not installed)"
    fi
    echo ""

    echo "Minikube profiles:"
    if command -v minikube &> /dev/null; then
        minikube profile list 2>/dev/null || echo "  (none)"
    else
        echo "  (minikube not installed)"
    fi
    echo ""

    echo "Current kubectl context:"
    run_kubectl config current-context 2>/dev/null || echo "  (none)"
}

# Main script
case "$1" in
    kind)
        setup_kind
        ;;
    k3d)
        setup_k3d
        ;;
    minikube)
        setup_minikube
        ;;
    delete)
        if [ -z "$2" ]; then
            log_error "Please specify cluster type: kind, k3d, or minikube"
            exit 1
        fi
        delete_cluster "$2"
        exit 0
        ;;
    status)
        show_status
        exit 0
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

# Verify cluster is ready
echo ""
log_info "Verifying cluster..."
run_kubectl cluster-info
echo ""
run_kubectl get nodes -o wide

echo ""
log_info "Cluster '${CLUSTER_NAME}' is ready for Marie Operator development!"
echo ""
echo "Next steps:"
echo "  1. cd deploy/operator"
echo "  2. Build the operator:     make build"
echo "  3. Install CRDs:           make install"
echo "  4. Run operator locally:   make run"
echo "  5. Or deploy to cluster:   make deploy IMG=marieai/marie-operator:dev"
echo ""
echo "Or use Helm for full stack deployment:"
echo "  cd deploy/helm/charts/marie"
echo "  helm dependency update"
echo "  helm install marie . -f values-local.yaml -n marie --create-namespace"
echo ""
echo "Service Port Mappings:"
echo "  Marie HTTP API:      http://localhost:${HTTP_HOST_PORT}"
echo "  Marie gRPC API:      localhost:${GRPC_HOST_PORT}"
echo "  Prometheus Metrics:  http://localhost:${METRICS_HOST_PORT}"
echo "  ClickHouse HTTP:     http://localhost:${CLICKHOUSE_HTTP_HOST_PORT}"
echo "  ClickHouse Native:   localhost:${CLICKHOUSE_NATIVE_HOST_PORT}"
echo "  Gitea Web UI:        http://localhost:${GITEA_HTTP_HOST_PORT}"
echo "  Gitea SSH:           localhost:${GITEA_SSH_HOST_PORT}"
echo ""
if ! command -v kubectl &> /dev/null; then
    echo "Note: kubectl is not in your PATH. You can use 'minikube kubectl --' instead."
    echo "  Example: minikube kubectl -- get pods"
    echo ""
fi
