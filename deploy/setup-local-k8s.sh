#!/bin/bash
# setup-local-k8s.sh
# Script to quickly set up a local Kubernetes cluster for Marie Operator development
#
# Usage:
#   ./setup-local-k8s.sh kind      # Recommended for operator development
#   ./setup-local-k8s.sh k3d       # Fastest startup
#   ./setup-local-k8s.sh minikube  # Most production-like

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-marie-dev}"
K8S_VERSION="${K8S_VERSION:-v1.29.0}"

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
        log_info "Installing kind..."
        curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
        chmod +x ./kind
        sudo mv ./kind /usr/local/bin/kind
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

    cat <<EOF | kind create cluster --config=-
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
        hostPort: 8080
        protocol: TCP
      # Marie gRPC API (NodePort 30052 -> localhost:52000)
      - containerPort: 30052
        hostPort: 52000
        protocol: TCP
      # Metrics (NodePort 30090 -> localhost:9090)
      - containerPort: 30090
        hostPort: 9090
        protocol: TCP
      # ClickHouse HTTP API (NodePort 30123 -> localhost:8123)
      - containerPort: 30123
        hostPort: 8123
        protocol: TCP
      # ClickHouse Native (NodePort 30900 -> localhost:9001)
      - containerPort: 30900
        hostPort: 9001
        protocol: TCP
      # Gitea Web UI (NodePort 30300 -> localhost:3001)
      - containerPort: 30300
        hostPort: 3001
        protocol: TCP
      # Gitea SSH (NodePort 30222 -> localhost:2222)
      - containerPort: 30222
        hostPort: 2222
        protocol: TCP
  # Worker nodes for executor pools
  - role: worker
  - role: worker
EOF

    log_info "Kind cluster '${CLUSTER_NAME}' created successfully!"
}

setup_k3d() {
    check_docker
    if ! check_kubectl; then
        log_error "kubectl is required for k3d. Please install kubectl first."
        exit 1
    fi

    if ! command -v k3d &> /dev/null; then
        log_info "Installing k3d..."
        curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
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

    k3d cluster create ${CLUSTER_NAME} \
        --servers 1 \
        --agents 2 \
        --port "8080:30080@server:0" \
        --port "52000:30052@server:0" \
        --port "9090:30090@server:0" \
        --port "8123:30123@server:0" \
        --port "9001:30900@server:0" \
        --port "3001:30300@server:0" \
        --port "2222:30222@server:0" \
        --k3s-arg "--disable=traefik@server:0" \
        --wait

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
    echo "  kind       Create a kind cluster (recommended for operator development)"
    echo "  k3d        Create a k3d cluster (fastest startup)"
    echo "  minikube   Create a minikube cluster (most production-like)"
    echo "  delete     Delete the cluster (requires cluster type as argument)"
    echo "  status     Show cluster status"
    echo ""
    echo "Options:"
    echo "  CLUSTER_NAME=<name>   Set cluster name (default: marie-dev)"
    echo "  K8S_VERSION=<version> Set Kubernetes version (default: v1.29.0)"
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
echo "  Marie HTTP API:      http://localhost:8080"
echo "  Marie gRPC API:      localhost:52000"
echo "  Prometheus Metrics:  http://localhost:9090"
echo "  ClickHouse HTTP:     http://localhost:8123"
echo "  ClickHouse Native:   localhost:9001"
echo "  Gitea Web UI:        http://localhost:3001"
echo "  Gitea SSH:           localhost:2222"
echo ""
if ! command -v kubectl &> /dev/null; then
    echo "Note: kubectl is not in your PATH. You can use 'minikube kubectl --' instead."
    echo "  Example: minikube kubectl -- get pods"
    echo ""
fi
