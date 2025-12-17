---
sidebar_position: 3
---

# Kubernetes Deployment

Deploy Marie-AI on Kubernetes using the Marie Operator for production-grade cluster management.

## Overview

The Marie Operator provides Kubernetes-native management of Marie-AI clusters with:
- **MarieCluster CRD**: Manages server and executor pools
- **MarieJob CRD**: Submits and tracks document processing jobs
- **GPU Support**: Auto-scaling GPU executor pools with scale-to-zero
- **Multi-pool Executors**: Different executor types (OCR, NER, Classifier)

## Local Development Setup

Before deploying to production, set up a local Kubernetes environment for testing the Marie Operator.

### Choosing a Local Kubernetes Distribution

| Tool | Startup Time | Memory | Best For | Recommendation |
|------|-------------|--------|----------|----------------|
| **kind** | ~20 seconds | ~500MB | Operator development, CI/CD | **Recommended for Marie Operator development** |
| **k3d** | ~5 seconds | ~400MB | Lightweight testing, edge scenarios | Good for quick iterations |
| **Minikube** | ~60 seconds | ~2GB | Full Kubernetes simulation, GPU testing | Best for production-like testing |
| **Docker Desktop** | N/A | ~2GB | macOS/Windows users | Convenient but resource-heavy |

For Marie Operator development, we recommend **kind** because:
- Fast cluster creation (~20 seconds)
- Multi-node cluster support for testing
- Used by Kubernetes itself for testing
- Easy image loading without registry setup
- Works well with kubebuilder-based operators

### Prerequisites

Ensure you have the following installed:

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| Docker | 20.10+ | Container runtime |
| Go | 1.22+ | Operator development |
| kubectl | 1.25+ | Kubernetes CLI |
| kind/minikube/k3d | Latest | Local Kubernetes |
| kustomize | 5.0+ | Manifest management |

#### 1. Install Docker

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker version
```

#### 2. Install kubectl

```bash
# Linux (x86_64)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# macOS (using Homebrew)
brew install kubectl

# Verify installation
kubectl version --client
```

#### 3. Install Go (1.22+)

```bash
# Download and install Go 1.22
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Verify
go version
```

#### 4. Install kustomize

```bash
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify
kustomize version
```

#### 5. Install Helm (Optional but recommended)

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

---

### Option A: kind (Recommended for Operator Development)

[kind](https://kind.sigs.k8s.io/) runs Kubernetes inside Docker containers, making it ideal for testing operators.

#### Install kind

```bash
# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# macOS
brew install kind

# Verify
kind version
```

#### Create a Development Cluster

Create a configuration file `kind-config.yaml` for Marie Operator development:

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: marie-dev
nodes:
  # Control plane node
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
      # Marie HTTP API
      - containerPort: 30080
        hostPort: 8080
        protocol: TCP
      # Marie gRPC API
      - containerPort: 30052
        hostPort: 52000
        protocol: TCP
      # Metrics
      - containerPort: 30090
        hostPort: 9090
        protocol: TCP
  # Worker nodes for executor pools
  - role: worker
  - role: worker
# Use a specific Kubernetes version
# kubernetesVersion: v1.29.0
```

Create the cluster:

```bash
# Create cluster from config
kind create cluster --config kind-config.yaml

# Verify cluster is running
kubectl cluster-info --context kind-marie-dev
kubectl get nodes

# Expected output:
# NAME                      STATUS   ROLES           AGE   VERSION
# marie-dev-control-plane   Ready    control-plane   60s   v1.29.0
# marie-dev-worker          Ready    <none>          30s   v1.29.0
# marie-dev-worker2         Ready    <none>          30s   v1.29.0
```

#### Load Local Images into kind

When developing the operator, load your local images:

```bash
# Build operator image
cd deploy/operator
make docker-build IMG=marieai/marie-operator:dev

# Load into kind cluster
kind load docker-image marieai/marie-operator:dev --name marie-dev

# Load Marie runtime images (if building locally)
kind load docker-image marieai/marie:3.0 --name marie-dev
kind load docker-image marieai/marie:3.0-cuda --name marie-dev
```

#### Install NGINX Ingress Controller (Optional)

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for ingress controller to be ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
```

#### kind Cluster Management Commands

```bash
# List clusters
kind get clusters

# Delete cluster
kind delete cluster --name marie-dev

# Get kubeconfig
kind get kubeconfig --name marie-dev > ~/.kube/marie-dev-config

# Export logs for debugging
kind export logs --name marie-dev ./kind-logs
```

---

### Option B: k3d (Fastest Startup)

[k3d](https://k3d.io/) runs k3s (lightweight Kubernetes) in Docker, offering the fastest startup times.

#### Install k3d

```bash
# Linux/macOS
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Or using wget
wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Verify
k3d version
```

#### Create a Development Cluster

Create a configuration file `k3d-config.yaml`:

```yaml
# k3d-config.yaml
apiVersion: k3d.io/v1alpha5
kind: Simple
metadata:
  name: marie-dev
servers: 1
agents: 2
image: rancher/k3s:v1.29.0-k3s1
ports:
  - port: 8080:80
    nodeFilters:
      - loadbalancer
  - port: 8443:443
    nodeFilters:
      - loadbalancer
  - port: 52000:30052
    nodeFilters:
      - loadbalancer
options:
  k3d:
    wait: true
    timeout: "60s"
  k3s:
    extraArgs:
      - arg: --disable=traefik
        nodeFilters:
          - server:*
  kubeconfig:
    updateDefaultKubeconfig: true
    switchCurrentContext: true
registries:
  create:
    name: marie-registry
    host: "0.0.0.0"
    hostPort: "5000"
```

Create the cluster:

```bash
# Create cluster from config
k3d cluster create --config k3d-config.yaml

# Or create with CLI options
k3d cluster create marie-dev \
  --servers 1 \
  --agents 2 \
  --port "8080:80@loadbalancer" \
  --port "52000:30052@loadbalancer" \
  --registry-create marie-registry:0.0.0.0:5000

# Verify
kubectl get nodes
```

#### Load Images into k3d

```bash
# Import local images
k3d image import marieai/marie-operator:dev -c marie-dev
k3d image import marieai/marie:3.0 -c marie-dev

# Or push to local registry
docker tag marieai/marie-operator:dev localhost:5000/marie-operator:dev
docker push localhost:5000/marie-operator:dev
```

#### k3d Cluster Management Commands

```bash
# List clusters
k3d cluster list

# Stop cluster (preserves state)
k3d cluster stop marie-dev

# Start stopped cluster
k3d cluster start marie-dev

# Delete cluster
k3d cluster delete marie-dev
```

---

### Option C: Minikube (Production-like Environment)

[Minikube](https://minikube.sigs.k8s.io/) provides the most production-like local environment, including GPU support.

#### Install Minikube

```bash
# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# macOS
brew install minikube

# Verify
minikube version
```

#### Create a Development Cluster

```bash
# Start cluster with sufficient resources for Marie-AI
minikube start \
  --cpus=4 \
  --memory=8192 \
  --disk-size=50g \
  --driver=docker \
  --kubernetes-version=v1.29.0 \
  --profile=marie-dev

# Enable useful addons
minikube addons enable metrics-server -p marie-dev
minikube addons enable ingress -p marie-dev
minikube addons enable dashboard -p marie-dev

# For GPU support (requires NVIDIA drivers and nvidia-docker)
minikube start \
  --driver=docker \
  --gpus all \
  --profile=marie-gpu
```

#### Load Images into Minikube

```bash
# Option 1: Load directly
minikube image load marieai/marie-operator:dev -p marie-dev
minikube image load marieai/marie:3.0 -p marie-dev

# Option 2: Use minikube's Docker daemon
eval $(minikube docker-env -p marie-dev)
docker build -t marieai/marie-operator:dev .

# Option 3: Enable registry addon
minikube addons enable registry -p marie-dev
```

#### Access Services

```bash
# Get service URL
minikube service marie-production-server-svc -p marie-dev --url

# Or use tunnel for LoadBalancer services
minikube tunnel -p marie-dev

# Access dashboard
minikube dashboard -p marie-dev
```

#### Minikube Cluster Management Commands

```bash
# List profiles (clusters)
minikube profile list

# Switch profile
minikube profile marie-dev

# Stop cluster
minikube stop -p marie-dev

# Delete cluster
minikube delete -p marie-dev

# SSH into node
minikube ssh -p marie-dev
```

---

### Option D: Docker Desktop Kubernetes

For macOS and Windows users, Docker Desktop includes a built-in Kubernetes option.

#### Enable Kubernetes

1. Open Docker Desktop Settings
2. Go to **Kubernetes** tab
3. Check **Enable Kubernetes**
4. Click **Apply & Restart**
5. Wait for Kubernetes to start (green indicator)

#### Verify Setup

```bash
# Check context
kubectl config current-context
# Should show: docker-desktop

# Verify nodes
kubectl get nodes
```

#### Limitations

- Single-node only
- No easy image loading (requires registry)
- Higher resource usage
- Limited configuration options

---

### Quick Comparison Script

Use this script to quickly set up any local Kubernetes option:

```bash
#!/bin/bash
# setup-local-k8s.sh

set -e

CLUSTER_NAME="marie-dev"
K8S_VERSION="v1.29.0"

case "$1" in
  kind)
    echo "Setting up kind cluster..."
    cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${CLUSTER_NAME}
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 8080
      - containerPort: 30052
        hostPort: 52000
  - role: worker
  - role: worker
EOF
    ;;
  k3d)
    echo "Setting up k3d cluster..."
    k3d cluster create ${CLUSTER_NAME} \
      --servers 1 \
      --agents 2 \
      --port "8080:30080@server:0" \
      --port "52000:30052@server:0"
    ;;
  minikube)
    echo "Setting up minikube cluster..."
    minikube start \
      --cpus=4 \
      --memory=8192 \
      --profile=${CLUSTER_NAME} \
      --kubernetes-version=${K8S_VERSION}
    ;;
  *)
    echo "Usage: $0 {kind|k3d|minikube}"
    exit 1
    ;;
esac

echo "Cluster ${CLUSTER_NAME} is ready!"
kubectl cluster-info
kubectl get nodes
```

Make it executable and run:

```bash
chmod +x setup-local-k8s.sh

# Choose your preferred option
./setup-local-k8s.sh kind      # Recommended
./setup-local-k8s.sh k3d       # Fastest
./setup-local-k8s.sh minikube  # Most features
```

---

### Verifying Your Setup

After setting up any local Kubernetes cluster, verify it's ready:

```bash
# Check cluster is accessible
kubectl cluster-info

# Verify nodes are ready
kubectl get nodes -o wide

# Check system pods are running
kubectl get pods -n kube-system

# Test creating a simple deployment
kubectl create deployment nginx --image=nginx
kubectl wait --for=condition=available deployment/nginx --timeout=60s
kubectl delete deployment nginx

echo "Local Kubernetes cluster is ready for Marie Operator development!"
```

---

## Installing the Marie Operator

### Step 1: Clone the Repository

```bash
git clone https://github.com/marieai/marie-ai.git
cd marie-ai/deploy/operator
```

### Step 2: Install CRDs

```bash
# Using make
make install

# Or using kubectl directly
kubectl apply -f config/crd/bases/marie.ai_marieclusters.yaml
kubectl apply -f config/crd/bases/marie.ai_mariejobs.yaml

# Verify CRDs are installed
kubectl get crds | grep marie
```

Expected output:
```
marieclusters.marie.ai   2024-01-15T10:00:00Z
mariejobs.marie.ai       2024-01-15T10:00:00Z
```

### Step 3: Deploy the Operator

```bash
# Using kustomize
kubectl apply -k config/default

# Or deploy manually
kubectl create namespace marie-system
kubectl apply -f config/rbac/role.yaml
kubectl apply -f config/manager/manager.yaml

# Verify operator is running
kubectl get pods -n marie-system
```

Expected output:
```
NAME                              READY   STATUS    RESTARTS   AGE
marie-operator-6d4b8c7f9-x2j4k   1/1     Running   0          30s
```

### Step 4: Check Operator Logs

```bash
kubectl logs -n marie-system -l app.kubernetes.io/name=marie-operator -f
```

---

## Creating a MarieCluster

### Basic Cluster (CPU only)

Create a file `marie-cluster.yaml`:

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieCluster
metadata:
  name: marie-dev
  namespace: default
spec:
  serverSpec:
    replicas: 1
    configFile: "/config/marie.yml"
    template:
      spec:
        containers:
          - name: marie
            image: marieai/marie:3.0
            imagePullPolicy: IfNotPresent
            ports:
              - containerPort: 52000
                name: grpc
              - containerPort: 8080
                name: http
            resources:
              requests:
                cpu: "500m"
                memory: "1Gi"
              limits:
                cpu: "2"
                memory: "4Gi"
            env:
              - name: MARIE_LOG_LEVEL
                value: "INFO"

  executorGroupSpecs:
    - groupName: ocr-cpu
      replicas: 1
      executorType: ocr
      gpu: false
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0
              command: ["marie", "executor", "--start"]
              ports:
                - containerPort: 52001
                  name: grpc
              resources:
                requests:
                  cpu: "500m"
                  memory: "1Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi"

  serviceType: ClusterIP
  enableIngress: false
```

Apply the cluster:

```bash
kubectl apply -f marie-cluster.yaml

# Watch cluster status
kubectl get mariecluster -w

# Check all resources
kubectl get all -l marie.ai/cluster=marie-dev
```

### Cluster with GPU Executors

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieCluster
metadata:
  name: marie-gpu
  namespace: default
spec:
  serverSpec:
    replicas: 1
    template:
      spec:
        containers:
          - name: marie
            image: marieai/marie:3.0-cuda
            ports:
              - containerPort: 52000
                name: grpc
              - containerPort: 8080
                name: http
            resources:
              requests:
                cpu: "1"
                memory: "2Gi"

  executorGroupSpecs:
    # GPU OCR pool
    - groupName: ocr-gpu
      replicas: 2
      minReplicas: 0  # Enable scale-to-zero
      maxReplicas: 10
      executorType: ocr
      gpu: true
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0-cuda
              command: ["marie", "executor", "--start", "--type", "ocr"]
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
                  nvidia.com/gpu: "1"
                limits:
                  nvidia.com/gpu: "1"
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          nodeSelector:
            accelerator: nvidia-gpu

    # CPU classifier pool
    - groupName: classifier
      replicas: 2
      executorType: classifier
      gpu: false
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0
              command: ["marie", "executor", "--start", "--type", "classifier"]
              resources:
                requests:
                  cpu: "1"
                  memory: "2Gi"

  serviceType: LoadBalancer
  enableIngress: true
```

---

## Submitting Jobs

### HTTP Submission Mode

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieJob
metadata:
  name: ocr-batch-001
  namespace: default
spec:
  clusterRef: marie-dev
  jobType: ocr
  submissionMode: HTTPSubmission

  input:
    uri: "s3://my-bucket/documents/batch-001/"
    contentType: "application/pdf"

  output:
    uri: "s3://my-bucket/results/batch-001/"
    format: json

  options:
    language: "eng"
    deskew: "true"

  priority: 0
  backoffLimit: 3

  sla:
    softDeadline: "30m"
    hardDeadline: "1h"

  ttlSecondsAfterFinished: 3600
```

Submit and monitor:

```bash
kubectl apply -f ocr-job.yaml

# Watch job status
kubectl get mariejob -w

# Get detailed status
kubectl describe mariejob ocr-batch-001

# Check job progress
kubectl get mariejob ocr-batch-001 -o jsonpath='{.status.progress}'
```

### Pipeline Job

```yaml
apiVersion: marie.ai/v1alpha1
kind: MarieJob
metadata:
  name: invoice-pipeline
spec:
  clusterRef: marie-dev
  jobType: pipeline
  entrypoint: "classify -> ocr -> ner -> extract"

  input:
    uri: "s3://invoices/incoming/"
    contentType: "application/pdf"

  output:
    uri: "s3://invoices/processed/"
    format: json

  options:
    extract_tables: "true"
    extract_key_values: "true"

  priority: 10
  sla:
    softDeadline: "15m"
```

---

## Testing End-to-End Locally

### 1. Start Local Cluster

```bash
# Using minikube
minikube start --cpus=4 --memory=8192

# Or using kind
kind create cluster --name marie-test
```

### 2. Build and Load Operator Image

```bash
cd deploy/operator

# Build the operator
make build

# Build Docker image
make docker-build IMG=marieai/marie-operator:dev

# Load image into local cluster
# For minikube:
minikube image load marieai/marie-operator:dev

# For kind:
kind load docker-image marieai/marie-operator:dev --name marie-test
```

### 3. Deploy Operator with Local Image

```bash
# Update the image in kustomization
cd config/manager
kustomize edit set image controller=marieai/marie-operator:dev
cd ../..

# Deploy
kubectl apply -k config/default

# Verify
kubectl get pods -n marie-system
```

### 4. Create Test Namespace and Cluster

```bash
kubectl create namespace marie-test

# Create a minimal test cluster
cat <<EOF | kubectl apply -f -
apiVersion: marie.ai/v1alpha1
kind: MarieCluster
metadata:
  name: test-cluster
  namespace: marie-test
spec:
  serverSpec:
    replicas: 1
    template:
      spec:
        containers:
          - name: marie
            image: marieai/marie:3.0
            ports:
              - containerPort: 52000
              - containerPort: 8080
            resources:
              requests:
                cpu: "100m"
                memory: "256Mi"
  executorGroupSpecs:
    - groupName: test-executor
      replicas: 1
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0
              resources:
                requests:
                  cpu: "100m"
                  memory: "256Mi"
EOF

# Watch cluster come up
kubectl get mariecluster -n marie-test -w
```

### 5. Access the Cluster

```bash
# Port forward to access locally
kubectl port-forward -n marie-test svc/test-cluster-server-svc 8080:8080 &

# Test the API
curl http://localhost:8080/health

# Or use minikube service
minikube service test-cluster-server-svc -n marie-test --url
```

### 6. Submit a Test Job

```bash
cat <<EOF | kubectl apply -f -
apiVersion: marie.ai/v1alpha1
kind: MarieJob
metadata:
  name: test-job
  namespace: marie-test
spec:
  clusterRef: test-cluster
  jobType: ocr
  input:
    data: "base64-encoded-test-document"
    contentType: "application/pdf"
  output:
    format: json
EOF

# Monitor job
kubectl get mariejob -n marie-test -w
```

### 7. Cleanup

```bash
# Delete test resources
kubectl delete namespace marie-test

# Delete operator
kubectl delete -k config/default

# Delete CRDs
make uninstall

# Stop local cluster
minikube stop  # or: kind delete cluster --name marie-test
```

---

## Useful Commands

### Cluster Management

```bash
# List all MarieCluster resources
kubectl get mariecluster -A

# Get detailed cluster info
kubectl describe mariecluster <name>

# Scale executor group
kubectl patch mariecluster <name> --type=merge -p '
spec:
  executorGroupSpecs:
    - groupName: ocr-gpu
      replicas: 5
'

# Suspend cluster (scale to zero)
kubectl patch mariecluster <name> --type=merge -p '{"spec":{"suspend":true}}'

# Resume cluster
kubectl patch mariecluster <name> --type=merge -p '{"spec":{"suspend":false}}'
```

### Job Management

```bash
# List all jobs
kubectl get mariejob -A

# Get job status
kubectl get mariejob <name> -o jsonpath='{.status}'

# Cancel a job (delete it)
kubectl delete mariejob <name>

# View job logs
kubectl logs -l marie.ai/job=<job-name>
```

### Debugging

```bash
# Check operator logs
kubectl logs -n marie-system deployment/marie-operator -f

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Describe failing pods
kubectl describe pod <pod-name>

# Get all Marie resources
kubectl get all -l app.kubernetes.io/name=marie-ai
```

---

## Troubleshooting

### CRDs Not Found

```bash
# Reinstall CRDs
kubectl apply -f config/crd/bases/

# Verify
kubectl get crd marieclusters.marie.ai
```

### Operator Not Starting

```bash
# Check RBAC
kubectl auth can-i create deployments --as=system:serviceaccount:marie-system:marie-operator-controller-manager

# Check logs
kubectl logs -n marie-system -l app.kubernetes.io/name=marie-operator --previous
```

### Pods Stuck in Pending

```bash
# Check node resources
kubectl describe nodes | grep -A5 "Allocated resources"

# Check events
kubectl get events --field-selector reason=FailedScheduling
```

### GPU Pods Not Scheduling

```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check tolerations match taints
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints
```

---

## Complete End-to-End Deployment Example

This section provides a complete deployment example including all required infrastructure components.

### Architecture Overview

A complete Marie-AI deployment requires:
- **PostgreSQL**: Job queue, metadata storage, event tracking
- **S3/MinIO**: Document and artifact storage
- **RabbitMQ**: Event messaging (optional but recommended)
- **Marie Server**: Gateway + scheduler
- **Marie Executors**: Document processing workers (OCR, NER, Classifier)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  PostgreSQL  │  │    MinIO     │  │   RabbitMQ   │                  │
│  │   (Storage)  │  │  (S3 Storage)│  │  (Messaging) │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         └────────────────┼─────────────────┘                           │
│                          │                                              │
│                 ┌────────▼────────┐                                    │
│                 │   Marie Server  │                                    │
│                 │ (Gateway+Sched) │                                    │
│                 └────────┬────────┘                                    │
│                          │                                              │
│         ┌────────────────┼────────────────┐                           │
│         │                │                │                            │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐                   │
│  │ OCR Executor│  │NER Executor │  │Classifier   │                   │
│  │   (GPU)     │  │   (CPU)     │  │  Executor   │                   │
│  └─────────────┘  └─────────────┘  └─────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Deploy Infrastructure Components

Create a namespace for infrastructure:

```bash
kubectl create namespace marie-infra
```

#### PostgreSQL Deployment

```yaml
# postgresql.yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: postgresql-secret
  namespace: marie-infra
type: Opaque
stringData:
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: marie-secure-password
  POSTGRES_DB: marie
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgresql-pvc
  namespace: marie-infra
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgresql
  namespace: marie-infra
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
        - name: postgresql
          image: postgres:15
          ports:
            - containerPort: 5432
          envFrom:
            - secretRef:
                name: postgresql-secret
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: postgresql-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgresql
  namespace: marie-infra
spec:
  selector:
    app: postgresql
  ports:
    - port: 5432
      targetPort: 5432
```

#### MinIO (S3-Compatible Storage)

```yaml
# minio.yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: minio-secret
  namespace: marie-infra
type: Opaque
stringData:
  MINIO_ROOT_USER: minioadmin
  MINIO_ROOT_PASSWORD: minioadmin123
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: marie-infra
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: marie-infra
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
        - name: minio
          image: minio/minio:latest
          args:
            - server
            - /data
            - --console-address
            - ":9001"
          ports:
            - containerPort: 9000
              name: api
            - containerPort: 9001
              name: console
          envFrom:
            - secretRef:
                name: minio-secret
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "2Gi"
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: minio-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: marie-infra
spec:
  selector:
    app: minio
  ports:
    - port: 9000
      targetPort: 9000
      name: api
    - port: 9001
      targetPort: 9001
      name: console
```

#### RabbitMQ (Optional - for Event Messaging)

```yaml
# rabbitmq.yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: rabbitmq-secret
  namespace: marie-infra
type: Opaque
stringData:
  RABBITMQ_DEFAULT_USER: marie
  RABBITMQ_DEFAULT_PASS: marie-mq-password
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rabbitmq
  namespace: marie-infra
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rabbitmq
  template:
    metadata:
      labels:
        app: rabbitmq
    spec:
      containers:
        - name: rabbitmq
          image: rabbitmq:3-management
          ports:
            - containerPort: 5672
              name: amqp
            - containerPort: 15672
              name: management
          envFrom:
            - secretRef:
                name: rabbitmq-secret
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: rabbitmq
  namespace: marie-infra
spec:
  selector:
    app: rabbitmq
  ports:
    - port: 5672
      targetPort: 5672
      name: amqp
    - port: 15672
      targetPort: 15672
      name: management
```

Deploy infrastructure:

```bash
kubectl apply -f postgresql.yaml
kubectl apply -f minio.yaml
kubectl apply -f rabbitmq.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n marie-infra --timeout=120s
kubectl wait --for=condition=ready pod -l app=minio -n marie-infra --timeout=120s
kubectl wait --for=condition=ready pod -l app=rabbitmq -n marie-infra --timeout=120s

# Verify all services are running
kubectl get pods -n marie-infra
```

### Step 2: Initialize Database Schema

Port-forward to PostgreSQL and create required tables:

```bash
kubectl port-forward -n marie-infra svc/postgresql 5432:5432 &

# Connect and create tables
PGPASSWORD=marie-secure-password psql -h localhost -U postgres -d marie << 'EOF'
-- Job queue table
CREATE TABLE IF NOT EXISTS job_queue (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    payload JSONB,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    soft_deadline TIMESTAMP,
    hard_deadline TIMESTAMP
);

-- Event tracking table
CREATE TABLE IF NOT EXISTS event_tracking (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(255),
    payload JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document metadata table
CREATE TABLE IF NOT EXISTS store_metadata (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(500),
    content_type VARCHAR(100),
    size_bytes BIGINT,
    storage_path VARCHAR(1000),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extraction results table
CREATE TABLE IF NOT EXISTS extract_metadata (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    job_id VARCHAR(255),
    extraction_type VARCHAR(100),
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_event_tracking_type ON event_tracking(event_type);
CREATE INDEX IF NOT EXISTS idx_store_metadata_doc_id ON store_metadata(doc_id);
EOF
```

### Step 3: Create MinIO Bucket

```bash
kubectl port-forward -n marie-infra svc/minio 9000:9000 &

# Install mc (MinIO Client) if not already installed
# wget https://dl.min.io/client/mc/release/linux-amd64/mc
# chmod +x mc && sudo mv mc /usr/local/bin/

# Configure MinIO client
mc alias set myminio http://localhost:9000 minioadmin minioadmin123

# Create bucket for documents
mc mb myminio/marie-documents

# Set bucket policy (optional - for public read access)
mc anonymous set download myminio/marie-documents
```

### Step 4: Create Marie Configuration ConfigMap

```yaml
# marie-config.yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: marie-config
  namespace: default
data:
  marie.yml: |
    jtype: Flow
    version: '1'
    protocol: grpc

    # Shared configuration
    shared_config:
      storage: &storage
        psql: &psql_conf_shared
          provider: postgresql
          hostname: postgresql.marie-infra.svc.cluster.local
          port: 5432
          username: postgres
          password: marie-secure-password
          database: marie
          default_table: shared_docs

      message: &message
        rabbitmq: &rabbitmq_conf_shared
          provider: rabbitmq
          hostname: rabbitmq.marie-infra.svc.cluster.local
          port: 5672
          username: marie
          password: marie-mq-password
          tls: False
          virtualhost: /

    # Event tracking
    toast:
      native:
        enabled: True
        path: /tmp/marie/events.json
      rabbitmq:
        <<: *rabbitmq_conf_shared
        enabled: True
      psql:
        <<: *psql_conf_shared
        default_table: event_tracking
        enabled: True

    # Document Storage
    storage:
      s3:
        enabled: True
        metadata_only: False
        endpoint_url: http://minio.marie-infra.svc.cluster.local:9000
        access_key_id: minioadmin
        secret_access_key: minioadmin123
        bucket_name: marie-documents
        region: us-east-1
        insecure: True
        addressing_style: path

      psql:
        <<: *psql_conf_shared
        default_table: store_metadata
        enabled: True

    # Job Queue scheduler
    scheduler:
      psql:
        <<: *psql_conf_shared
        default_table: job_queue
        enabled: True

    # Gateway configuration
    with:
      port:
        - 51000
        - 52000
      protocol:
        - http
        - grpc
      discovery: False
      host: 0.0.0.0
      monitoring: true
      port_monitoring: 9090
      event_tracking: True

      expose_endpoints:
        /document/extract:
          methods: ["POST"]
          summary: Extract text from documents
          tags:
            - extract
        /status:
          methods: ["POST"]
          summary: Check job status
          tags:
            - status
        /ner/extract:
          methods: ["POST"]
          summary: Extract named entities
          tags:
            - ner
        /document/classify:
          methods: ["POST"]
          summary: Classify documents
          tags:
            - classify

    prefetch: 1

    executors:
      - name: extract_executor
        uses:
          jtype: TextExtractionExecutor
          with:
            storage:
              psql:
                <<: *psql_conf_shared
                default_table: extract_metadata
                enabled: True
          metas:
            py_modules:
              - marie.executor.text
        timeout_ready: 3000000
        replicas: 1
        env:
          CUDA_VISIBLE_DEVICES: "0"
```

Apply the ConfigMap:

```bash
kubectl apply -f marie-config.yaml
```

### Step 5: Deploy Marie Operator and Cluster

```bash
# Install CRDs and operator
cd deploy/operator
make install
kubectl apply -k config/default

# Wait for operator to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=marie-operator -n marie-system --timeout=120s
```

### Step 6: Create Production-Ready MarieCluster

```yaml
# marie-cluster-production.yaml
---
apiVersion: marie.ai/v1alpha1
kind: MarieCluster
metadata:
  name: marie-production
  namespace: default
spec:
  serverSpec:
    replicas: 2
    configFile: "/config/marie.yml"
    template:
      spec:
        containers:
          - name: marie
            image: marieai/marie:3.0-cuda
            imagePullPolicy: IfNotPresent
            ports:
              - containerPort: 51000
                name: http
              - containerPort: 52000
                name: grpc
              - containerPort: 9090
                name: metrics
            resources:
              requests:
                cpu: "1"
                memory: "2Gi"
              limits:
                cpu: "2"
                memory: "4Gi"
            env:
              - name: MARIE_LOG_LEVEL
                value: "INFO"
            volumeMounts:
              - name: config
                mountPath: /config
            livenessProbe:
              httpGet:
                path: /health
                port: 51000
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /ready
                port: 51000
              initialDelaySeconds: 10
              periodSeconds: 5
        volumes:
          - name: config
            configMap:
              name: marie-config

  executorGroupSpecs:
    # GPU OCR pool
    - groupName: ocr-gpu
      replicas: 2
      minReplicas: 1
      maxReplicas: 10
      executorType: ocr
      gpu: true
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0-cuda
              imagePullPolicy: IfNotPresent
              command: ["marie", "executor", "--start", "--type", "ocr"]
              ports:
                - containerPort: 52001
                  name: grpc
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "4"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
              env:
                - name: MARIE_LOG_LEVEL
                  value: "INFO"
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          nodeSelector:
            accelerator: nvidia-gpu

    # CPU Classifier pool
    - groupName: classifier
      replicas: 2
      minReplicas: 1
      maxReplicas: 5
      executorType: classifier
      gpu: false
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0
              imagePullPolicy: IfNotPresent
              command: ["marie", "executor", "--start", "--type", "classifier"]
              ports:
                - containerPort: 52001
                  name: grpc
              resources:
                requests:
                  cpu: "1"
                  memory: "2Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi"

    # NER Executor pool
    - groupName: ner
      replicas: 1
      minReplicas: 0
      maxReplicas: 3
      executorType: ner
      gpu: false
      template:
        spec:
          containers:
            - name: marie
              image: marieai/marie:3.0
              imagePullPolicy: IfNotPresent
              command: ["marie", "executor", "--start", "--type", "ner"]
              ports:
                - containerPort: 52001
                  name: grpc
              resources:
                requests:
                  cpu: "1"
                  memory: "2Gi"
                limits:
                  cpu: "2"
                  memory: "4Gi"

  serviceType: LoadBalancer
  enableIngress: true
  suspend: false
  marieVersion: "3.0"
```

Deploy the cluster:

```bash
kubectl apply -f marie-cluster-production.yaml

# Watch cluster creation
kubectl get mariecluster -w

# Check all resources
kubectl get pods,svc,deployments -l marie.ai/cluster=marie-production
```

### Step 7: Verify Deployment

```bash
# Check cluster status
kubectl describe mariecluster marie-production

# Check all pods are running
kubectl get pods -l marie.ai/cluster=marie-production

# Get service endpoints
kubectl get svc -l marie.ai/cluster=marie-production

# Test API connectivity
kubectl port-forward svc/marie-production-server-svc 8080:51000 &
curl http://localhost:8080/health
```

### Step 8: Submit a Test Job

```yaml
# test-job.yaml
---
apiVersion: marie.ai/v1alpha1
kind: MarieJob
metadata:
  name: e2e-test-job
  namespace: default
spec:
  clusterRef: marie-production
  jobType: ocr
  submissionMode: HTTPSubmission
  priority: 10

  input:
    uri: "s3://marie-documents/test-documents/"
    contentType: "application/pdf"

  output:
    uri: "s3://marie-documents/results/"
    format: json

  options:
    language: "eng"
    deskew: "true"
    enhance_quality: "true"

  sla:
    softDeadline: "5m"
    hardDeadline: "15m"

  backoffLimit: 3
  ttlSecondsAfterFinished: 3600
```

Submit and monitor:

```bash
# Upload a test document first
mc cp test-document.pdf myminio/marie-documents/test-documents/

# Submit the job
kubectl apply -f test-job.yaml

# Watch job progress
kubectl get mariejob e2e-test-job -w

# Get detailed status
kubectl describe mariejob e2e-test-job

# Check job logs
kubectl logs -l marie.ai/job=e2e-test-job

# Check results in MinIO
mc ls myminio/marie-documents/results/
```

### Complete Deployment Script

For convenience, here's a complete deployment script:

```bash
#!/bin/bash
# deploy-marie-e2e.sh

set -e

echo "=== Marie-AI End-to-End Deployment ==="

# Create namespaces
kubectl create namespace marie-infra --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace marie-system --dry-run=client -o yaml | kubectl apply -f -

# Deploy infrastructure
echo "Deploying PostgreSQL..."
kubectl apply -f postgresql.yaml

echo "Deploying MinIO..."
kubectl apply -f minio.yaml

echo "Deploying RabbitMQ..."
kubectl apply -f rabbitmq.yaml

# Wait for infrastructure
echo "Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app=postgresql -n marie-infra --timeout=180s
kubectl wait --for=condition=ready pod -l app=minio -n marie-infra --timeout=180s
kubectl wait --for=condition=ready pod -l app=rabbitmq -n marie-infra --timeout=180s

# Initialize database (simplified - in production use migrations)
echo "Initializing database..."
kubectl exec -n marie-infra deployment/postgresql -- psql -U postgres -d marie -c "
CREATE TABLE IF NOT EXISTS job_queue (id SERIAL PRIMARY KEY, job_id VARCHAR(255) UNIQUE);
CREATE TABLE IF NOT EXISTS event_tracking (id SERIAL PRIMARY KEY, event_id VARCHAR(255) UNIQUE);
"

# Apply Marie configuration
echo "Applying Marie configuration..."
kubectl apply -f marie-config.yaml

# Install operator
echo "Installing Marie Operator..."
cd deploy/operator
make install
kubectl apply -k config/default
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=marie-operator -n marie-system --timeout=120s

# Deploy cluster
echo "Deploying MarieCluster..."
kubectl apply -f marie-cluster-production.yaml

# Wait for cluster
echo "Waiting for MarieCluster to be ready..."
sleep 30
kubectl get mariecluster marie-production

echo "=== Deployment Complete ==="
echo "Access the API at: kubectl port-forward svc/marie-production-server-svc 8080:51000"
```

---

## Helm Chart Deployment (Recommended)

For production deployments, use the Marie-AI Helm chart which provides a complete, configurable deployment including all infrastructure dependencies.

### Prerequisites

```bash
# Install Helm 3
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version
```

### Quick Start

```bash
cd deploy/helm/charts/marie

# Update dependencies (PostgreSQL, RabbitMQ, MinIO, etcd)
helm dependency update

# Install with default values
helm install marie . -n marie --create-namespace

# Or install with local development values
helm install marie . -f values-local.yaml -n marie --create-namespace
```

### Local Development Deployment

The `values-local.yaml` file is configured to match a typical local Docker environment:

| Component | Image | Notes |
|-----------|-------|-------|
| PostgreSQL | `ghcr.io/ferretdb/postgres-documentdb:17-0.103.0` | FerretDB with MongoDB compatibility |
| MinIO | `minio/minio:latest` | S3-compatible storage |
| RabbitMQ | `rabbitmq:3-management-alpine` | Message queue with management UI |
| Marie | `marieai/marie:3.0` | Main application |

```bash
# Deploy for local development
helm install marie . \
  -f values-local.yaml \
  -n marie \
  --create-namespace

# Watch pods come up
kubectl get pods -n marie -w

# Check deployment status
kubectl get all -n marie
```

### Custom Values

Create a custom values file for your environment:

```yaml
# my-values.yaml
marie:
  image:
    repository: marieai/marie
    tag: "3.0-cuda"  # Use CUDA for GPU support

server:
  replicas: 2
  service:
    type: LoadBalancer  # Use LoadBalancer for cloud deployments

executor:
  pools:
    - name: gpu-ocr
      enabled: true
      replicas: 2
      gpu: true
      resources:
        requests:
          nvidia.com/gpu: 1
        limits:
          nvidia.com/gpu: 1

postgresql:
  auth:
    postgresPassword: "your-secure-password"
    database: marie

minio:
  auth:
    rootPassword: "your-minio-password"

rabbitmq:
  auth:
    password: "your-rabbitmq-password"
```

Deploy with custom values:

```bash
helm install marie . -f my-values.yaml -n marie --create-namespace
```

### Accessing Services

After deployment, access the services:

```bash
# Get service info
kubectl get svc -n marie

# Port-forward to Marie Server
kubectl port-forward -n marie svc/marie-server 52000:52000 &
kubectl port-forward -n marie svc/marie-server 8080:8080 &

# Port-forward to MinIO Console
kubectl port-forward -n marie svc/marie-minio 9001:9001 &
# Open http://localhost:9001 (minioadmin/minioadmin123)

# Port-forward to RabbitMQ Management
kubectl port-forward -n marie svc/marie-rabbitmq 15672:15672 &
# Open http://localhost:15672 (marie/marie-mq-password)
```

### Minikube Deployment

For minikube, use NodePort services:

```bash
# Start minikube with sufficient resources
minikube start --cpus=4 --memory=8192 -p marie-dev

# Deploy
helm install marie . -f values-local.yaml -n marie --create-namespace

# Get minikube IP
minikube ip -p marie-dev

# Access via NodePort (check NOTES output for ports)
```

### Upgrading

```bash
# Update values and upgrade
helm upgrade marie . -f values-local.yaml -n marie

# Rollback if needed
helm rollback marie -n marie
```

### Uninstalling

```bash
# Uninstall release (keeps PVCs by default)
helm uninstall marie -n marie

# Delete namespace and all resources
kubectl delete namespace marie

# Delete PVCs if needed
kubectl delete pvc -n marie --all
```

### Helm Chart Structure

```
deploy/helm/charts/marie/
├── Chart.yaml              # Chart metadata and dependencies
├── values.yaml             # Default configuration
├── values-local.yaml       # Local development configuration
├── templates/
│   ├── _helpers.tpl        # Template helpers
│   ├── configmap.yaml      # Environment configuration
│   ├── serviceaccount.yaml # ServiceAccount
│   └── NOTES.txt           # Post-install instructions
└── charts/
    ├── server/             # Marie Server subchart
    │   ├── templates/
    │   │   ├── configmap.yaml
    │   │   ├── deployment.yaml
    │   │   ├── service.yaml
    │   │   ├── ingress.yaml
    │   │   └── hpa.yaml
    │   └── values.yaml
    └── executor/           # Executor subchart
        ├── templates/
        │   ├── configmap.yaml
        │   ├── deployment.yaml
        │   ├── statefulset.yaml
        │   └── service.yaml
        └── values.yaml
```

### Dependencies

The chart includes these Bitnami dependencies:

| Dependency | Version | Condition |
|------------|---------|-----------|
| postgresql | 13.2.24 | `postgresql.enabled` |
| rabbitmq | 12.5.6 | `rabbitmq.enabled` |
| etcd | 9.10.0 | `etcd.enabled` |
| minio | 14.1.0 | `minio.enabled` |

To use external services instead of bundled ones:

```yaml
# Disable bundled PostgreSQL, use external
postgresql:
  enabled: false
  external:
    host: "your-postgres-host.example.com"
    port: 5432
    database: marie
    username: marie
    existingSecret: "postgres-credentials"
    existingSecretPasswordKey: password
```

---

## Next steps

- [Monitoring and observability](./observability.md) - Set up monitoring
- [Docker deployment](./docker.md) - Single-node Docker setup
- [Control plane](./control-plane.md) - Control plane configuration
