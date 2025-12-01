# Marie Operator Makefile
# Inspired by KubeRay's build system

# Build metadata
BUILD_TIME      := $(shell date "+%F %T")
COMMIT_SHA1     := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
VERSION         ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Image URL to use all building/pushing image targets
IMG ?= marieai/marie-operator:latest
# ENVTEST_K8S_VERSION refers to the version of kubebuilder assets to be downloaded by envtest binary.
ENVTEST_K8S_VERSION ?= 1.29.0
# CRD generation options
CRD_OPTIONS ?= "crd:maxDescLen=0,generateEmbeddedObjectMeta=true"

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# Container Engine to be used for building images (docker or podman)
ENGINE ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: manifests
manifests: controller-gen ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	$(CONTROLLER_GEN) $(CRD_OPTIONS) rbac:roleName=marie-operator-manager-role webhook paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: generate
generate: controller-gen ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./..."

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: fumpt
fumpt: gofumpt ## Run gofumpt for stricter formatting.
	$(GOFUMPT) -l -w .

.PHONY: lint
lint: golangci-lint ## Run golangci-lint against code.
	$(GOLANGCI_LINT) run --timeout 5m ./...

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint with auto-fix.
	$(GOLANGCI_LINT) run --fix --timeout 5m ./...

.PHONY: test
test: manifests generate fmt vet envtest ## Run unit tests.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./... -coverprofile cover.out

.PHONY: test-e2e
test-e2e: manifests fmt vet ## Run e2e tests (requires running cluster).
	go test -timeout 30m -v ./test/e2e/...

.PHONY: test-coverage
test-coverage: test ## Run tests and display coverage report.
	go tool cover -html=cover.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

##@ Build

.PHONY: build
build: fmt vet ## Build manager binary.
	go build \
		-ldflags " \
			-X 'main.buildTime=${BUILD_TIME}' \
			-X 'main.commitSha=${COMMIT_SHA1}' \
			-X 'main.version=${VERSION}' \
		" \
		-o bin/manager cmd/manager/main.go

.PHONY: run
run: manifests generate fmt vet ## Run a controller from your host.
	go run ./cmd/manager/main.go

.PHONY: docker-image
docker-image: ## Build docker image only (without rebuilding binary).
	${ENGINE} build -t ${IMG} .

.PHONY: docker-build
docker-build: build docker-image ## Build binary and docker image.

.PHONY: docker-push
docker-push: ## Push docker image.
	${ENGINE} push ${IMG}

.PHONY: docker-multi-arch-build
docker-multi-arch-build: ## Build multi-arch image (amd64 and arm64).
	${ENGINE} buildx build --platform linux/amd64,linux/arm64 -t ${IMG} .

.PHONY: docker-multi-arch-push
docker-multi-arch-push: ## Build and push multi-arch image.
	${ENGINE} buildx build --push --platform linux/amd64,linux/arm64 -t ${IMG} .

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: manifests kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	($(KUSTOMIZE) build config/crd | kubectl create -f -) || ($(KUSTOMIZE) build config/crd | kubectl replace -f -)

.PHONY: uninstall
uninstall: manifests kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/crd | kubectl delete --ignore-not-found=$(ignore-not-found) -f -

.PHONY: deploy
deploy: manifests kustomize ## Deploy controller to the K8s cluster specified in ~/.kube/config.
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default | kubectl apply --server-side=true -f -

.PHONY: undeploy
undeploy: ## Undeploy controller from the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/default | kubectl delete --ignore-not-found=$(ignore-not-found) -f -

.PHONY: helm
helm: manifests ## Sync CRDs into Helm chart (when Helm chart is available).
	@if [ -d "../helm/marie-operator/crds" ]; then \
		rm -rf ../helm/marie-operator/crds/; \
		cp -r config/crd/bases/ ../helm/marie-operator/crds/; \
		echo "CRDs synced to Helm chart"; \
	else \
		echo "Helm chart directory not found, skipping"; \
	fi

##@ Local Development

.PHONY: kind-cluster
kind-cluster: ## Create a local kind cluster for development.
	../setup-local-k8s.sh kind

.PHONY: kind-load
kind-load: docker-build ## Build and load image into kind cluster.
	kind load docker-image ${IMG} --name marie-dev

.PHONY: kind-deploy
kind-deploy: kind-load install deploy ## Build, load, and deploy to kind cluster.

.PHONY: kind-delete
kind-delete: ## Delete the local kind cluster.
	kind delete cluster --name marie-dev

##@ Code Quality

.PHONY: verify
verify: fmt vet lint test ## Run all verification checks.

.PHONY: verify-codegen
verify-codegen: generate manifests ## Verify generated code is up to date.
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Generated files are out of date. Please run 'make generate manifests' and commit the changes."; \
		git diff; \
		exit 1; \
	fi

.PHONY: pre-commit
pre-commit: verify-codegen verify ## Run all pre-commit checks.

##@ API Documentation

.PHONY: api-docs
api-docs: crd-ref-docs ## Generate CRD API documentation.
	$(CRD_REF_DOCS) \
		--config=./hack/api-docs-config.yaml \
		--source-path=./api \
		--renderer=markdown \
		--output-path=../../docs/docs/reference/operator-api.md

##@ Build Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
ENVTEST ?= $(LOCALBIN)/setup-envtest
GOFUMPT ?= $(LOCALBIN)/gofumpt
GOLANGCI_LINT ?= $(LOCALBIN)/golangci-lint
CRD_REF_DOCS ?= $(LOCALBIN)/crd-ref-docs

## Tool Versions
KUSTOMIZE_VERSION ?= v5.3.0
CONTROLLER_TOOLS_VERSION ?= v0.17.0
GOLANGCI_LINT_VERSION ?= v1.61.0
CRD_REF_DOCS_VERSION ?= v0.0.12

.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	@if test -x $(LOCALBIN)/kustomize && ! $(LOCALBIN)/kustomize version | grep -q $(KUSTOMIZE_VERSION); then \
		echo "$(LOCALBIN)/kustomize version is not expected $(KUSTOMIZE_VERSION). Removing it before installing."; \
		rm -rf $(LOCALBIN)/kustomize; \
	fi
	test -s $(LOCALBIN)/kustomize || GOBIN=$(LOCALBIN) go install sigs.k8s.io/kustomize/kustomize/v5@$(KUSTOMIZE_VERSION)

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	test -s $(LOCALBIN)/controller-gen && $(LOCALBIN)/controller-gen --version | grep -q $(CONTROLLER_TOOLS_VERSION) || \
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION)

.PHONY: envtest
envtest: $(ENVTEST) ## Download envtest-setup locally if necessary.
$(ENVTEST): $(LOCALBIN)
	test -s $(LOCALBIN)/setup-envtest || GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-runtime/tools/setup-envtest@latest

.PHONY: gofumpt
gofumpt: $(GOFUMPT) ## Download gofumpt locally if necessary.
$(GOFUMPT): $(LOCALBIN)
	test -s $(LOCALBIN)/gofumpt || GOBIN=$(LOCALBIN) go install mvdan.cc/gofumpt@latest

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	test -s $(LOCALBIN)/golangci-lint || GOBIN=$(LOCALBIN) go install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

.PHONY: crd-ref-docs
crd-ref-docs: $(CRD_REF_DOCS) ## Download crd-ref-docs locally if necessary.
$(CRD_REF_DOCS): $(LOCALBIN)
	test -s $(LOCALBIN)/crd-ref-docs || GOBIN=$(LOCALBIN) go install github.com/elastic/crd-ref-docs@$(CRD_REF_DOCS_VERSION)

.PHONY: clean
clean: ## Clean build artifacts.
	rm -rf bin/
	rm -rf $(LOCALBIN)
	rm -f cover.out coverage.html
