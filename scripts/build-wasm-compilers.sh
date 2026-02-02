#!/bin/bash
# build-wasm-compilers.sh - Build Wasm compiler containers for Marie
#
# Usage:
#   ./scripts/build-wasm-compilers.sh [build|push|test|all]
#
# Commands:
#   build  - Build all compiler containers locally
#   push   - Push containers to registry (requires docker login)
#   test   - Test each compiler with a hello world program
#   all    - Build, test, and push

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPILERS_DIR="${PROJECT_ROOT}/Dockerfiles/compilers"

# Configuration
REGISTRY="${WASM_REGISTRY:-marieai}"
VERSION="${WASM_VERSION:-$(git describe --tags --always 2>/dev/null || echo "dev")}"
LANGUAGES=(rust python js)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Marie Wasm Compiler Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Registry: ${REGISTRY}"
echo "Version: ${VERSION}"
echo ""

build_compilers() {
    echo -e "${BLUE}=== Building Wasm Compiler Containers ===${NC}"

    for lang in "${LANGUAGES[@]}"; do
        local image="${REGISTRY}/marie-compiler-${lang}:${VERSION}"
        echo -e "${YELLOW}Building ${image}...${NC}"

        docker build \
            -t "${image}" \
            -t "${REGISTRY}/marie-compiler-${lang}:latest" \
            -f "${COMPILERS_DIR}/${lang}/Dockerfile" \
            "${COMPILERS_DIR}/"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✔ Built ${image}${NC}"
        else
            echo -e "${RED}✖ Failed to build ${image}${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}=== All compilers built successfully ===${NC}"
}

push_compilers() {
    echo -e "${BLUE}=== Pushing Wasm Compiler Containers ===${NC}"

    for lang in "${LANGUAGES[@]}"; do
        local image="${REGISTRY}/marie-compiler-${lang}:${VERSION}"
        echo -e "${YELLOW}Pushing ${image}...${NC}"

        docker push "${image}"
        docker push "${REGISTRY}/marie-compiler-${lang}:latest"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✔ Pushed ${image}${NC}"
        else
            echo -e "${RED}✖ Failed to push ${image}${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}=== All compilers pushed successfully ===${NC}"
}

test_compilers() {
    echo -e "${BLUE}=== Testing Wasm Compiler Containers ===${NC}"

    local tmpdir
    tmpdir=$(mktemp -d)
    # Cleanup function that handles files created by containers with different UIDs
    # Use a subshell trap to capture tmpdir value
    trap "docker run --rm -v '${tmpdir}:/cleanup:rw' alpine:3.19 rm -rf /cleanup/* 2>/dev/null || true; rm -rf '${tmpdir}' 2>/dev/null || true" EXIT

    # Test Rust compiler
    echo -e "${YELLOW}Testing Rust compiler...${NC}"
    mkdir -p "${tmpdir}/rust"
    chmod 777 "${tmpdir}/rust"
    cat > "${tmpdir}/rust/lib.rs" << 'RUSTEOF'
wit_bindgen::generate!({
    world: "node",
    path: "/opt/wit",
    generate_all,
});

fn execute(_input: Vec<Item>, _env: Env, _ctx: Context) -> Response {
    Response::Ok(vec![
        Item {
            json: r#"{"message": "Hello from Rust!"}"#.to_string(),
            binary: None,
        }
    ])
}

struct Component;
impl Guest for Component {
    fn execute(input: Vec<Item>, env: Env, ctx: Context) -> Response {
        execute(input, env, ctx)
    }
}
export!(Component);
RUSTEOF

    if docker run --rm \
        -v "${tmpdir}/rust:/workspace:rw" \
        "${REGISTRY}/marie-compiler-rust:${VERSION}" 2>&1; then
        if [ -f "${tmpdir}/rust/output.wasm" ]; then
            local size=$(stat -c%s "${tmpdir}/rust/output.wasm" 2>/dev/null || stat -f%z "${tmpdir}/rust/output.wasm")
            echo -e "${GREEN}✔ Rust compiler works (output: ${size} bytes)${NC}"
        else
            echo -e "${RED}✖ Rust compiler produced no output${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✖ Rust compiler failed${NC}"
        exit 1
    fi

    # Test Python compiler
    echo -e "${YELLOW}Testing Python compiler...${NC}"
    mkdir -p "${tmpdir}/python"
    chmod 777 "${tmpdir}/python"
    cat > "${tmpdir}/python/main.py" << 'PYEOF'
import node
from node.imports.types import Item, Env, Context, Response, Response_Ok

class Node(node.Node):
    def execute(self, input: list[Item], env: Env, ctx: Context) -> Response:
        return Response_Ok([
            Item(json='{"message": "Hello from Python!"}', binary=None)
        ])
PYEOF

    if docker run --rm \
        -v "${tmpdir}/python:/workspace:rw" \
        "${REGISTRY}/marie-compiler-python:${VERSION}" 2>&1; then
        if [ -f "${tmpdir}/python/output.wasm" ]; then
            local size=$(stat -c%s "${tmpdir}/python/output.wasm" 2>/dev/null || stat -f%z "${tmpdir}/python/output.wasm")
            echo -e "${GREEN}✔ Python compiler works (output: ${size} bytes)${NC}"
        else
            echo -e "${RED}✖ Python compiler produced no output${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✖ Python compiler failed${NC}"
        exit 1
    fi

    # Test JavaScript compiler
    echo -e "${YELLOW}Testing JavaScript compiler...${NC}"
    mkdir -p "${tmpdir}/js"
    chmod 777 "${tmpdir}/js"
    cat > "${tmpdir}/js/main.js" << 'EOF'
export function execute(input, env, ctx) {
    return { ok: [{ json: '{"message": "Hello from JavaScript!"}' }] };
}
EOF

    if docker run --rm \
        -v "${tmpdir}/js:/workspace:rw" \
        "${REGISTRY}/marie-compiler-js:${VERSION}" 2>&1; then
        if [ -f "${tmpdir}/js/output.wasm" ]; then
            local size=$(stat -c%s "${tmpdir}/js/output.wasm" 2>/dev/null || stat -f%z "${tmpdir}/js/output.wasm")
            echo -e "${GREEN}✔ JavaScript compiler works (output: ${size} bytes)${NC}"
        else
            echo -e "${RED}✖ JavaScript compiler produced no output${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✖ JavaScript compiler failed${NC}"
        exit 1
    fi

    # Test TypeScript compiler (uses same container as JS)
    echo -e "${YELLOW}Testing TypeScript compiler...${NC}"
    mkdir -p "${tmpdir}/ts"
    chmod 777 "${tmpdir}/ts"
    cat > "${tmpdir}/ts/main.ts" << 'EOF'
interface Item {
    json: string;
    binary?: number[] | null;
}

interface Context {
    workflowId: string;
    executionId: string;
    nodeId: string;
    runIndex: number;
}

type Response = { ok: Item[] } | { err: string };

export function execute(
    input: Item[],
    env: Record<string, unknown>,
    ctx: Context
): Response {
    const message = `Hello from TypeScript! Node: ${ctx.nodeId}`;
    return { ok: [{ json: JSON.stringify({ message }) }] };
}
EOF

    if docker run --rm \
        -v "${tmpdir}/ts:/workspace:rw" \
        "${REGISTRY}/marie-compiler-js:${VERSION}" 2>&1; then
        if [ -f "${tmpdir}/ts/output.wasm" ]; then
            local size=$(stat -c%s "${tmpdir}/ts/output.wasm" 2>/dev/null || stat -f%z "${tmpdir}/ts/output.wasm")
            echo -e "${GREEN}✔ TypeScript compiler works (output: ${size} bytes)${NC}"
        else
            echo -e "${RED}✖ TypeScript compiler produced no output${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✖ TypeScript compiler failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}=== All compiler tests passed ===${NC}"
}

verify_wasmtime() {
    echo -e "${BLUE}=== Verifying Wasmtime Installation ===${NC}"

    if python3 -c "import wasmtime; print(f'Wasmtime version: {wasmtime.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✔ Wasmtime Python bindings available${NC}"
    else
        echo -e "${YELLOW}⚠ Wasmtime not installed. Installing...${NC}"
        pip install 'wasmtime>=21.0.0'
        python3 -c "import wasmtime; print(f'Wasmtime version: {wasmtime.__version__}')"
        echo -e "${GREEN}✔ Wasmtime installed${NC}"
    fi
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build all compiler containers locally"
    echo "  push      Push containers to registry"
    echo "  test      Test each compiler with hello world"
    echo "  verify    Verify wasmtime-py is installed"
    echo "  all       Build, test, and push"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  WASM_REGISTRY  Docker registry (default: marieai)"
    echo "  WASM_VERSION   Image version tag (default: git tag or 'dev')"
}

# Main command handling
case "${1:-build}" in
    build)
        build_compilers
        ;;
    push)
        push_compilers
        ;;
    test)
        test_compilers
        ;;
    verify)
        verify_wasmtime
        ;;
    all)
        build_compilers
        test_compilers
        push_compilers
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
