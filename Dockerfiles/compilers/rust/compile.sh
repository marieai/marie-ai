#!/bin/bash
# Rust Wasm Component Compiler Script
# Compiles Rust source code to a WebAssembly component

set -euo pipefail

echo "=== Marie Rust Wasm Compiler ==="

# Check for source file
if [ ! -f "lib.rs" ] && [ ! -f "src/lib.rs" ]; then
    echo "Error: lib.rs or src/lib.rs not found in workspace"
    exit 1
fi

# Generate Cargo.toml if not present (single-file mode)
if [ ! -f "Cargo.toml" ]; then
    echo "Generating Cargo.toml..."
    cat > Cargo.toml << 'EOF'
[package]
name = "marie-user-node"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wit-bindgen = "0.51"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[package.metadata.component]
package = "marie:user-node"

[package.metadata.component.target]
path = "/opt/wit"
world = "node"
EOF

    # Create src directory if needed
    if [ -f "lib.rs" ] && [ ! -d "src" ]; then
        mkdir -p src
        mv lib.rs src/lib.rs
    fi
fi

echo "Building Wasm component..."

# Initialize cargo if needed
if [ ! -d ".cargo" ]; then
    mkdir -p .cargo
fi

# Build the component
cargo component build --release 2>&1

# Find the output wasm file - check wasip1 first (newer), then wasi (older)
echo "Looking for compiled wasm file..."

WASM_FILE=""
if [ -d "target/wasm32-wasip1/release" ]; then
    WASM_FILE=$(find target/wasm32-wasip1/release -maxdepth 1 -name "*.wasm" -type f 2>/dev/null | head -1)
fi

if [ -z "$WASM_FILE" ] && [ -d "target/wasm32-wasi/release" ]; then
    WASM_FILE=$(find target/wasm32-wasi/release -maxdepth 1 -name "*.wasm" -type f 2>/dev/null | head -1)
fi

if [ -z "$WASM_FILE" ]; then
    echo "Error: No .wasm file found in target directory"
    echo "Contents of target/:"
    find target -name "*.wasm" -type f 2>/dev/null || true
    exit 1
fi

echo "Found: $WASM_FILE"

# Copy to expected output location
cp "$WASM_FILE" /workspace/output.wasm || {
    echo "Error: Failed to copy wasm file"
    exit 1
}

WASM_SIZE=$(stat -c%s /workspace/output.wasm 2>/dev/null || stat -f%z /workspace/output.wasm)
echo "=== Compilation successful ==="
echo "Output: /workspace/output.wasm"
echo "Size: $WASM_SIZE bytes"
