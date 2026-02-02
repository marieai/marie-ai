#!/bin/bash
# Python Wasm Component Compiler Script
# Compiles Python source code to a WebAssembly component

set -euo pipefail

echo "=== Marie Python Wasm Compiler ==="

# Check for source file
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in workspace"
    exit 1
fi

echo "Generating Python bindings from WIT..."

# Generate bindings first - this creates a 'node' module with the interface
# --world-module node keeps the module named 'node' (new default is 'wit_world')
componentize-py \
    --wit-path /opt/wit \
    --world node \
    --world-module node \
    bindings \
    . \
    2>&1

echo "Compiling Python to Wasm component..."

# Compile to component using componentize-py
componentize-py \
    --wit-path /opt/wit \
    --world node \
    --world-module node \
    componentize \
    main \
    -o /workspace/output.wasm \
    2>&1

if [ ! -f "/workspace/output.wasm" ]; then
    echo "Error: Compilation produced no output"
    exit 1
fi

WASM_SIZE=$(stat -c%s /workspace/output.wasm 2>/dev/null || stat -f%z /workspace/output.wasm)
echo "=== Compilation successful ==="
echo "Output: /workspace/output.wasm"
echo "Size: $WASM_SIZE bytes"
