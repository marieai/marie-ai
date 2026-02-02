#!/bin/bash
# JavaScript/TypeScript Wasm Component Compiler Script
# Compiles JavaScript or TypeScript source code to a WebAssembly component
# Supports: main.js, main.ts, index.js, index.ts

set -euo pipefail

echo "=== Marie JavaScript/TypeScript Wasm Compiler ==="

# Find the main source file (prefer .ts over .js)
MAIN_FILE=""
for candidate in main.ts index.ts main.js index.js; do
    if [ -f "$candidate" ]; then
        MAIN_FILE="$candidate"
        break
    fi
done

if [ -z "$MAIN_FILE" ]; then
    echo "Error: No source file found. Expected one of: main.ts, index.ts, main.js, index.js"
    exit 1
fi

echo "Source file: $MAIN_FILE"

# Detect language
if [[ "$MAIN_FILE" == *.ts ]]; then
    echo "Language: TypeScript"
else
    echo "Language: JavaScript"
fi

# Install dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "Installing dependencies..."
    mkdir -p node_modules
    npm install --production 2>&1 || echo "Warning: npm install had issues, continuing..."
fi

echo "Creating WIT import stubs..."

# Create stub implementations for WIT imports
# These are placeholders that get replaced by host implementations at runtime
mkdir -p stubs

# Console stub (logging)
cat > stubs/marie__node\$console.js << 'STUBEOF'
// Stub for marie:node/console - host-provided at runtime
export const LogLevel = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
export function log(level, message) {
    console.log(`[${Object.keys(LogLevel)[level] || 'INFO'}] ${message}`);
}
STUBEOF

# HTTP client stub
cat > stubs/marie__node\$http-client.js << 'STUBEOF'
// Stub for marie:node/http-client - host-provided at runtime
export function fetch(req) {
    throw new Error('HTTP client not available during compilation');
}
STUBEOF

# Secrets stub
cat > stubs/marie__node\$secrets.js << 'STUBEOF'
// Stub for marie:node/secrets - host-provided at runtime
export function get(name) {
    throw new Error('Secrets not available during compilation');
}
STUBEOF

# KV stub (key-value storage)
cat > stubs/marie__node\$kv.js << 'STUBEOF'
// Stub for marie:node/kv - host-provided at runtime
export function get(key) { return null; }
export function put(key, value, ttlSeconds) { }
export function del(key) { }
STUBEOF

# Events stub
cat > stubs/marie__node\$events.js << 'STUBEOF'
// Stub for marie:node/events - host-provided at runtime
export function emit(eventType, payload) { }
STUBEOF

echo "Bundling with esbuild..."

# Bundle with esbuild (handles both JS and TS natively)
# TypeScript is transpiled automatically
esbuild "$MAIN_FILE" \
    --bundle \
    --format=esm \
    --platform=neutral \
    --target=es2022 \
    --alias:'marie:node/console'=./stubs/marie__node\$console.js \
    --alias:'marie:node/http-client'=./stubs/marie__node\$http-client.js \
    --alias:'marie:node/secrets'=./stubs/marie__node\$secrets.js \
    --alias:'marie:node/kv'=./stubs/marie__node\$kv.js \
    --alias:'marie:node/events'=./stubs/marie__node\$events.js \
    --outfile=bundled.js \
    2>&1

if [ ! -f "bundled.js" ]; then
    echo "Error: esbuild bundling failed"
    exit 1
fi

echo "Compiling to Wasm component..."

# Compile bundled JS to component
jco componentize \
    bundled.js \
    --wit /opt/wit \
    --world-name node \
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
