#!/bin/bash
# Build mock_planners wheel for Docker testing
# This packages the test mock planners as a wheel that can be loaded by the gateway
#
# Usage: ./scripts/build_mock_planners_wheel.sh [output_dir]
# Default output: ./dist/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_ROOT/dist}"
BUILD_DIR="$PROJECT_ROOT/build/mock_planners_wheel"
VERSION="0.1.0"

echo "Building mock_planners wheel..."
echo "  Project root: $PROJECT_ROOT"
echo "  Output dir: $OUTPUT_DIR"

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create package structure matching the import path: tests.integration.scheduler.mock_query_plans
mkdir -p "$BUILD_DIR/tests/integration/scheduler/mock_plans"

# Create __init__.py files for each level
touch "$BUILD_DIR/tests/__init__.py"
touch "$BUILD_DIR/tests/integration/__init__.py"
touch "$BUILD_DIR/tests/integration/scheduler/__init__.py"

# Copy the mock planners source files
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_query_plans.py" "$BUILD_DIR/tests/integration/scheduler/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/__init__.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/base.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/branching.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/guardrail.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/hitl.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"
cp "$PROJECT_ROOT/tests/integration/scheduler/mock_plans/traditional.py" "$BUILD_DIR/tests/integration/scheduler/mock_plans/"

# Create pyproject.toml
cat > "$BUILD_DIR/pyproject.toml" << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "marie-mock-planners"
version = "0.1.0"
description = "Mock query planners for Marie-AI testing"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "marie-ai",  # Depends on marie for query planner base classes
]

[project.optional-dependencies]
dev = []

[tool.setuptools.packages.find]
where = ["."]
include = ["tests*"]
EOF

# Create README
cat > "$BUILD_DIR/README.md" << 'EOF'
# Marie Mock Planners

Mock query planners for testing the Marie-AI scheduler and gateway.

## Installation

```bash
pip install marie-mock-planners-0.1.0-py3-none-any.whl
```

## Usage

The planners are auto-registered via the `@register_query_plan` decorator when imported.

Configure in gateway YAML:
```yaml
query_planners:
  planners:
    - name: mock_planners
      py_module: tests.integration.scheduler.mock_query_plans
```

## Available Plans

- Traditional: mock_simple, mock_medium, mock_complex, mock_with_subgraphs, mock_parallel_subgraphs
- Branching: mock_branch_simple, mock_switch_complexity, mock_branch_multi_condition, etc.
- Guardrail: mock_guardrail_simple, mock_guardrail_retry_loop, etc.
- HITL: mock_hitl_approval, mock_hitl_correction, mock_hitl_router, etc.
EOF

# Build the wheel
echo "Building wheel..."
cd "$BUILD_DIR"
pip wheel . --no-deps -w dist/

# Copy to output directory
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/dist/"*.whl "$OUTPUT_DIR/"

WHEEL_FILE=$(ls "$OUTPUT_DIR"/marie_mock_planners-*.whl 2>/dev/null | head -1)

echo ""
echo "Wheel built successfully!"
echo "  Output: $WHEEL_FILE"
echo ""
echo "To install in Docker, copy to your wheels directory:"
echo "  cp $WHEEL_FILE /mnt/data/marie-ai/config/wheels/"
echo ""
echo "Or install directly:"
echo "  pip install $WHEEL_FILE"
