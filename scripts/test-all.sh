#!/bin/bash
# Run tests for all packages in the monorepo

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "Running Tests for All Packages"
echo "============================================================"

FAILED=0

# Test main package
echo ""
echo -e "${YELLOW}Testing marie-ai...${NC}"
cd "$REPO_ROOT"
if pytest tests/ -v; then
    echo -e "${GREEN}✓ marie-ai tests passed${NC}"
else
    echo -e "${RED}✗ marie-ai tests failed${NC}"
    FAILED=1
fi

# Test MCP package
echo ""
echo -e "${YELLOW}Testing marie-mcp...${NC}"
cd "$REPO_ROOT/packages/marie-mcp"
if pytest tests/ -v; then
    echo -e "${GREEN}✓ marie-mcp tests passed${NC}"
else
    echo -e "${RED}✗ marie-mcp tests failed${NC}"
    FAILED=1
fi

echo ""
echo "============================================================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
