#!/bin/bash
# Format code in all packages

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "Formatting Code for All Packages"
echo "============================================================"

echo ""
echo -e "${YELLOW}Installing formatters...${NC}"
pip install black isort -q

echo ""
echo -e "${YELLOW}Formatting main package...${NC}"
cd "$REPO_ROOT"
black marie/
isort marie/
echo -e "${GREEN}✓ Main package formatted${NC}"

echo ""
echo -e "${YELLOW}Formatting marie-mcp package...${NC}"
cd "$REPO_ROOT/packages/marie-mcp"
black src/
isort src/
echo -e "${GREEN}✓ MCP package formatted${NC}"

echo ""
echo "============================================================"
echo -e "${GREEN}All code formatted!${NC}"
echo "============================================================"
