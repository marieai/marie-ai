#!/bin/bash
# Install all packages in the monorepo for development

set -e

echo "============================================================"
echo "Installing Marie AI Monorepo Packages"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}Step 1/2: Installing main marie-ai package...${NC}"
cd "$REPO_ROOT"
pip install -e .
echo -e "${GREEN}✓ Main package installed${NC}"

echo ""
echo -e "${YELLOW}Step 2/2: Installing marie-mcp package...${NC}"
cd "$REPO_ROOT/packages/marie-mcp"
pip install -e ".[dev]"
echo -e "${GREEN}✓ MCP package installed${NC}"

echo ""
echo "============================================================"
echo -e "${GREEN}All packages installed successfully!${NC}"
echo "============================================================"
echo ""
echo "Installed packages:"
echo "  • marie-ai (editable mode)"
echo "  • marie-mcp (editable mode)"
echo ""
echo "You can now:"
echo "  • Run marie server: marie server --start --uses config/service/marie-dev.yml"
echo "  • Run MCP server: marie-mcp"
echo "  • Run tests: pytest tests/"
echo ""
