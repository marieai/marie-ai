"""Marie MCP Server - Main entry point."""

import sys

from mcp.server.fastmcp import FastMCP

from .clients.marie_client import MarieClient
from .config import Config
from .tools.document_processing import register_document_tools
from .tools.job_management import register_job_tools
from .tools.system_monitoring import register_monitoring_tools

# Initialize FastMCP server
mcp = FastMCP("marie-mcp")


def initialize_server() -> MarieClient:
    """
    Initialize the MCP server and register all tools.

    Returns:
        Configured MarieClient instance
    """
    try:
        # Validate configuration
        Config.validate()

        # Print configuration summary
        print(Config.summary(), file=sys.stderr)

        # Initialize Marie client
        client = MarieClient(
            base_url=Config.MARIE_BASE_URL,
            api_key=Config.MARIE_API_KEY,
            timeout=Config.REQUEST_TIMEOUT,
        )

        # Register all tools
        print("Registering MCP tools...", file=sys.stderr)
        register_document_tools(mcp, client)
        register_job_tools(mcp, client)
        register_monitoring_tools(mcp, client)
        print("✓ All tools registered successfully", file=sys.stderr)

        return client

    except Exception as e:
        print(f"ERROR: Failed to initialize server: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the MCP server."""
    print("=" * 60, file=sys.stderr)
    print("Marie MCP Server", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Initialize server and register tools
    client = initialize_server()

    print("\nStarting MCP server...", file=sys.stderr)
    print("Ready to accept connections via STDIO", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Run the MCP server
    try:
        mcp.run()
    finally:
        # Cleanup
        import asyncio

        asyncio.run(client.close())


if __name__ == "__main__":
    main()
