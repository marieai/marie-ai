"""Marie MCP server entry point."""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .clients.marie_client import MarieClient
from .config import Config
from .tools.document_processing import register_document_tools
from .tools.job_management import register_job_tools
from .tools.system_monitoring import register_monitoring_tools


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[dict[str, MarieClient]]:
    """Create, use and close the gateway client in the server event loop."""
    Config.validate()
    print(Config.summary(), file=sys.stderr)
    async with MarieClient() as client:
        print('All tools registered successfully', file=sys.stderr)
        yield {'client': client}


def create_server() -> FastMCP:
    """Create an MCP server with managed gateway client lifetime."""
    server = FastMCP('marie-mcp', lifespan=server_lifespan)
    register_document_tools(server)
    register_job_tools(server)
    register_monitoring_tools(server)
    return server


mcp = create_server()


def main() -> None:
    """Run the server over STDIO."""
    print('Starting Marie MCP server via STDIO', file=sys.stderr)
    mcp.run()


if __name__ == '__main__':
    main()
