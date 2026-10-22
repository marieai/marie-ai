"""Helpers shared by MCP tools."""

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ToolError

from ..clients.marie_client import MarieClient


def get_request_client(client: MarieClient | None, ctx: Context | None) -> MarieClient:
    """Use an explicit client or the client owned by the current server lifespan."""
    if client is not None:
        return client
    if ctx is None:
        raise ToolError('Gateway client is unavailable outside the server lifespan')
    return ctx.request_context.lifespan_context['client']
