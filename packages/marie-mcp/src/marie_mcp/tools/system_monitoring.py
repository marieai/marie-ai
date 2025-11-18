"""System monitoring MCP tools."""

import json
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from ..clients.marie_client import MarieClient, MarieClientError


def register_monitoring_tools(mcp: FastMCP, client: MarieClient) -> None:
    """Register system monitoring tools."""

    @mcp.tool()
    async def get_deployments(ctx: Context = None) -> str:
        """
        Get all deployments and executor information.

        Returns information about active executors, their configuration,
        and deployment nodes.

        Example:
            get_deployments()
        """
        try:
            if ctx:
                ctx.info("Fetching deployment information")

            result = await client.get_deployments()
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to get deployments: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def get_capacity(ctx: Context = None) -> str:
        """
        Get slot capacity and availability information.

        Returns information about available processing slots, capacity limits,
        and current resource utilization.

        Example:
            get_capacity()
        """
        try:
            if ctx:
                ctx.info("Fetching capacity information")

            result = await client.get_capacity()
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to get capacity: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def get_debug_info(ctx: Context = None) -> str:
        """
        Get scheduler debug information.

        Returns detailed debug information from the job scheduler including
        active DAGs, execution state, and internal metrics.

        Example:
            get_debug_info()
        """
        try:
            if ctx:
                ctx.info("Fetching scheduler debug information")

            result = await client.get_debug_info()
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to get debug info: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def health_check(ctx: Context = None) -> str:
        """
        Perform a health check on the Marie gateway.

        Returns basic health status of the gateway service.

        Example:
            health_check()
        """
        try:
            if ctx:
                ctx.info("Performing health check")

            result = await client.health_check()
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Health check failed: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
