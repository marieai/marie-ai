"""Job inspection and result download tools."""

import asyncio
import json
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from ..clients.marie_client import MarieClient
from ..utils.file_utils import resolve_output_path
from ..utils.s3_utils import download_from_s3, s3_asset_path, s3_exists
from . import get_request_client


def register_job_tools(mcp: FastMCP, client: MarieClient | None = None) -> None:
    """Register supported job tools; gateway stop/delete routes are placeholders."""

    @mcp.tool()
    async def get_job_status(
        job_id: Annotated[str, Field(description='Job UUID returned from submission')],
        ctx: Context = None,
    ) -> str:
        """Get payload-free lifecycle details for a job."""
        try:
            request_client = get_request_client(client, ctx)
            if ctx:
                await ctx.info(f'Fetching status for job: {job_id}')
            return json.dumps(await request_client.get_job_status(job_id), indent=2)
        except Exception as error:
            message = f'Failed to get job status: {error}'
            if ctx:
                await ctx.error(message)
            raise ToolError(message) from error

    @mcp.tool()
    async def list_jobs(
        state: Annotated[
            str | None,
            Field(
                description='Optional scheduler state, such as active, completed or failed'
            ),
        ] = None,
        ctx: Context = None,
    ) -> str:
        """List jobs, optionally filtered by scheduler state."""
        try:
            request_client = get_request_client(client, ctx)
            if ctx:
                await ctx.info(f'Listing jobs (state={state or "all"})')
            return json.dumps(await request_client.list_jobs(state), indent=2)
        except Exception as error:
            message = f'Failed to list jobs: {error}'
            if ctx:
                await ctx.error(message)
            raise ToolError(message) from error

    @mcp.tool()
    async def get_job_results(
        ref_id: Annotated[
            str, Field(description='Document reference ID used for submission')
        ],
        ref_type: Annotated[
            str, Field(description='Document type used for submission')
        ],
        output_path: Annotated[
            str,
            Field(
                description='Destination within MARIE_OUTPUT_DIR; relative paths start there'
            ),
        ],
        ctx: Context = None,
    ) -> str:
        """Download completed job metadata from S3 to the configured output directory."""
        try:
            destination = resolve_output_path(output_path)
            s3_base_path = s3_asset_path(ref_id, ref_type, include_filename=False)
            s3_meta_path = f'{s3_base_path}/{ref_id}.meta.json'
            if ctx:
                await ctx.info(f'Checking results at {s3_meta_path}')
            if not await asyncio.to_thread(s3_exists, s3_meta_path):
                raise ToolError(
                    'Results are not available; the job may still be running or failed'
                )
            if ctx:
                await ctx.report_progress(50, 100)
            destination.parent.mkdir(parents=True, exist_ok=True)
            success = await asyncio.to_thread(
                download_from_s3, s3_meta_path, str(destination)
            )
            if not success:
                raise ToolError('Result download failed')
            if ctx:
                await ctx.report_progress(100, 100)
                await ctx.info(f'Results downloaded to {destination}')
            return json.dumps(
                {
                    'status': 'success',
                    'output_path': str(destination),
                    's3_path': s3_meta_path,
                    'message': f'Results downloaded to {destination}',
                },
                indent=2,
            )
        except Exception as error:
            message = f'Failed to download results: {error}'
            if ctx:
                await ctx.error(message)
            raise ToolError(message) from error
