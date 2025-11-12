"""Job management MCP tools."""

import json
from typing import Annotated, Optional

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from ..clients.marie_client import MarieClient, MarieClientError
from ..utils.s3_utils import download_from_s3, s3_asset_path, s3_exists


def register_job_tools(mcp: FastMCP, client: MarieClient) -> None:
    """Register job management tools."""

    @mcp.tool()
    async def get_job_status(
        job_id: Annotated[str, Field(description="Job ID returned from submit_job")],
        ctx: Context = None,
    ) -> str:
        """
        Get detailed status information for a specific job.

        Returns job state, progress, and any error messages.

        Example:
            get_job_status(job_id="job_123456")
        """
        try:
            if ctx:
                ctx.info(f"Fetching status for job: {job_id}")

            result = await client.get_job_status(job_id)
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to get job status: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def list_jobs(
        state: Annotated[
            Optional[str],
            Field(
                description="Filter by state: 'created', 'pending', 'active', 'running', 'completed', 'failed', 'cancelled'"
            ),
        ] = None,
        ctx: Context = None,
    ) -> str:
        """
        List all jobs, optionally filtered by state.

        Returns a list of job objects with their current status.

        Example:
            list_jobs()  # All jobs
            list_jobs(state="active")  # Only active jobs
            list_jobs(state="completed")  # Only completed jobs
        """
        try:
            if ctx:
                ctx.info(f"Listing jobs (state={state or 'all'})")

            result = await client.list_jobs(state)
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to list jobs: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def stop_job(
        job_id: Annotated[str, Field(description="Job ID to stop")],
        ctx: Context = None,
    ) -> str:
        """
        Stop a running job.

        This will cancel the job execution if it's currently running.

        Example:
            stop_job(job_id="job_123456")
        """
        try:
            if ctx:
                ctx.info(f"Stopping job: {job_id}")

            result = await client.stop_job(job_id)
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to stop job: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def delete_job(
        job_id: Annotated[str, Field(description="Job ID to delete")],
        ctx: Context = None,
    ) -> str:
        """
        Delete a job from the scheduler.

        This removes the job and its metadata from the system.

        Example:
            delete_job(job_id="job_123456")
        """
        try:
            if ctx:
                ctx.info(f"Deleting job: {job_id}")

            result = await client.delete_job(job_id)
            return json.dumps(result, indent=2)

        except MarieClientError as e:
            error_msg = f"Failed to delete job: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def get_job_results(
        ref_id: Annotated[
            str, Field(description="Document reference ID used when submitting the job")
        ],
        ref_type: Annotated[
            str, Field(description="Document type used when submitting the job")
        ],
        output_path: Annotated[
            str,
            Field(
                description="Local path where results should be saved (e.g., './results/output.json')"
            ),
        ],
        ctx: Context = None,
    ) -> str:
        """
        Download job results from S3 storage.

        After a job completes, results are stored in S3. This tool downloads
        the results to your local machine.

        Example:
            get_job_results(
                ref_id="invoice_12345",
                ref_type="invoice",
                output_path="./results/invoice_12345.json"
            )
        """
        try:
            if ctx:
                ctx.info(f"Fetching results for ref_id={ref_id}, ref_type={ref_type}")

            # Generate S3 path for metadata file
            s3_base_path = s3_asset_path(ref_id, ref_type, include_filename=False)
            s3_meta_path = f"{s3_base_path}/{ref_id}.meta.json"

            if ctx:
                ctx.info(f"Checking S3 path: {s3_meta_path}")

            # Check if results exist
            if not s3_exists(s3_meta_path):
                error_msg = (
                    f"Results not found at {s3_meta_path}. "
                    "Job may still be running or failed."
                )
                if ctx:
                    ctx.error(error_msg)
                return json.dumps({"status": "error", "message": error_msg})

            # Download results
            if ctx:
                ctx.info(f"Downloading results to {output_path}")
                await ctx.report_progress(50, 100)

            success = download_from_s3(s3_meta_path, output_path)

            if not success:
                raise Exception(f"Failed to download from {s3_meta_path}")

            if ctx:
                await ctx.report_progress(100, 100)
                ctx.info(f"Results downloaded successfully to {output_path}")

            return json.dumps(
                {
                    "status": "success",
                    "output_path": output_path,
                    "s3_path": s3_meta_path,
                    "message": f"Results downloaded to {output_path}",
                },
                indent=2,
            )

        except Exception as e:
            error_msg = f"Failed to download results: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
