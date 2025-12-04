"""Example: Direct usage of Marie MCP client (without MCP framework)."""

import asyncio
from pathlib import Path

from marie_mcp.clients.marie_client import MarieClient
from marie_mcp.config import Config
from marie_mcp.utils.s3_utils import s3_asset_path, upload_to_s3


async def main():
    """Example direct usage of Marie client."""

    # Validate configuration
    Config.validate()

    # Create client
    print("Creating Marie client...")
    async with MarieClient() as client:

        # Example 1: Submit OCR extraction job
        print("\n" + "=" * 60)
        print("Example 1: OCR Extraction Job")
        print("=" * 60)

        # Prepare document
        file_path = "example_invoice.pdf"
        ref_id = "invoice_12345"
        ref_type = "invoice"

        # Upload to S3
        print(f"Uploading {file_path} to S3...")
        s3_path = s3_asset_path(ref_id, ref_type, include_filename=True)
        upload_to_s3(file_path, s3_path)
        print(f"Uploaded to: {s3_path}")

        # Build metadata
        from datetime import datetime, timedelta

        now = datetime.utcnow()

        metadata = {
            "on": "extract_executor://document/extract",
            "project_id": Config.MARIE_API_KEY,
            "ref_id": ref_id,
            "ref_type": ref_type,
            "uri": s3_path,
            "policy": "allow_all",
            "planner": "extract",
            "type": "pipeline",
            "name": "default",
            "page_classifier": {"enabled": False},
            "page_splitter": {"enabled": False},
            "page_cleaner": {"enabled": False},
            "page_boundary": {"enabled": False},
            "template_matching": {"enabled": False, "definition_id": "0"},
            "soft_sla": now.isoformat(),
            "hard_sla": (now + timedelta(hours=4)).isoformat(),
        }

        # Submit job
        print("Submitting OCR extraction job...")
        result = await client.submit_job(queue_name="extract", metadata=metadata)
        job_id = result.get("job_id")
        print(f"Job submitted! Job ID: {job_id}")

        # Example 2: Check job status
        print("\n" + "=" * 60)
        print("Example 2: Check Job Status")
        print("=" * 60)

        print(f"Fetching status for job {job_id}...")
        status = await client.get_job_status(job_id)
        print(f"Job status: {status}")

        # Example 3: List all jobs
        print("\n" + "=" * 60)
        print("Example 3: List Jobs")
        print("=" * 60)

        print("Fetching all active jobs...")
        jobs = await client.list_jobs(state="active")
        print(f"Active jobs: {jobs}")

        # Example 4: System monitoring
        print("\n" + "=" * 60)
        print("Example 4: System Monitoring")
        print("=" * 60)

        # Health check
        print("Performing health check...")
        health = await client.health_check()
        print(f"Health: {health}")

        # Capacity
        print("Getting capacity information...")
        capacity = await client.get_capacity()
        print(f"Capacity: {capacity}")

        # Deployments
        print("Getting deployment information...")
        deployments = await client.get_deployments()
        print(f"Deployments: {deployments}")

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
