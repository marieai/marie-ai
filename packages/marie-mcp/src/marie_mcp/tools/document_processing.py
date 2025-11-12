"""Document processing MCP tools."""

import json
from datetime import datetime, timedelta
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from ..clients.marie_client import MarieClient, MarieClientError
from ..utils.constants import PLANNER_EXTRACT, QUEUE_EXTRACT, QUEUE_GEN5_EXTRACT
from ..utils.s3_utils import s3_asset_path, upload_to_s3
from ..utils.validators import (
    ValidationError,
    validate_file,
    validate_ref_id,
    validate_template_id,
)


def register_document_tools(mcp: FastMCP, client: MarieClient) -> None:
    """Register document processing tools."""

    @mcp.tool()
    async def extract_document_ocr(
        file_path: Annotated[
            str, Field(description="Path to local document file (PDF, TIFF, JPEG, PNG)")
        ],
        ref_id: Annotated[
            str,
            Field(
                description="Unique document reference ID (e.g., filename or document ID)"
            ),
        ],
        ref_type: Annotated[
            str,
            Field(description="Document type/category (e.g., 'invoice', 'receipt')"),
        ] = "document",
        enable_page_classifier: Annotated[
            bool, Field(description="Enable page classification")
        ] = False,
        enable_page_splitter: Annotated[
            bool, Field(description="Enable automatic page splitting")
        ] = False,
        enable_template_matching: Annotated[
            bool, Field(description="Enable template matching")
        ] = False,
        sla_hours: Annotated[
            int, Field(description="Hard SLA deadline in hours (default: 4)")
        ] = 4,
        ctx: Context = None,
    ) -> str:
        """
        Submit an OCR extraction job to Marie AI.

        This uploads the document to S3 and submits a job for OCR text extraction
        with layout preservation. The job runs asynchronously and results are
        stored in S3.

        Returns job_id for tracking via get_job_status().

        Example:
            extract_document_ocr(
                file_path="/path/to/invoice.pdf",
                ref_id="invoice_12345",
                ref_type="invoice"
            )
        """
        try:
            # Step 1: Validate file
            if ctx:
                ctx.info(f"Validating file: {file_path}")
            validate_file(file_path, max_size_mb=50)
            validate_ref_id(ref_id)

            # Step 2: Upload to S3
            if ctx:
                ctx.info("Uploading document to S3...")
                await ctx.report_progress(20, 100)

            s3_path = s3_asset_path(ref_id, ref_type, include_filename=True)
            upload_success = upload_to_s3(file_path, s3_path)

            if not upload_success:
                raise Exception(f"Failed to upload {file_path} to {s3_path}")

            # Step 3: Build job metadata
            if ctx:
                ctx.info("Submitting OCR extraction job...")
                await ctx.report_progress(50, 100)

            now = datetime.utcnow()
            soft_sla = now
            hard_sla = now + timedelta(hours=sla_hours)

            metadata = {
                "on": "extract_executor://document/extract",
                "project_id": client.api_key,  # API key as project_id
                "ref_id": ref_id,
                "ref_type": ref_type,
                "uri": s3_path,
                "policy": "allow_all",
                "planner": PLANNER_EXTRACT,  # OCR extraction planner
                "type": "pipeline",
                "name": "default",
                "page_classifier": {"enabled": enable_page_classifier},
                "page_splitter": {"enabled": enable_page_splitter},
                "page_cleaner": {"enabled": False},
                "page_boundary": {"enabled": False},
                "template_matching": {
                    "enabled": enable_template_matching,
                    "definition_id": "0",
                },
                "soft_sla": soft_sla.isoformat(),
                "hard_sla": hard_sla.isoformat(),
            }

            # Step 4: Submit job
            result = await client.submit_job(
                queue_name=QUEUE_EXTRACT, metadata=metadata
            )

            if ctx:
                await ctx.report_progress(100, 100)
                job_id = result.get("job_id", "unknown")
                ctx.info(f"Job submitted successfully: {job_id}")

            return json.dumps(
                {
                    "status": "success",
                    "job_id": result.get("job_id"),
                    "s3_path": s3_path,
                    "ref_id": ref_id,
                    "ref_type": ref_type,
                    "message": "OCR extraction job submitted. Use get_job_status() to track progress.",
                },
                indent=2,
            )

        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except MarieClientError as e:
            error_msg = f"Marie API error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Failed to submit OCR extraction job: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    @mcp.tool()
    async def extract_document_data(
        file_path: Annotated[str, Field(description="Path to local document file")],
        template_id: Annotated[
            str,
            Field(
                description="Template ID / Planner ID for data extraction (e.g., '117183')"
            ),
        ],
        ref_id: Annotated[str, Field(description="Unique document reference ID")],
        ref_type: Annotated[
            str, Field(description="Document type/category")
        ] = "document",
        sla_hours: Annotated[int, Field(description="Hard SLA deadline in hours")] = 4,
        ctx: Context = None,
    ) -> str:
        """
        Submit a data extraction job to Marie AI using a specific template.

        This is for structured data extraction (Gen5) using predefined templates.
        The template_id determines which fields are extracted from the document.

        Returns job_id for tracking.

        Example:
            extract_document_data(
                file_path="/path/to/form.pdf",
                template_id="117183",
                ref_id="form_67890",
                ref_type="medical_form"
            )
        """
        try:
            if ctx:
                ctx.info(f"Validating file: {file_path}")
            validate_file(file_path, max_size_mb=50)
            validate_ref_id(ref_id)
            validate_template_id(template_id)

            # Upload to S3
            if ctx:
                ctx.info("Uploading document to S3...")
                await ctx.report_progress(20, 100)

            s3_path = s3_asset_path(ref_id, ref_type, include_filename=True)
            upload_success = upload_to_s3(file_path, s3_path)

            if not upload_success:
                raise Exception(f"Failed to upload {file_path} to {s3_path}")

            # Build metadata
            if ctx:
                ctx.info(f"Submitting data extraction job with template: {template_id}")
                await ctx.report_progress(50, 100)

            now = datetime.utcnow()
            soft_sla = now
            hard_sla = now + timedelta(hours=sla_hours)

            metadata = {
                "on": "extract_executor://document/extract",
                "project_id": client.api_key,
                "ref_id": ref_id,
                "ref_type": ref_type,
                "uri": s3_path,
                "policy": "allow_all",
                "planner": template_id,  # Template ID as planner
                "type": "pipeline",
                "name": "default",
                "page_classifier": {"enabled": False},
                "page_splitter": {"enabled": False},
                "page_cleaner": {"enabled": False},
                "page_boundary": {"enabled": False},
                "template_matching": {"enabled": False, "definition_id": "0"},
                "soft_sla": soft_sla.isoformat(),
                "hard_sla": hard_sla.isoformat(),
            }

            # Submit job
            result = await client.submit_job(
                queue_name=QUEUE_GEN5_EXTRACT, metadata=metadata
            )

            if ctx:
                await ctx.report_progress(100, 100)
                job_id = result.get("job_id", "unknown")
                ctx.info(f"Data extraction job submitted: {job_id}")

            return json.dumps(
                {
                    "status": "success",
                    "job_id": result.get("job_id"),
                    "template_id": template_id,
                    "s3_path": s3_path,
                    "ref_id": ref_id,
                    "ref_type": ref_type,
                    "message": "Data extraction job submitted. Use get_job_status() to track progress.",
                },
                indent=2,
            )

        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except MarieClientError as e:
            error_msg = f"Marie API error: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            error_msg = f"Failed to submit data extraction job: {str(e)}"
            if ctx:
                ctx.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
