"""
HITL Correction Executor.

Pauses workflow execution and requests human correction of AI-extracted data.
"""

from typing import Any, Dict, Optional

from docarray import DocList
from docarray.documents import TextDoc

from marie import requests
from marie.executor.hitl.base_hitl_executor import BaseHitlExecutor


class HitlCorrectionExecutor(BaseHitlExecutor):
    """
    HITL Data Correction Executor.

    Pauses workflow execution and requests human correction of AI-extracted data.
    Supports:
    - Text correction
    - Structured data correction with field-level validation
    - Annotation/bounding box corrections
    - Classification corrections
    - Auto-validation for high-confidence fields
    - Side-by-side comparison UI

    Example Configuration:
        ```python
        {
            "title": "Correct Invoice Data Extraction",
            "description": "Review and correct extracted invoice fields",
            "correction_type": "structured",
            "fields": [
                {
                    "key": "invoice_number",
                    "label": "Invoice Number",
                    "type": "text",
                    "required": True,
                    "confidence_threshold": 0.9,
                },
                {
                    "key": "total_amount",
                    "label": "Total Amount",
                    "type": "number",
                    "required": True,
                    "confidence_threshold": 0.95,
                },
            ],
            "auto_validate": {
                "enabled": True,
                "confidence_threshold": 0.9,
                "exempt_fields": ["total_amount"],
            },
            "priority": "high",
        }
        ```
    """

    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        poll_interval: int = 5,
        max_poll_attempts: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize HITL Correction Executor.

        Args:
            db_config: Database configuration
            poll_interval: Seconds between polling for responses
            max_poll_attempts: Maximum polling attempts before timeout
            **kwargs: Additional executor arguments
        """
        super().__init__(
            db_config=db_config,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            **kwargs,
        )

    @requests(on="/hitl/correction")
    async def correction(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """
        Request human correction of extracted data and wait for response.

        Args:
            docs: Input documents with extracted data
            parameters: Request parameters including:
                - dag_id: DAG ID
                - job_id: Job ID
                - title: Correction request title
                - description: Request description (optional)
                - correction_type: Type (text, structured, annotation, classification)
                - priority: Priority level (low, medium, high, critical)
                - fields: Field definitions for structured data
                - original_data: Original extracted data
                - field_confidences: Confidence scores per field
                - auto_validate: Auto-validation configuration
                - timeout: Timeout configuration
                - assigned_to: User/role assignment
            **kwargs: Additional arguments

        Returns:
            DocList[TextDoc]: Documents with corrected data
        """
        self.logger.info("HITL Correction request received")

        # Extract parameters
        dag_id = parameters.get("dag_id")
        job_id = parameters.get("job_id")
        title = parameters.get("title", "Data Correction Required")
        description = parameters.get("description")
        correction_type = parameters.get("correction_type", "structured")
        priority = parameters.get("priority", "medium")
        fields = parameters.get("fields", [])
        original_data = parameters.get("original_data", {})
        field_confidences = parameters.get("field_confidences", {})
        auto_validate_config = parameters.get("auto_validate", {})
        timeout_config = parameters.get("timeout", {})
        assigned_to = parameters.get("assigned_to", {})

        # Prepare context data
        context_data = {
            "original_data": original_data,
            "field_confidences": field_confidences,
        }

        # Check for auto-validation
        validated_data = {}
        needs_review = False

        if auto_validate_config.get("enabled", False):
            threshold = auto_validate_config.get("confidence_threshold", 0.9)
            exempt_fields = set(auto_validate_config.get("exempt_fields", []))

            for field_key, field_value in original_data.items():
                confidence = field_confidences.get(field_key, 0)

                # Check if field can be auto-validated
                if field_key not in exempt_fields and confidence >= threshold:
                    # Auto-validate this field
                    validated_data[field_key] = field_value
                    self.logger.info(
                        f"Auto-validated field '{field_key}' (confidence: {confidence} >= {threshold})"
                    )
                else:
                    # Field needs human review
                    needs_review = True
                    self.logger.info(
                        f"Field '{field_key}' needs human review (confidence: {confidence}, exempt: {field_key in exempt_fields})"
                    )

            # If all fields auto-validated, return immediately
            if not needs_review:
                self.logger.info("All fields auto-validated - skipping human review")
                for doc in docs:
                    if not hasattr(doc, "tags"):
                        doc.tags = {}
                    doc.tags["hitl_corrected_data"] = validated_data
                    doc.tags["hitl_auto_validated"] = True
                return docs

        # Create HITL request in database
        config = {
            "correction_type": correction_type,
            "fields": fields,
            "auto_validate": auto_validate_config,
            "timeout": timeout_config,
            "assigned_to": assigned_to,
            "context_data": context_data,
        }

        timeout_seconds = None
        if timeout_config.get("enabled"):
            timeout_seconds = timeout_config.get("duration_seconds")

        request_id = await self.create_hitl_request(
            dag_id=dag_id,
            job_id=job_id,
            request_type="correction",
            title=title,
            description=description,
            priority=priority,
            context_data=context_data,
            config=config,
            timeout_seconds=timeout_seconds,
        )

        self.logger.info(f"HITL correction request created: {request_id}")
        self.logger.info("Waiting for human response...")

        # Poll for response
        response = await self.poll_for_response(request_id, timeout_seconds)

        # Process response
        corrected_data = response.get("corrected_data", {})
        feedback = response.get("feedback")
        status = response.get("status")

        self.logger.info(
            f"Received response for request {request_id}: {len(corrected_data)} fields corrected, status={status}"
        )

        # Merge auto-validated data with human corrections
        final_data = {**validated_data, **corrected_data}

        # Add corrected data to documents
        for doc in docs:
            if not hasattr(doc, "tags"):
                doc.tags = {}
            doc.tags["hitl_request_id"] = request_id
            doc.tags["hitl_corrected_data"] = final_data
            doc.tags["hitl_original_data"] = original_data
            doc.tags["hitl_feedback"] = feedback
            doc.tags["hitl_status"] = status
            doc.tags["hitl_auto_validated"] = False
            doc.tags["hitl_auto_validated_fields"] = list(validated_data.keys())

        return docs

    @requests(on="/default")
    async def default(
        self, docs: DocList[TextDoc], parameters: Dict[str, Any], **kwargs
    ):
        """Default endpoint - delegates to correction."""
        return await self.correction(docs, parameters, **kwargs)
