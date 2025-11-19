"""
HITL Approval Executor.

Pauses workflow execution and requests human approval.
"""

from typing import Any, Dict, Optional

from docarray import DocList
from docarray.documents import TextDoc

from marie import requests
from marie.executor.hitl.base_hitl_executor import BaseHitlExecutor


class HitlApprovalExecutor(BaseHitlExecutor):
    """
    HITL Approval Executor.

    Pauses workflow execution and requests human approval. Supports:
    - Binary approval (approve/reject)
    - Single choice selection
    - Multi choice selection
    - Ranked choice selection
    - Auto-approval based on confidence threshold
    - Timeout handling with configurable strategies

    Example Configuration:
        ```python
        {
            "title": "Approve Invoice Classification",
            "description": "Review AI classification of this invoice",
            "approval_type": "binary",
            "priority": "high",
            "auto_approve": {"enabled": True, "confidence_threshold": 0.95},
            "timeout": {"enabled": True, "duration_seconds": 86400, "strategy": "use_default"},
            "assigned_to": {"user_ids": ["user-123"], "roles": ["finance_manager"]},
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
        Initialize HITL Approval Executor.

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

    @requests(on="/hitl/approval")
    async def approval(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """
        Request human approval and wait for response.

        Args:
            docs: Input documents
            parameters: Request parameters including:
                - dag_id: DAG ID
                - job_id: Job ID
                - title: Approval request title
                - description: Request description (optional)
                - approval_type: Type (binary, single_choice, multi_choice, ranked_choice)
                - priority: Priority level (low, medium, high, critical)
                - context_data: Context information to display
                - options: Choice options (for choice-based approvals)
                - auto_approve: Auto-approval configuration
                - timeout: Timeout configuration
                - assigned_to: User/role assignment
            **kwargs: Additional arguments

        Returns:
            DocList[TextDoc]: Documents with approval decision added
        """
        self.logger.info("HITL Approval request received")

        # Extract parameters
        dag_id = parameters.get("dag_id")
        job_id = parameters.get("job_id")
        title = parameters.get("title", "Approval Required")
        description = parameters.get("description")
        approval_type = parameters.get("approval_type", "binary")
        priority = parameters.get("priority", "medium")
        context_data = parameters.get("context_data", {})
        options = parameters.get("options", [])
        auto_approve_config = parameters.get("auto_approve", {})
        timeout_config = parameters.get("timeout", {})
        assigned_to = parameters.get("assigned_to", {})

        # Check for auto-approval
        if auto_approve_config.get("enabled", False):
            confidence = context_data.get("confidence")
            threshold = auto_approve_config.get("confidence_threshold", 0.95)

            if confidence is not None and confidence >= threshold:
                self.logger.info(
                    f"Auto-approving request (confidence: {confidence} >= {threshold})"
                )
                # Add auto-approval decision to docs
                for doc in docs:
                    if not hasattr(doc, "tags"):
                        doc.tags = {}
                    doc.tags["hitl_decision"] = "approve"
                    doc.tags["hitl_auto_approved"] = True
                    doc.tags["hitl_confidence"] = confidence
                return docs

        # Create HITL request in database
        config = {
            "approval_type": approval_type,
            "options": options,
            "auto_approve": auto_approve_config,
            "timeout": timeout_config,
            "assigned_to": assigned_to,
        }

        timeout_seconds = None
        if timeout_config.get("enabled"):
            timeout_seconds = timeout_config.get("duration_seconds")

        request_id = await self.create_hitl_request(
            dag_id=dag_id,
            job_id=job_id,
            request_type="approval",
            title=title,
            description=description,
            priority=priority,
            context_data=context_data,
            config=config,
            timeout_seconds=timeout_seconds,
        )

        self.logger.info(f"HITL approval request created: {request_id}")
        self.logger.info("Waiting for human response...")

        # Poll for response
        response = await self.poll_for_response(request_id, timeout_seconds)

        # Process response
        decision = response.get("decision")
        feedback = response.get("feedback")
        status = response.get("status")

        self.logger.info(
            f"Received response for request {request_id}: decision={decision}, status={status}"
        )

        # Add decision to documents
        for doc in docs:
            if not hasattr(doc, "tags"):
                doc.tags = {}
            doc.tags["hitl_request_id"] = request_id
            doc.tags["hitl_decision"] = decision
            doc.tags["hitl_feedback"] = feedback
            doc.tags["hitl_status"] = status
            doc.tags["hitl_auto_approved"] = False

        return docs

    @requests(on="/default")
    async def default(
        self, docs: DocList[TextDoc], parameters: Dict[str, Any], **kwargs
    ):
        """Default endpoint - delegates to approval."""
        return await self.approval(docs, parameters, **kwargs)
