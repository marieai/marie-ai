"""
HITL Router Executor.

Routes workflow execution based on AI confidence scores without pausing.
"""

from typing import Any, Dict, Optional

from docarray import DocList
from docarray.documents import TextDoc

from marie import requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger


class HitlRouterExecutor(MarieExecutor):
    """
    HITL Confidence Router Executor.

    Routes workflow execution based on AI confidence scores without pausing.
    This is a non-blocking HITL node that routes data through different paths:
    - High confidence (>= auto_approve_threshold) → Auto approve path
    - Medium confidence (>= human_review_threshold) → Human review path
    - Low confidence (< human_review_threshold) → Reject path

    Unlike approval/correction executors, this doesn't create database requests
    or wait for human input - it makes immediate routing decisions based on
    confidence and conditional rules.

    Example Configuration:
        ```python
        {
            "auto_approve_threshold": 0.95,
            "human_review_threshold": 0.7,
            "below_threshold_action": "review",  # or "reject"
            "always_review_if": [{"field": "total_amount", "condition": "gt", "value": 10000}],
        }
        ```
    """

    def __init__(self, **kwargs):
        """
        Initialize HITL Router Executor.

        Args:
            **kwargs: Additional executor arguments
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("HITL Router Executor initialized")

    def _evaluate_condition(
        self, data: Dict[str, Any], condition: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a conditional routing rule.

        Args:
            data: Data to evaluate against
            condition: Condition configuration with:
                - field: Field name to check
                - condition: Condition type (exists, empty, equals, gt, lt, gte, lte)
                - value: Value to compare against (optional)

        Returns:
            bool: True if condition matches
        """
        field = condition.get("field")
        condition_type = condition.get("condition")
        expected_value = condition.get("value")

        # Get field value from data
        field_value = data.get(field)

        # Evaluate condition
        if condition_type == "exists":
            return field_value is not None
        elif condition_type == "empty":
            return field_value is None or field_value == "" or field_value == []
        elif condition_type == "equals":
            return field_value == expected_value
        elif condition_type == "gt":
            try:
                return float(field_value) > float(expected_value)
            except (TypeError, ValueError):
                return False
        elif condition_type == "lt":
            try:
                return float(field_value) < float(expected_value)
            except (TypeError, ValueError):
                return False
        elif condition_type == "gte":
            try:
                return float(field_value) >= float(expected_value)
            except (TypeError, ValueError):
                return False
        elif condition_type == "lte":
            try:
                return float(field_value) <= float(expected_value)
            except (TypeError, ValueError):
                return False
        else:
            self.logger.warning(f"Unknown condition type: {condition_type}")
            return False

    @requests(on="/hitl/router")
    async def router(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """
        Route documents based on confidence and conditional rules.

        Args:
            docs: Input documents
            parameters: Router parameters including:
                - auto_approve_threshold: Threshold for auto-approval (0-1)
                - human_review_threshold: Threshold for human review (0-1)
                - below_threshold_action: Action for below threshold (review or reject)
                - always_review_if: Conditional routing rules
                - data: Data to evaluate conditions against
                - confidence: Overall confidence score
            **kwargs: Additional arguments

        Returns:
            DocList[TextDoc]: Documents with routing decision added
        """
        self.logger.info("HITL Router request received")

        # Extract parameters
        auto_approve_threshold = parameters.get("auto_approve_threshold", 0.95)
        human_review_threshold = parameters.get("human_review_threshold", 0.7)
        below_threshold_action = parameters.get("below_threshold_action", "review")
        always_review_if = parameters.get("always_review_if", [])
        data = parameters.get("data", {})
        confidence = parameters.get("confidence")

        # Validate confidence
        if confidence is None:
            self.logger.warning(
                "No confidence score provided - routing to human review"
            )
            routing_decision = "human_review"
            routing_reason = "No confidence score provided"
        else:
            # Check conditional routing rules first
            should_force_review = False
            force_review_reason = None

            for rule in always_review_if:
                if self._evaluate_condition(data, rule):
                    should_force_review = True
                    force_review_reason = f"Condition matched: {rule.get('field')} {rule.get('condition')} {rule.get('value')}"
                    self.logger.info(f"Forcing human review: {force_review_reason}")
                    break

            if should_force_review:
                routing_decision = "human_review"
                routing_reason = force_review_reason
            else:
                # Route based on confidence thresholds
                if confidence >= auto_approve_threshold:
                    routing_decision = "auto_approve"
                    routing_reason = (
                        f"Confidence {confidence:.2%} >= {auto_approve_threshold:.2%}"
                    )
                    self.logger.info(f"Auto-approving: {routing_reason}")
                elif confidence >= human_review_threshold:
                    routing_decision = "human_review"
                    routing_reason = (
                        f"Confidence {confidence:.2%} >= {human_review_threshold:.2%}"
                    )
                    self.logger.info(f"Routing to human review: {routing_reason}")
                else:
                    # Below minimum threshold
                    if below_threshold_action == "review":
                        routing_decision = "human_review"
                        routing_reason = f"Confidence {confidence:.2%} < {human_review_threshold:.2%} - routing to review"
                        self.logger.info(
                            f"Low confidence, routing to review: {routing_reason}"
                        )
                    else:  # reject
                        routing_decision = "reject"
                        routing_reason = f"Confidence {confidence:.2%} < {human_review_threshold:.2%} - rejecting"
                        self.logger.info(f"Low confidence, rejecting: {routing_reason}")

        # Add routing decision to documents
        for doc in docs:
            if not hasattr(doc, "tags"):
                doc.tags = {}
            doc.tags["hitl_routing_decision"] = routing_decision
            doc.tags["hitl_routing_reason"] = routing_reason
            doc.tags["hitl_confidence"] = confidence
            doc.tags["hitl_auto_approve_threshold"] = auto_approve_threshold
            doc.tags["hitl_human_review_threshold"] = human_review_threshold

        self.logger.info(f"Routing decision: {routing_decision} ({routing_reason})")
        return docs

    @requests(on="/default")
    async def default(
        self, docs: DocList[TextDoc], parameters: Dict[str, Any], **kwargs
    ):
        """Default endpoint - delegates to router."""
        return await self.router(docs, parameters, **kwargs)
