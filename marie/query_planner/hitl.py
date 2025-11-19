"""
Human-in-the-Loop (HITL) Query Definitions for Marie-AI.

This module provides query definitions for HITL workflow nodes that pause
execution for human review and decision-making.

HITL nodes enable workflows to:
- Request approval from human reviewers
- Collect data corrections from domain experts
- Route tasks based on AI confidence scores

The HITL system integrates with Marie Studio's frontend for a complete
human-in-the-loop workflow experience.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from marie.query_planner.base import QueryDefinition, QueryTypeRegistry


@QueryTypeRegistry.register("HITL_APPROVAL")
class HitlApprovalQueryDefinition(QueryDefinition):
    """
    HITL Approval Query Definition.

    Pauses workflow execution and requests human approval. Supports:
    - Binary approval (approve/reject)
    - Single choice selection
    - Multi choice selection
    - Ranked choice selection
    - Auto-approval based on confidence threshold
    - Timeout handling with configurable strategies
    - Multi-user approval requirements

    Example:
        ```python
        approval_def = HitlApprovalQueryDefinition(
            endpoint="hitl/approval",
            title="Approve Invoice Classification",
            description="Review AI classification of this invoice",
            approval_type="binary",
            priority="high",
            auto_approve={"enabled": True, "confidence_threshold": 0.95},
            timeout={
                "enabled": True,
                "duration_seconds": 86400,  # 24 hours
                "strategy": "use_default",
            },
            assigned_to={"user_ids": ["user-123"], "roles": ["finance_manager"]},
        )
        ```
    """

    method: str = "HITL_APPROVAL"
    endpoint: str = "hitl/approval"

    # HITL Request Configuration
    title: str = Field(..., description="Title of the approval request")
    description: Optional[str] = Field(
        None, description="Detailed description for reviewers"
    )
    approval_type: str = Field(
        "binary",
        description="Type of approval: binary, single_choice, multi_choice, ranked_choice",
    )
    priority: str = Field(
        "medium", description="Priority level: low, medium, high, critical"
    )

    # Options for choice-based approvals
    options: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of options for choice-based approvals"
    )
    default_option: Optional[str] = Field(
        None, description="Default option if timeout occurs"
    )

    # Auto-approval configuration
    auto_approve: Optional[Dict[str, Any]] = Field(
        None, description="Auto-approval settings based on confidence threshold"
    )

    # Timeout configuration
    timeout: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "duration_seconds": 86400,
            "strategy": "use_default",
        },
        description="Timeout configuration",
    )

    # Assignment configuration
    assigned_to: Optional[Dict[str, Any]] = Field(
        None, description="User/role assignment configuration"
    )

    # Notification configuration
    notifications: Dict[str, Any] = Field(
        default_factory=lambda: {
            "channels": ["email", "in_app"],
            "on_request": True,
            "on_reminder": True,
            "reminder_interval_seconds": 3600,
            "max_reminders": 3,
        },
        description="Notification settings",
    )

    # UI customization
    ui: Optional[Dict[str, Any]] = Field(None, description="UI customization options")

    params: dict = Field(
        default_factory=lambda: {"layout": None, "context_data": {}, "confidence": None}
    )

    def validate_params(self):
        """Validate HITL approval parameters."""
        if not self.title:
            raise ValueError("HITL approval requests must have a title")

        if self.approval_type not in [
            "binary",
            "single_choice",
            "multi_choice",
            "ranked_choice",
        ]:
            raise ValueError(f"Invalid approval_type: {self.approval_type}")

        if self.approval_type in ["single_choice", "multi_choice", "ranked_choice"]:
            if not self.options or len(self.options) == 0:
                raise ValueError(f"Approval type {self.approval_type} requires options")

        if self.priority not in ["low", "medium", "high", "critical"]:
            raise ValueError(f"Invalid priority: {self.priority}")

        # Validate auto-approve configuration
        if self.auto_approve and self.auto_approve.get("enabled"):
            threshold = self.auto_approve.get("confidence_threshold", 0.95)
            if not 0 <= threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1")

        # Validate timeout configuration
        if self.timeout.get("enabled"):
            if self.timeout.get("duration_seconds", 0) <= 0:
                raise ValueError("timeout duration_seconds must be positive")
            if self.timeout.get("strategy") not in [
                "use_default",
                "fail",
                "escalate",
                "skip_downstream",
            ]:
                raise ValueError(
                    f"Invalid timeout strategy: {self.timeout.get('strategy')}"
                )


@QueryTypeRegistry.register("HITL_CORRECTION")
class HitlCorrectionQueryDefinition(QueryDefinition):
    """
    HITL Data Correction Query Definition.

    Pauses workflow execution and requests human correction of AI-extracted data.
    Supports:
    - Text correction
    - Structured data correction with field-level validation
    - Annotation/bounding box corrections
    - Classification corrections
    - Auto-validation for high-confidence fields
    - Side-by-side comparison UI

    Example:
        ```python
        correction_def = HitlCorrectionQueryDefinition(
            endpoint="hitl/correction",
            title="Correct Invoice Data Extraction",
            description="Review and correct extracted invoice fields",
            correction_type="structured",
            fields=[
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
            auto_validate={
                "enabled": True,
                "confidence_threshold": 0.9,
                "exempt_fields": ["total_amount"],  # Always require human review
            },
            priority="high",
        )
        ```
    """

    method: str = "HITL_CORRECTION"
    endpoint: str = "hitl/correction"

    # HITL Request Configuration
    title: str = Field(..., description="Title of the correction request")
    description: Optional[str] = Field(
        None, description="Detailed description for reviewers"
    )
    correction_type: str = Field(
        "structured", description="Type: text, structured, annotation, classification"
    )
    priority: str = Field(
        "medium", description="Priority level: low, medium, high, critical"
    )

    # Field definitions for structured correction
    fields: Optional[List[Dict[str, Any]]] = Field(
        None, description="Field definitions for structured data correction"
    )

    # Auto-validation configuration
    auto_validate: Optional[Dict[str, Any]] = Field(
        None, description="Auto-validate high-confidence fields without human review"
    )

    # Timeout configuration
    timeout: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "duration_seconds": 86400,
            "strategy": "use_original",
        },
        description="Timeout configuration",
    )

    # Assignment configuration
    assigned_to: Optional[Dict[str, Any]] = Field(
        None, description="User/role assignment configuration"
    )

    # Notification configuration
    notifications: Dict[str, Any] = Field(
        default_factory=lambda: {"channels": ["email", "in_app"], "on_request": True},
        description="Notification settings",
    )

    # UI customization
    ui: Optional[Dict[str, Any]] = Field(
        None,
        description="UI customization (side-by-side, highlight low confidence, etc.)",
    )

    params: dict = Field(
        default_factory=lambda: {
            "layout": None,
            "original_data": {},
            "field_confidences": {},
        }
    )

    def validate_params(self):
        """Validate HITL correction parameters."""
        if not self.title:
            raise ValueError("HITL correction requests must have a title")

        if self.correction_type not in [
            "text",
            "structured",
            "annotation",
            "classification",
        ]:
            raise ValueError(f"Invalid correction_type: {self.correction_type}")

        if self.correction_type == "structured":
            if not self.fields or len(self.fields) == 0:
                raise ValueError("Structured correction requires field definitions")

            # Validate field definitions
            for field in self.fields:
                if "key" not in field or "label" not in field or "type" not in field:
                    raise ValueError("Each field must have key, label, and type")

                if field["type"] not in [
                    "text",
                    "number",
                    "date",
                    "enum",
                    "boolean",
                    "json",
                ]:
                    raise ValueError(f"Invalid field type: {field['type']}")

        if self.priority not in ["low", "medium", "high", "critical"]:
            raise ValueError(f"Invalid priority: {self.priority}")


@QueryTypeRegistry.register("HITL_ROUTER")
class HitlRouterQueryDefinition(QueryDefinition):
    """
    HITL Confidence Router Query Definition.

    Routes workflow execution based on AI confidence scores without pausing.
    This is a non-blocking HITL node that routes data through different paths:
    - High confidence (>= auto_approve_threshold) → Auto approve path
    - Medium confidence (>= human_review_threshold) → Human review path
    - Low confidence (< human_review_threshold) → Reject path

    Example:
        ```python
        router_def = HitlRouterQueryDefinition(
            endpoint="hitl/router",
            auto_approve_threshold=0.95,
            human_review_threshold=0.7,
            below_threshold_action="review",  # or "reject"
            always_review_if=[
                {
                    "field": "total_amount",
                    "condition": "gt",
                    "value": 10000,  # Always review invoices > $10k
                }
            ],
        )
        ```
    """

    method: str = "HITL_ROUTER"
    endpoint: str = "hitl/router"

    # Routing thresholds
    auto_approve_threshold: float = Field(
        0.95, description="Threshold for auto-approval (0-1)"
    )
    human_review_threshold: float = Field(
        0.7, description="Threshold for human review (0-1)"
    )
    below_threshold_action: str = Field(
        "review", description="Action for below threshold: review or reject"
    )

    # Conditional routing rules
    always_review_if: Optional[List[Dict[str, Any]]] = Field(
        None, description="Force human review if conditions match"
    )

    params: dict = Field(
        default_factory=lambda: {
            "layout": None,
            "data": {},
            "confidence": None,
            "metadata": {},
        }
    )

    def validate_params(self):
        """Validate HITL router parameters."""
        if not 0 <= self.auto_approve_threshold <= 1:
            raise ValueError("auto_approve_threshold must be between 0 and 1")

        if not 0 <= self.human_review_threshold <= 1:
            raise ValueError("human_review_threshold must be between 0 and 1")

        if self.auto_approve_threshold < self.human_review_threshold:
            raise ValueError("auto_approve_threshold must be >= human_review_threshold")

        if self.below_threshold_action not in ["review", "reject"]:
            raise ValueError("below_threshold_action must be 'review' or 'reject'")

        # Validate conditional rules
        if self.always_review_if:
            valid_conditions = ["exists", "empty", "equals", "gt", "lt"]
            for rule in self.always_review_if:
                if "field" not in rule or "condition" not in rule:
                    raise ValueError("Each rule must have 'field' and 'condition'")
                if rule["condition"] not in valid_conditions:
                    raise ValueError(f"Invalid condition: {rule['condition']}")


__all__ = [
    "HitlApprovalQueryDefinition",
    "HitlCorrectionQueryDefinition",
    "HitlRouterQueryDefinition",
]
