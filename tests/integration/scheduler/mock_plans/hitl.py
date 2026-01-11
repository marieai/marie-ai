"""
Human-in-the-Loop (HITL) Mock Query Plans

HITL node patterns for human review, approval, and correction workflows.

Plans:
    - query_planner_mock_hitl_approval: HITL Approval workflow
    - query_planner_mock_hitl_correction: HITL Data Correction workflow
    - query_planner_mock_hitl_router: Confidence-based routing workflow
    - query_planner_mock_hitl_complete_workflow: Complete HITL workflow with all types
"""

from .base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
    increment_uuid7str,
    register_query_plan,
)


@register_query_plan("mock_hitl_approval")
def query_planner_mock_hitl_approval(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Mock query plan demonstrating HITL Approval workflow.

    Structure:
        START -> CLASSIFY_DOCUMENT -> HITL_APPROVAL -> PROCESS_APPROVED -> END

    This plan demonstrates:
    - AI classification of a document
    - HITL approval request for human verification
    - Auto-approval based on confidence threshold
    - Timeout handling
    - Continuation after approval
    """
    from marie.query_planner.hitl import HitlApprovalQueryDefinition

    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # CLASSIFY DOCUMENT (AI Classification)
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify Invoice Document",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="invoice_classifier_v1",
            endpoint="mock_executor_a://classify/invoice",
            params={
                "layout": layout,
                "model": "invoice_classifier_v1",
                "confidence_output": True
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # HITL APPROVAL (Human Review)
    hitl_approval = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: HITL Approval - Verify Invoice Classification",
        dependencies=[classify.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlApprovalQueryDefinition(
            endpoint="hitl/approval",
            title="Verify Invoice Classification",
            description="Please review the AI classification of this invoice and approve or reject",
            approval_type="binary",
            priority="high",
            auto_approve={
                "enabled": True,
                "confidence_threshold": 0.95
            },
            timeout={
                "enabled": True,
                "duration_seconds": 86400,  # 24 hours
                "strategy": "use_default"
            },
            assigned_to={
                "user_ids": [],
                "roles": ["finance_manager", "accountant"]
            },
            notifications={
                "channels": ["email", "in_app"],
                "on_request": True,
                "on_reminder": True,
                "reminder_interval_seconds": 3600,
                "max_reminders": 3
            },
            ui={
                "show_confidence_score": True,
                "allow_feedback": True,
                "required_feedback": False
            },
            params={
                "layout": layout,
                "context_data": {
                    "document_type": "invoice",
                    "classification": "commercial_invoice"
                },
                "confidence": 0.87  # Below auto-approve threshold
            }
        ),
    )
    planner_info.current_id += 1
    nodes.append(hitl_approval)

    # PROCESS APPROVED INVOICE
    process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Process Approved Invoice",
        dependencies=[hitl_approval.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://invoice/process",
            params={"layout": layout, "mode": "approved"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(process)

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[process.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_hitl_correction")
def query_planner_mock_hitl_correction(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Mock query plan demonstrating HITL Data Correction workflow.

    Structure:
        START -> EXTRACT_INVOICE_DATA -> HITL_CORRECTION -> VALIDATE_DATA -> END

    This plan demonstrates:
    - AI extraction of structured invoice data
    - HITL correction request for human verification/correction
    - Field-level confidence thresholds
    - Auto-validation of high-confidence fields
    - Side-by-side comparison UI
    """
    from marie.query_planner.hitl import HitlCorrectionQueryDefinition

    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # EXTRACT INVOICE DATA (AI Extraction)
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Extract Invoice Data",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="invoice_extractor_v2",
            endpoint="mock_executor_a://extract/invoice",
            params={
                "layout": layout,
                "model": "invoice_extractor_v2",
                "fields": ["invoice_number", "date", "vendor", "total_amount", "line_items"]
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # HITL CORRECTION (Human Correction)
    hitl_correction = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: HITL Correction - Review Extracted Invoice Data",
        dependencies=[extract.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlCorrectionQueryDefinition(
            endpoint="hitl/correction",
            title="Review Extracted Invoice Data",
            description="Please verify and correct the AI-extracted invoice fields",
            correction_type="structured",
            priority="high",
            fields=[
                {
                    "key": "invoice_number",
                    "label": "Invoice Number",
                    "type": "text",
                    "required": True,
                    "confidence_threshold": 0.9,
                    "validation": r"^INV-\d{6}$"
                },
                {
                    "key": "invoice_date",
                    "label": "Invoice Date",
                    "type": "date",
                    "required": True,
                    "confidence_threshold": 0.85
                },
                {
                    "key": "vendor_name",
                    "label": "Vendor Name",
                    "type": "text",
                    "required": True,
                    "confidence_threshold": 0.8
                },
                {
                    "key": "total_amount",
                    "label": "Total Amount",
                    "type": "number",
                    "required": True,
                    "confidence_threshold": 0.95
                },
                {
                    "key": "currency",
                    "label": "Currency",
                    "type": "enum",
                    "required": True,
                    "options": ["USD", "EUR", "GBP", "JPY"],
                    "confidence_threshold": 0.9
                }
            ],
            auto_validate={
                "enabled": True,
                "confidence_threshold": 0.9,
                "exempt_fields": ["total_amount"]  # Always require human review for amount
            },
            timeout={
                "enabled": True,
                "duration_seconds": 86400,
                "strategy": "use_original"  # Use original extraction if timeout
            },
            assigned_to={
                "user_ids": [],
                "roles": ["data_entry_clerk", "finance_team"]
            },
            notifications={
                "channels": ["email", "in_app"],
                "on_request": True
            },
            ui={
                "highlight_low_confidence": True,
                "show_original_value": True,
                "side_by_side_comparison": True
            },
            params={
                "layout": layout,
                "original_data": {
                    "invoice_number": "INV-123456",
                    "invoice_date": "2025-01-15",
                    "vendor_name": "Acme Corporation",
                    "total_amount": 1234.56,
                    "currency": "USD"
                },
                "field_confidences": {
                    "invoice_number": 0.92,
                    "invoice_date": 0.78,  # Below threshold, needs review
                    "vendor_name": 0.88,
                    "total_amount": 0.96,  # High but exempt, needs review
                    "currency": 0.94
                }
            }
        ),
    )
    planner_info.current_id += 1
    nodes.append(hitl_correction)

    # VALIDATE CORRECTED DATA
    validate = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Validate Corrected Data",
        dependencies=[hitl_correction.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://validate/invoice",
            params={"layout": layout, "validation_rules": "strict"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(validate)

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[validate.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_hitl_router")
def query_planner_mock_hitl_router(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Mock query plan demonstrating HITL Router workflow.

    Structure:
        START -> CLASSIFY -> HITL_ROUTER -> [AUTO_PROCESS | HITL_REVIEW | REJECT] -> END

    This plan demonstrates:
    - Confidence-based routing without pausing
    - Three different paths based on AI confidence:
      * High confidence (>= 0.95): Auto-process
      * Medium confidence (0.7-0.95): Human review
      * Low confidence (< 0.7): Reject
    - Conditional routing rules
    """
    from marie.query_planner.hitl import HitlRouterQueryDefinition

    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # CLASSIFY DOCUMENT
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify Document",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="document_classifier_v1",
            endpoint="mock_executor_a://classify/document",
            params={
                "layout": layout,
                "confidence_output": True
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # Pre-allocate IDs for downstream nodes
    auto_process_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    human_review_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    reject_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # HITL ROUTER (Confidence-based routing)
    hitl_router = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: HITL Router - Route by Confidence",
        dependencies=[classify.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlRouterQueryDefinition(
            endpoint="hitl/router",
            auto_approve_threshold=0.95,
            human_review_threshold=0.7,
            below_threshold_action="reject",
            always_review_if=[
                {
                    "field": "total_amount",
                    "condition": "gt",
                    "value": 10000  # Always review large invoices
                },
                {
                    "field": "vendor_type",
                    "condition": "equals",
                    "value": "new"  # Always review new vendors
                }
            ],
            params={
                "layout": layout,
                "data": {
                    "document_type": "invoice",
                    "vendor_type": "existing",
                    "total_amount": 5000
                },
                "confidence": 0.88,  # Medium confidence
                "metadata": {
                    "classifier_model": "document_classifier_v1",
                    "timestamp": "2025-01-18T10:30:00Z"
                }
            }
        ),
    )
    planner_info.current_id += 1
    nodes.append(hitl_router)

    # AUTO PROCESS PATH (High confidence)
    auto_process = Query(
        task_id=auto_process_id,
        query_str=f"{planner_info.current_id}: Auto Process (High Confidence)",
        dependencies=[hitl_router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://process/auto",
            params={"layout": layout, "mode": "auto_approved"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(auto_process)

    # HUMAN REVIEW PATH (Medium confidence)
    human_review = Query(
        task_id=human_review_id,
        query_str=f"{planner_info.current_id}: Human Review (Medium Confidence)",
        dependencies=[hitl_router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://review/human",
            params={"layout": layout, "mode": "human_review"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(human_review)

    # REJECT PATH (Low confidence)
    reject = Query(
        task_id=reject_id,
        query_str=f"{planner_info.current_id}: Reject (Low Confidence)",
        dependencies=[hitl_router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://reject/low_confidence",
            params={"layout": layout, "mode": "reject"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(reject)

    # MERGE all paths
    merge = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Merge Results",
        dependencies=[auto_process.task_id, human_review.task_id, reject.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(merge)

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merge.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_hitl_complete_workflow")
def query_planner_mock_hitl_complete_workflow(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Complete HITL workflow demonstrating all three HITL node types.

    Structure:
        START -> EXTRACT -> HITL_ROUTER -> [
            High Confidence: AUTO_PROCESS -> END
            Medium Confidence: HITL_CORRECTION -> HITL_APPROVAL -> PROCESS -> END
            Low Confidence: REJECT -> END
        ]

    This comprehensive plan demonstrates:
    - Router for initial confidence-based routing
    - Correction for data verification
    - Approval for final sign-off
    - Multiple HITL touchpoints in a single workflow
    """
    from marie.query_planner.hitl import (
        HitlApprovalQueryDefinition,
        HitlCorrectionQueryDefinition,
        HitlRouterQueryDefinition,
    )

    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # EXTRACT (AI extracts invoice data)
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Extract Invoice Data",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="invoice_extractor_v2",
            endpoint="mock_executor_a://extract/invoice",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # Pre-allocate IDs
    auto_process_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    correction_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    reject_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # HITL ROUTER
    router = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: HITL Router",
        dependencies=[extract.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlRouterQueryDefinition(
            endpoint="hitl/router",
            auto_approve_threshold=0.95,
            human_review_threshold=0.7,
            below_threshold_action="reject",
            params={"layout": layout, "confidence": 0.82}
        ),
    )
    planner_info.current_id += 1
    nodes.append(router)

    # HIGH CONFIDENCE PATH: Auto-process
    auto_process = Query(
        task_id=auto_process_id,
        query_str=f"{planner_info.current_id}: Auto Process",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://process/auto",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(auto_process)

    # MEDIUM CONFIDENCE PATH: Correction -> Approval
    correction = Query(
        task_id=correction_id,
        query_str=f"{planner_info.current_id}: HITL Correction",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlCorrectionQueryDefinition(
            endpoint="hitl/correction",
            title="Verify Invoice Data",
            description="Please verify the extracted invoice fields",
            correction_type="structured",
            priority="high",
            fields=[
                {"key": "invoice_number", "label": "Invoice #", "type": "text", "required": True},
                {"key": "total_amount", "label": "Total", "type": "number", "required": True}
            ],
            params={"layout": layout}
        ),
    )
    planner_info.current_id += 1
    nodes.append(correction)

    approval = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: HITL Approval",
        dependencies=[correction.task_id],
        node_type=QueryType.COMPUTE,
        definition=HitlApprovalQueryDefinition(
            endpoint="hitl/approval",
            title="Final Approval",
            description="Approve the corrected invoice for processing",
            approval_type="binary",
            priority="medium",
            params={"layout": layout}
        ),
    )
    planner_info.current_id += 1
    nodes.append(approval)

    process_approved = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Process Approved Invoice",
        dependencies=[approval.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://process/approved",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(process_approved)

    # LOW CONFIDENCE PATH: Reject
    reject = Query(
        task_id=reject_id,
        query_str=f"{planner_info.current_id}: Reject Low Confidence",
        dependencies=[router.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://reject",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(reject)

    # MERGE all paths
    merge = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Merge All Paths",
        dependencies=[auto_process.task_id, process_approved.task_id, reject.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(merge)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merge.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)
