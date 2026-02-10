"""Query Plan Builder tools for workflow generation.

This module provides tools for the QueryPlanBuilderAgent to:
- Get available node types and their descriptions
- Get common patterns for business verticals
- Validate generated query plans
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from marie.agent.tools.registry import register_tool
from marie.schemas.query_plan import NodeMethod, QueryPlanDefinition


class NodeTypeInfo(BaseModel):
    """Information about a node type."""

    method: str
    label: str
    description: str
    category: str
    node_type: str
    verticals: List[str]
    default_endpoint: Optional[str] = None


# Node type catalog with complete descriptions
NODE_CATALOG: List[NodeTypeInfo] = [
    # Flow Control
    NodeTypeInfo(
        method="START_END",
        label="Start/End",
        description="Workflow start or end marker. Use at the beginning and end of every workflow.",
        category="flow",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="noop",
    ),
    NodeTypeInfo(
        method="EXECUTOR_ENDPOINT",
        label="Executor",
        description="Call a custom executor endpoint for processing. Use for OCR, extraction, or custom logic.",
        category="executors",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    # AI & LLM
    NodeTypeInfo(
        method="LLM",
        label="LLM Node",
        description="Direct LLM call for text generation or analysis.",
        category="ai",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="PROMPT",
        label="Prompt",
        description="Call an LLM with a structured prompt template. Best for classification, summarization, or Q&A.",
        category="ai",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="AGENT",
        label="AI Agent",
        description="Execute an AI agent with tools for complex multi-step reasoning tasks.",
        category="ai",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="agent_executor://agent/execute",
    ),
    # Transform
    NodeTypeInfo(
        method="CODE",
        label="Code",
        description="Execute custom code (Python, JavaScript, Rust) for data transformation.",
        category="transform",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    # Control Flow
    NodeTypeInfo(
        method="BRANCH",
        label="Branch",
        description="Conditional routing based on JSONPath conditions. Routes to different paths based on data values.",
        category="control-flow",
        node_type="BRANCH",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="SWITCH",
        label="Switch",
        description="Value-based routing with multiple cases. Like a switch statement for workflow routing.",
        category="control-flow",
        node_type="SWITCH",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="MERGER_ENHANCED",
        label="Enhanced Merger",
        description="Merge multiple execution paths. Strategies: WAIT_ALL_ACTIVE, WAIT_ANY, WAIT_N.",
        category="control-flow",
        node_type="MERGER",
        verticals=["all"],
    ),
    # Quality Control
    NodeTypeInfo(
        method="GUARDRAIL",
        label="Guardrail",
        description="Validate output quality with metrics (faithfulness, relevance, schema validation). Routes to pass/fail paths.",
        category="quality",
        node_type="GUARDRAIL",
        verticals=["all"],
        default_endpoint="guardrail://control",
    ),
    # HITL
    NodeTypeInfo(
        method="HITL_ROUTER",
        label="HITL Router",
        description="Route to human review based on confidence scores. Auto-approve above threshold, human review below.",
        category="hitl",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="HITL_APPROVAL",
        label="HITL Approval",
        description="Human approval step for review and sign-off.",
        category="hitl",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    NodeTypeInfo(
        method="HITL_CORRECTION",
        label="HITL Correction",
        description="Human data correction for fixing extraction errors.",
        category="hitl",
        node_type="COMPUTE",
        verticals=["all"],
    ),
    # Annotators
    NodeTypeInfo(
        method="ANNOTATOR_LLM",
        label="LLM Annotator",
        description="LLM-based document field extraction using prompts.",
        category="annotators",
        node_type="COMPUTE",
        verticals=["healthcare", "finance", "insurance", "legal"],
        default_endpoint="annotator_executor://document/annotate",
    ),
    NodeTypeInfo(
        method="ANNOTATOR_TABLE",
        label="Table Annotator",
        description="Extract and structure table data from documents.",
        category="annotators",
        node_type="COMPUTE",
        verticals=["finance", "insurance", "healthcare"],
        default_endpoint="annotator_executor://document/annotate",
    ),
    NodeTypeInfo(
        method="ANNOTATOR_EMBEDDING",
        label="Embedding Annotator",
        description="FAISS-based hybrid field extraction using embeddings.",
        category="annotators",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="annotator_executor://document/annotate",
    ),
    NodeTypeInfo(
        method="ANNOTATOR_REGEX",
        label="Regex Annotator",
        description="Pattern-based extraction using regular expressions.",
        category="annotators",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="annotator_executor://document/annotate",
    ),
    # RAG & Retrieval
    NodeTypeInfo(
        method="RAG_SEARCH",
        label="RAG Search",
        description="Search documents in vector store with hybrid search and reranking.",
        category="rag",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="vector_store_executor://rag/search",
    ),
    NodeTypeInfo(
        method="RAG_INGEST",
        label="RAG Ingest",
        description="Embed and store documents in vector store for later retrieval.",
        category="rag",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="vector_store_executor://rag/embed_and_store",
    ),
    NodeTypeInfo(
        method="RAG_DELETE",
        label="RAG Delete",
        description="Delete documents from vector store.",
        category="rag",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="vector_store_executor://rag/delete_source",
    ),
    NodeTypeInfo(
        method="CONTEXT_CACHE",
        label="Context Cache",
        description="Preload documents into LLM context window (CAG - Context-Augmented Generation).",
        category="rag",
        node_type="COMPUTE",
        verticals=["all"],
        default_endpoint="context_cache://preload",
    ),
    # Healthcare
    NodeTypeInfo(
        method="MEDICAL_RECORD_EXTRACTOR",
        label="Medical Record Extractor",
        description="Extract patient info, diagnoses, and treatments from medical records.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    NodeTypeInfo(
        method="HIPAA_REDACTOR",
        label="HIPAA Redactor",
        description="Automatically redact PHI to ensure HIPAA compliance.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    NodeTypeInfo(
        method="CLINICAL_NOTES_PARSER",
        label="Clinical Notes Parser",
        description="Parse and structure unstructured clinical notes.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    NodeTypeInfo(
        method="ICD_CLASSIFIER",
        label="ICD Code Classifier",
        description="Classify diagnoses using ICD-10/ICD-11 codes.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    NodeTypeInfo(
        method="PATIENT_ANONYMIZER",
        label="Patient Anonymizer",
        description="De-identify patient data for research and analytics.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    NodeTypeInfo(
        method="LAB_RESULTS_EXTRACTOR",
        label="Lab Results Extractor",
        description="Extract and normalize laboratory test results.",
        category="healthcare",
        node_type="COMPUTE",
        verticals=["healthcare"],
    ),
    # Finance
    NodeTypeInfo(
        method="FINANCIAL_STATEMENT_PARSER",
        label="Financial Statement Parser",
        description="Parse balance sheets, income statements, and cash flows.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    NodeTypeInfo(
        method="TRANSACTION_CLASSIFIER",
        label="Transaction Classifier",
        description="Categorize and classify financial transactions.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    NodeTypeInfo(
        method="FRAUD_DETECTOR",
        label="Fraud Detector",
        description="Detect suspicious patterns and potential fraud.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    NodeTypeInfo(
        method="KYC_VALIDATOR",
        label="KYC Validator",
        description="Validate Know Your Customer documents and data.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    NodeTypeInfo(
        method="INVOICE_EXTRACTOR",
        label="Invoice Extractor",
        description="Extract line items, amounts, and vendor info from invoices.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    NodeTypeInfo(
        method="TAX_DOCUMENT_PARSER",
        label="Tax Document Parser",
        description="Parse W-2s, 1099s, and other tax forms.",
        category="finance",
        node_type="COMPUTE",
        verticals=["finance"],
    ),
    # Insurance
    NodeTypeInfo(
        method="CLAIMS_PROCESSOR",
        label="Claims Processor",
        description="Process and validate insurance claims automatically.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    NodeTypeInfo(
        method="POLICY_PARSER",
        label="Policy Parser",
        description="Extract coverage details, limits, and exclusions from policies.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    NodeTypeInfo(
        method="RISK_ASSESSOR",
        label="Risk Assessor",
        description="Evaluate risk factors and calculate risk scores.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    NodeTypeInfo(
        method="COVERAGE_VALIDATOR",
        label="Coverage Validator",
        description="Verify coverage eligibility and policy limits.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    NodeTypeInfo(
        method="UNDERWRITING_ANALYZER",
        label="Underwriting Analyzer",
        description="Analyze applications for underwriting decisions.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    NodeTypeInfo(
        method="LOSS_ESTIMATOR",
        label="Loss Estimator",
        description="Estimate claim amounts and loss projections.",
        category="insurance",
        node_type="COMPUTE",
        verticals=["insurance"],
    ),
    # Legal
    NodeTypeInfo(
        method="CONTRACT_ANALYZER",
        label="Contract Analyzer",
        description="Analyze contracts for key terms, obligations, and risks.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
    NodeTypeInfo(
        method="CLAUSE_EXTRACTOR",
        label="Clause Extractor",
        description="Extract and categorize contract clauses.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
    NodeTypeInfo(
        method="LEGAL_ENTITY_RECOGNIZER",
        label="Legal Entity Recognizer",
        description="Identify parties, dates, and legal entities in documents.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
    NodeTypeInfo(
        method="COMPLIANCE_CHECKER",
        label="Compliance Checker",
        description="Check documents against regulatory requirements.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
    NodeTypeInfo(
        method="SIGNATURE_VALIDATOR",
        label="Signature Validator",
        description="Validate and verify document signatures.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
    NodeTypeInfo(
        method="PRIVILEGE_DETECTOR",
        label="Privilege Detector",
        description="Identify attorney-client privileged content.",
        category="legal",
        node_type="COMPUTE",
        verticals=["legal"],
    ),
]


@register_tool(
    name="get_node_types",
    description="Get all available node types with descriptions. Optionally filter by business vertical.",
)
def get_node_types(vertical: Optional[str] = None) -> str:
    """Returns all available node types with descriptions.

    Args:
        vertical: Optional business vertical to filter/highlight nodes.
                  Options: healthcare, finance, insurance, legal, all

    Returns:
        JSON string with node type information.
    """
    result = []

    for node in NODE_CATALOG:
        # Filter by vertical if specified
        if vertical and vertical != "all":
            if "all" not in node.verticals and vertical not in node.verticals:
                continue

        result.append(
            {
                "method": node.method,
                "label": node.label,
                "description": node.description,
                "category": node.category,
                "node_type": node.node_type,
                "verticals": node.verticals,
                "recommended": (
                    vertical is not None
                    and vertical != "all"
                    and vertical in node.verticals
                ),
            }
        )

    return json.dumps(
        {
            "total": len(result),
            "vertical_filter": vertical,
            "nodes": result,
        },
        indent=2,
    )


@register_tool(
    name="validate_query_plan",
    description="Validate a query plan JSON against the schema. Returns validation errors or 'valid'.",
)
def validate_query_plan(plan_json: str) -> str:
    """Validates a query plan against the Pydantic schema.

    Args:
        plan_json: JSON string of the query plan to validate.

    Returns:
        'valid' if the plan is valid, or a description of validation errors.
    """
    try:
        plan_data = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"

    try:
        QueryPlanDefinition.model_validate(plan_data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{location}: {error['msg']}")
        return f"Validation errors:\n" + "\n".join(errors)

    # Additional semantic validation
    if not plan_data.get("nodes"):
        return "Plan must have at least one node"

    nodes = plan_data["nodes"]
    node_ids = {node["task_id"] for node in nodes}

    # Check for duplicate task_ids
    if len(node_ids) != len(nodes):
        return "Duplicate task_id found in nodes"

    # Check dependency references
    for node in nodes:
        for dep in node.get("dependencies", []):
            if dep not in node_ids:
                return f"Node '{node['task_id']}' references unknown dependency '{dep}'"

    # Check for start node
    start_nodes = [
        n for n in nodes if n.get("definition", {}).get("method") == "START_END"
    ]
    if not start_nodes:
        return "Plan should have at least one START_END node"

    # Check for DAG (no cycles) - simplified check
    visited = set()
    rec_stack = set()

    def has_cycle(node_id: str) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)

        node = next((n for n in nodes if n["task_id"] == node_id), None)
        if node:
            for dep in node.get("dependencies", []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

        rec_stack.remove(node_id)
        return False

    for node in nodes:
        if node["task_id"] not in visited:
            if has_cycle(node["task_id"]):
                return "Plan contains a cycle - dependencies must form a DAG"

    return "valid"


@register_tool(
    name="get_node_defaults",
    description="Get default configuration for a specific node method.",
)
def get_node_defaults(method: str) -> str:
    """Get default configuration values for a node method.

    Args:
        method: The node method (e.g., BRANCH, GUARDRAIL, RAG_SEARCH)

    Returns:
        JSON string with default configuration values.
    """
    defaults: Dict[str, Any] = {
        "method": method,
        "endpoint": "",
        "params": {},
    }

    method_upper = method.upper()

    if method_upper == "START_END" or method_upper == "NOOP":
        defaults["endpoint"] = "noop"

    elif method_upper == "BRANCH":
        defaults["paths"] = []
        defaults["default_path_id"] = ""
        defaults["evaluation_mode"] = "first_match"

    elif method_upper == "SWITCH":
        defaults["switch_field"] = "$.metadata.type"
        defaults["cases"] = {}

    elif method_upper == "MERGER_ENHANCED":
        defaults["merge_strategy"] = "WAIT_ALL_ACTIVE"

    elif method_upper == "GUARDRAIL":
        defaults["endpoint"] = "guardrail://control"
        defaults["metrics"] = []
        defaults["aggregation_mode"] = "all"
        defaults["pass_threshold"] = 0.8
        defaults["guardrail_paths"] = [
            {"path_id": "pass", "target_node_ids": []},
            {"path_id": "fail", "target_node_ids": []},
        ]

    elif method_upper.startswith("HITL_"):
        defaults["title"] = ""
        defaults["description"] = ""
        defaults["priority"] = "medium"
        if method_upper == "HITL_ROUTER":
            defaults["auto_approve_threshold"] = 0.9
            defaults["human_review_threshold"] = 0.7

    elif method_upper.startswith("ANNOTATOR_"):
        defaults["endpoint"] = "annotator_executor://document/annotate"
        defaults["layout_id"] = ""
        defaults["annotator_name"] = ""
        if method_upper in ("ANNOTATOR_LLM", "ANNOTATOR_TABLE"):
            defaults["annotator_model_config"] = {
                "model_name": "deepseek_r1_32",
                "expect_output": "json",
            }
        elif method_upper == "ANNOTATOR_EMBEDDING":
            defaults["embedding_config"] = {
                "embedding_threshold": 0.85,
                "fuzzy_threshold": 0.90,
                "target_labels": [],
            }
        elif method_upper == "ANNOTATOR_REGEX":
            defaults["regex_patterns"] = []

    elif method_upper == "AGENT":
        defaults["endpoint"] = "agent_executor://agent/execute"
        defaults["agent_config_override"] = {
            "max_iterations": 10,
            "temperature": 0.7,
            "timeout_seconds": 300,
        }

    elif method_upper == "RAG_SEARCH":
        defaults["endpoint"] = "vector_store_executor://rag/search"
        defaults["index_name"] = "default"
        defaults["top_k"] = 5
        defaults["hybrid"] = False
        defaults["include_citations"] = True

    elif method_upper == "RAG_INGEST":
        defaults["endpoint"] = "vector_store_executor://rag/embed_and_store"
        defaults["index_name"] = "default"
        defaults["node_type"] = "text"
        defaults["batch_mode"] = False

    elif method_upper == "RAG_DELETE":
        defaults["endpoint"] = "vector_store_executor://rag/delete_source"

    elif method_upper == "CONTEXT_CACHE":
        defaults["endpoint"] = "context_cache://preload"
        defaults["index_name"] = "default"
        defaults["max_tokens"] = 100000
        defaults["ttl_seconds"] = 3600
        defaults["truncation_strategy"] = "end"

    # Find node info for additional context
    node_info = next((n for n in NODE_CATALOG if n.method == method_upper), None)
    if node_info:
        defaults["_info"] = {
            "label": node_info.label,
            "description": node_info.description,
            "node_type": node_info.node_type,
            "category": node_info.category,
        }

    return json.dumps(defaults, indent=2)
