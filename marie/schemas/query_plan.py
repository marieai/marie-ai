"""Query Plan schemas for workflow generation.

This module defines the Pydantic models for Query Plan definitions,
including node types, methods, and the complete plan structure.
These schemas are used by the QueryPlanBuilderAgent to generate
valid workflow plans from natural language descriptions.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NodeMethod(str, Enum):
    """Available node methods for query plan nodes."""

    # Flow Control
    NOOP = "NOOP"
    START_END = "START_END"
    EXECUTOR_ENDPOINT = "EXECUTOR_ENDPOINT"

    # AI & LLM
    LLM = "LLM"
    PROMPT = "PROMPT"
    AGENT = "AGENT"

    # Transform
    CODE = "CODE"

    # Control Flow
    BRANCH = "BRANCH"
    SWITCH = "SWITCH"
    MERGE = "MERGE"
    MERGER_ENHANCED = "MERGER_ENHANCED"

    # Quality Control
    GUARDRAIL = "GUARDRAIL"

    # Human-in-the-Loop
    HITL_ROUTER = "HITL_ROUTER"
    HITL_APPROVAL = "HITL_APPROVAL"
    HITL_CORRECTION = "HITL_CORRECTION"

    # Annotators
    ANNOTATOR_LLM = "ANNOTATOR_LLM"
    ANNOTATOR_TABLE = "ANNOTATOR_TABLE"
    ANNOTATOR_EMBEDDING = "ANNOTATOR_EMBEDDING"
    ANNOTATOR_REGEX = "ANNOTATOR_REGEX"

    # RAG & Retrieval
    RAG_SEARCH = "RAG_SEARCH"
    RAG_INGEST = "RAG_INGEST"
    RAG_DELETE = "RAG_DELETE"
    CONTEXT_CACHE = "CONTEXT_CACHE"

    # Healthcare (Industry-specific)
    MEDICAL_RECORD_EXTRACTOR = "MEDICAL_RECORD_EXTRACTOR"
    HIPAA_REDACTOR = "HIPAA_REDACTOR"
    CLINICAL_NOTES_PARSER = "CLINICAL_NOTES_PARSER"
    ICD_CLASSIFIER = "ICD_CLASSIFIER"
    PATIENT_ANONYMIZER = "PATIENT_ANONYMIZER"
    LAB_RESULTS_EXTRACTOR = "LAB_RESULTS_EXTRACTOR"

    # Finance (Industry-specific)
    FINANCIAL_STATEMENT_PARSER = "FINANCIAL_STATEMENT_PARSER"
    TRANSACTION_CLASSIFIER = "TRANSACTION_CLASSIFIER"
    FRAUD_DETECTOR = "FRAUD_DETECTOR"
    KYC_VALIDATOR = "KYC_VALIDATOR"
    INVOICE_EXTRACTOR = "INVOICE_EXTRACTOR"
    TAX_DOCUMENT_PARSER = "TAX_DOCUMENT_PARSER"

    # Insurance (Industry-specific)
    CLAIMS_PROCESSOR = "CLAIMS_PROCESSOR"
    POLICY_PARSER = "POLICY_PARSER"
    RISK_ASSESSOR = "RISK_ASSESSOR"
    COVERAGE_VALIDATOR = "COVERAGE_VALIDATOR"
    UNDERWRITING_ANALYZER = "UNDERWRITING_ANALYZER"
    LOSS_ESTIMATOR = "LOSS_ESTIMATOR"

    # Legal (Industry-specific)
    CONTRACT_ANALYZER = "CONTRACT_ANALYZER"
    CLAUSE_EXTRACTOR = "CLAUSE_EXTRACTOR"
    LEGAL_ENTITY_RECOGNIZER = "LEGAL_ENTITY_RECOGNIZER"
    COMPLIANCE_CHECKER = "COMPLIANCE_CHECKER"
    SIGNATURE_VALIDATOR = "SIGNATURE_VALIDATOR"
    PRIVILEGE_DETECTOR = "PRIVILEGE_DETECTOR"


NodeType = Literal["COMPUTE", "BRANCH", "SWITCH", "MERGER", "GUARDRAIL"]


class BranchCondition(BaseModel):
    """Condition for a branch path."""

    jsonpath: str = Field(..., description="JSONPath expression to evaluate")
    operator: str = Field(
        ...,
        description="Comparison operator: eq, ne, gt, lt, gte, lte, contains, regex",
    )
    value: Any = Field(..., description="Value to compare against")
    description: Optional[str] = Field(
        default=None, description="Human-readable description"
    )


class BranchPath(BaseModel):
    """A path in a branch node."""

    path_id: str = Field(..., description="Unique identifier for this path")
    condition: Optional[BranchCondition] = Field(
        default=None, description="Condition to evaluate for this path"
    )
    condition_function: Optional[str] = Field(
        default=None, description="Custom Python function for complex conditions"
    )
    target_node_ids: List[str] = Field(
        default_factory=list, description="IDs of downstream nodes"
    )
    priority: int = Field(default=0, description="Evaluation priority (higher first)")
    description: Optional[str] = Field(default=None, description="Path description")


class GuardrailMetric(BaseModel):
    """A quality metric for guardrail validation."""

    type: str = Field(
        ...,
        description="Metric type: faithfulness, relevance, json_schema, regex_match, length_check, contains_keywords, llm_judge, executor",
    )
    name: str = Field(..., description="Human-readable metric name")
    weight: float = Field(default=1.0, description="Weight for weighted averaging")
    threshold: float = Field(default=0.8, description="Minimum score to pass")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Metric-specific parameters"
    )


class GuardrailPath(BaseModel):
    """A path in a guardrail node (pass or fail)."""

    path_id: Literal["pass", "fail"] = Field(..., description="Path type")
    target_node_ids: List[str] = Field(
        default_factory=list, description="IDs of downstream nodes"
    )
    description: Optional[str] = Field(default=None, description="Path description")


class AnnotatorModelConfig(BaseModel):
    """Configuration for LLM-based annotators."""

    model_name: str = Field(
        default="deepseek_r1_32", description="Model to use for annotation"
    )
    expect_output: Literal["json", "markdown"] = Field(
        default="json", description="Expected output format"
    )
    multimodal: bool = Field(default=False, description="Enable multimodal processing")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding-based annotators."""

    embedding_threshold: float = Field(
        default=0.85, description="Minimum similarity for embedding match"
    )
    fuzzy_threshold: float = Field(
        default=0.90, description="Minimum similarity for fuzzy match"
    )
    target_labels: List[str] = Field(
        default_factory=list, description="Labels to extract"
    )


class RegexPattern(BaseModel):
    """A regex pattern for extraction."""

    name: str = Field(..., description="Pattern name")
    pattern: str = Field(..., description="Regular expression pattern")
    type: Optional[str] = Field(default=None, description="Value type hint")


class AgentConfigOverride(BaseModel):
    """Override configuration for agent nodes."""

    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    temperature: float = Field(default=0.7, description="LLM temperature")
    timeout_seconds: int = Field(default=300, description="Execution timeout")


class NodeDefinition(BaseModel):
    """Definition of a query plan node's behavior."""

    method: str = Field(..., description="Node method (from NodeMethod enum)")
    endpoint: str = Field(default="", description="Executor endpoint URL or pattern")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )

    # LLM/Prompt fields
    model_name: Optional[str] = Field(default=None, description="LLM model to use")

    # Control flow fields
    paths: Optional[List[BranchPath]] = Field(
        default=None, description="Paths for BRANCH nodes"
    )
    merge_strategy: Optional[str] = Field(
        default=None, description="Merge strategy for MERGER nodes"
    )
    switch_field: Optional[str] = Field(
        default=None, description="JSONPath for SWITCH evaluation"
    )
    cases: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Case mappings for SWITCH nodes"
    )
    default_case: Optional[List[str]] = Field(
        default=None, description="Default case for SWITCH nodes"
    )

    # HITL fields
    auto_approve_threshold: Optional[float] = Field(
        default=None, description="Confidence threshold for auto-approval"
    )
    human_review_threshold: Optional[float] = Field(
        default=None, description="Confidence threshold for human review"
    )
    below_threshold_action: Optional[str] = Field(
        default=None, description="Action when below threshold"
    )
    title: Optional[str] = Field(default=None, description="HITL task title")
    description: Optional[str] = Field(
        default=None, description="HITL task description"
    )
    approval_type: Optional[str] = Field(default=None, description="Type of approval")
    correction_type: Optional[str] = Field(
        default=None, description="Type of correction"
    )
    priority: Optional[str] = Field(
        default=None, description="Task priority: low, medium, high"
    )
    fields: Optional[List[Any]] = Field(
        default=None, description="Fields for correction"
    )

    # Annotator fields
    layout_id: Optional[str] = Field(default=None, description="Layout identifier")
    annotator_name: Optional[str] = Field(default=None, description="Annotator name")
    system_prompt_path: Optional[str] = Field(
        default=None, description="Path to system prompt"
    )
    user_prompt_path: Optional[str] = Field(
        default=None, description="Path to user prompt"
    )
    annotator_model_config: Optional[AnnotatorModelConfig] = Field(
        default=None, description="Annotator model configuration"
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        default=None, description="Embedding annotator configuration"
    )
    regex_patterns: Optional[List[RegexPattern]] = Field(
        default=None, description="Regex patterns for extraction"
    )

    # Agent fields
    agent_id: Optional[str] = Field(default=None, description="Agent UUID")
    agent_slug: Optional[str] = Field(default=None, description="Agent slug name")
    agent_version: Optional[int] = Field(default=None, description="Agent version")
    task_description: Optional[str] = Field(
        default=None, description="Task description for agent"
    )
    agent_config_override: Optional[AgentConfigOverride] = Field(
        default=None, description="Agent configuration overrides"
    )

    # Guardrail fields
    input_source: Optional[str] = Field(
        default=None, description="JSONPath to input data"
    )
    context_source: Optional[str] = Field(
        default=None, description="JSONPath to context data"
    )
    query_source: Optional[str] = Field(
        default=None, description="JSONPath to query data"
    )
    metrics: Optional[List[GuardrailMetric]] = Field(
        default=None, description="Quality metrics to evaluate"
    )
    aggregation_mode: Optional[str] = Field(
        default=None, description="How to aggregate metrics: all, any, weighted_average"
    )
    pass_threshold: Optional[float] = Field(
        default=None, description="Overall pass threshold"
    )
    guardrail_paths: Optional[List[GuardrailPath]] = Field(
        default=None, description="Pass/fail paths"
    )
    fail_fast: Optional[bool] = Field(default=None, description="Stop on first failure")
    include_feedback: Optional[bool] = Field(
        default=None, description="Include feedback in output"
    )
    evaluation_timeout: Optional[int] = Field(
        default=None, description="Evaluation timeout in seconds"
    )

    # RAG fields
    index_name: Optional[str] = Field(default=None, description="Index/collection name")
    source_ids: Optional[List[str]] = Field(
        default=None, description="Source IDs to search"
    )
    top_k: Optional[int] = Field(default=None, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum similarity score"
    )
    hybrid: Optional[bool] = Field(default=None, description="Enable hybrid search")
    hybrid_alpha: Optional[float] = Field(
        default=None, description="Hybrid search balance"
    )
    rerank: Optional[bool] = Field(default=None, description="Enable reranking")
    rerank_model: Optional[str] = Field(default=None, description="Reranking model")
    rerank_top_k: Optional[int] = Field(
        default=None, description="Top-k after reranking"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters for search"
    )
    include_citations: Optional[bool] = Field(
        default=None, description="Include citations in response"
    )
    include_content: Optional[bool] = Field(
        default=None, description="Include content in response"
    )
    source_id: Optional[str] = Field(
        default=None, description="Source ID for ingest/delete"
    )
    ref_doc_id: Optional[str] = Field(default=None, description="Reference document ID")
    node_type: Optional[str] = Field(
        default=None, description="Node type: text, image, document"
    )
    batch_mode: Optional[bool] = Field(default=None, description="Enable batch mode")
    batch_size: Optional[int] = Field(default=None, description="Batch size")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata for ingest"
    )
    node_ids: Optional[List[str]] = Field(
        default=None, description="Node IDs for deletion"
    )

    # Context Cache fields
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to cache"
    )
    cache_key: Optional[str] = Field(default=None, description="Cache key identifier")
    ttl_seconds: Optional[int] = Field(
        default=None, description="Cache time-to-live in seconds"
    )
    include_metadata: Optional[bool] = Field(
        default=None, description="Include metadata in cache"
    )
    truncation_strategy: Optional[str] = Field(
        default=None, description="Truncation strategy: end, middle, semantic"
    )


class QueryPlanNode(BaseModel):
    """A single node in a query plan."""

    task_id: str = Field(..., description="Unique identifier for this node (UUID)")
    query_str: str = Field(..., description="Human-readable label for the node")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of upstream task_ids this node depends on (DAG edges)",
    )
    node_type: NodeType = Field(
        default="COMPUTE",
        description="Node type: COMPUTE, BRANCH, SWITCH, MERGER, GUARDRAIL",
    )
    definition: NodeDefinition = Field(..., description="Node behavior definition")


class QueryPlanDefinition(BaseModel):
    """Complete query plan definition with all nodes."""

    nodes: List[QueryPlanNode] = Field(
        default_factory=list, description="List of nodes in the plan"
    )


class GeneratePlanRequest(BaseModel):
    """Request to generate a query plan from natural language."""

    description: str = Field(
        ..., description="Natural language description of the workflow"
    )
    vertical: Optional[str] = Field(
        default=None,
        description="Business vertical: healthcare, finance, insurance, legal",
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context continuity"
    )


class GeneratePlanResponse(BaseModel):
    """Response from query plan generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    plan: Optional[QueryPlanDefinition] = Field(
        default=None, description="Generated query plan"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Agent's reasoning for the plan structure"
    )
    conversation_id: str = Field(..., description="Conversation ID for follow-ups")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RefinePlanRequest(BaseModel):
    """Request to refine an existing query plan."""

    current_plan: QueryPlanDefinition = Field(
        ..., description="Current query plan to refine"
    )
    feedback: str = Field(..., description="User feedback for refinement")
    conversation_id: str = Field(
        ..., description="Conversation ID for context continuity"
    )


class RefinePlanResponse(BaseModel):
    """Response from query plan refinement."""

    success: bool = Field(..., description="Whether refinement succeeded")
    plan: Optional[QueryPlanDefinition] = Field(
        default=None, description="Refined query plan"
    )
    changes: List[str] = Field(
        default_factory=list, description="Summary of changes made"
    )
    conversation_id: str = Field(..., description="Conversation ID for follow-ups")
    error: Optional[str] = Field(default=None, description="Error message if failed")
