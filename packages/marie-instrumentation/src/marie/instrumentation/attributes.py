class MarieSpanAttributes:
    TASK_ID = "marie.task_id"
    REQUEST_ID = "marie.request_id"
    BATCH_SIZE = "marie.batch_size"
    LATENCY_SECONDS = "marie.latency_seconds"
    SUCCESSFUL_COUNT = "marie.successful_count"
    FAILED_COUNT = "marie.failed_count"
    HAS_REASONING = "marie.has_reasoning"

    EXECUTOR = "marie.executor"
    ENDPOINT = "marie.endpoint"
    ANNOTATOR_NAME = "marie.annotator_name"
    LAYOUT_ID = "marie.layout_id"
    JOB_ID = "marie.job_id"

    PASS_INDEX = "marie.pass_index"
    PASS_TYPE = "marie.pass_type"
    MODEL_NAME = "marie.model_name"
    REFINE_PASSES = "marie.refine_passes"
    PROCESSING_MODE = "marie.processing_mode"
    MULTIMODAL = "marie.multimodal"

    LLM_DISPATCH_REQUEST_ID = "marie.llm_dispatch.request_id"
    LLM_DISPATCH_PRODUCER_ID = "marie.llm_dispatch.producer_id"
    LLM_DISPATCH_POOL_ID = "marie.llm_dispatch.pool_id"
    LLM_DISPATCH_FABRIC_GROUP_ID = "marie.llm_dispatch.fabric_group_id"
    LLM_DISPATCH_GATEWAY_ID = "marie.llm_dispatch.gateway_id"
    LLM_DISPATCH_DISPATCHER_ID = "marie.llm_dispatch.dispatcher_id"
    LLM_DISPATCH_PROFILE_KEY = "marie.llm_dispatch.dispatch_profile_key"
    LLM_DISPATCH_BACKEND_ADDRESS = "marie.llm_dispatch.backend_address"
    LLM_DISPATCH_MODEL = "marie.llm_dispatch.model"
    LLM_DISPATCH_QUEUE_WAIT_MS = "marie.llm_dispatch.queue_wait_ms"
    LLM_DISPATCH_MESSAGE_COUNT = "marie.llm_dispatch.message_count"
    LLM_DISPATCH_CONTRACT_VERSION = "marie.llm_dispatch.contract_version"
    LLM_DISPATCH_STATUS = "marie.llm_dispatch.status"
    LLM_DISPATCH_EXECUTION_MS = "marie.llm_dispatch.execution_ms"
    LLM_DISPATCH_TOTAL_LATENCY_MS = "marie.llm_dispatch.total_latency_ms"
    LLM_DISPATCH_ERROR_TYPE = "marie.llm_dispatch.error_type"
    LLM_DISPATCH_ERROR_MESSAGE = "marie.llm_dispatch.error_message"

    @staticmethod
    def marie(name: str) -> str:
        return f"marie.{name}"

    @staticmethod
    def media_reference(
        direction: str,
        message_index: int,
        content_index: int,
        field: str,
    ) -> str:
        return f"marie.otel.media.{direction}.{message_index}.{content_index}.{field}"

    @staticmethod
    def media_count(direction: str) -> str:
        return f"marie.otel.media.{direction}.count"

    @staticmethod
    def media_reference_mode(direction: str) -> str:
        return f"marie.otel.media.{direction}.reference_mode"

    @staticmethod
    def media_reference_error(direction: str) -> str:
        return f"marie.otel.media.{direction}.reference_error"
