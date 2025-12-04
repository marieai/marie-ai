"""Constants used throughout the Marie MCP Server."""

# Size constants
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Time constants (in seconds)
SECOND = 1
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".heic"}
SUPPORTED_DOCUMENT_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx"}
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_DOCUMENT_FORMATS

# Job states
JOB_STATE_CREATED = "created"
JOB_STATE_PENDING = "pending"
JOB_STATE_ACTIVE = "active"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"
JOB_STATE_CANCELLED = "cancelled"

# Queue names
QUEUE_EXTRACT = "extract"  # OCR extraction
QUEUE_GEN5_EXTRACT = "gen5_extract"  # Data extraction with templates

# Planner names
PLANNER_EXTRACT = "extract"  # OCR planner
