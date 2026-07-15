"""Marie-side document extraction plugin integration."""

from marie.extraction.artifacts import read_extraction_artifact
from marie.extraction.format_router import FormatRouter
from marie.extraction.models import (
    CapabilitySnapshot,
    ExtractionSuccess,
    NotExtractable,
    parse_extraction_result,
)
from marie.extraction.result_adapter import (
    build_extraction_metadata,
    write_extraction_metadata,
)

__all__ = [
    "CapabilitySnapshot",
    "ExtractionSuccess",
    "FormatRouter",
    "NotExtractable",
    "build_extraction_metadata",
    "parse_extraction_result",
    "read_extraction_artifact",
    "write_extraction_metadata",
]
