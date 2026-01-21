"""Document Understanding Design Patterns.

A library of pre-built patterns for common document analysis tasks,
inspired by Landing.ai's vision-agent approach but tailored for
Visual Document Understanding (VDU).

Categories:
    - Text extraction patterns (OCR, small text, rotated text)
    - Layout analysis patterns (tables, forms, multi-column)
    - Entity extraction patterns (NER, key-value, structured data)
    - Document classification patterns
    - Quality assessment patterns
    - Multi-page document patterns
"""

from marie.agent.patterns.document_patterns import (
    DOCUMENT_DESIGN_PATTERNS,
    DOCUMENT_TOOL_CATEGORIES,
    categorize_document_task,
    get_pattern_for_category,
)

__all__ = [
    "DOCUMENT_DESIGN_PATTERNS",
    "DOCUMENT_TOOL_CATEGORIES",
    "get_pattern_for_category",
    "categorize_document_task",
]
