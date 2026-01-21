"""Integration tests for Document Design Patterns.

Tests the pattern library for Visual Document Understanding tasks.
"""

import pytest

from marie.agent.patterns import (
    DOCUMENT_DESIGN_PATTERNS,
    DOCUMENT_TOOL_CATEGORIES,
    categorize_document_task,
    get_pattern_for_category,
)
from marie.agent.patterns.document_patterns import (
    TOOL_SUGGESTIONS,
    build_pattern_prompt,
    get_suggested_tools,
)


class TestDocumentDesignPatterns:
    """Test the design patterns content."""

    def test_patterns_not_empty(self):
        """Test that design patterns are defined."""
        assert DOCUMENT_DESIGN_PATTERNS
        assert len(DOCUMENT_DESIGN_PATTERNS) > 100  # Should be substantial

    def test_patterns_have_categories(self):
        """Test that patterns contain expected categories."""
        expected_categories = [
            "small_text",
            "rotated_text",
            "table_extraction",
            "form_extraction",
            "invoice_extraction",
            "multi_column",
            "handwriting",
            "document_comparison",
            "quality_assessment",
            "multi_page",
        ]

        for category in expected_categories:
            assert f"<category>{category}" in DOCUMENT_DESIGN_PATTERNS, \
                f"Missing category: {category}"

    def test_patterns_have_code_examples(self):
        """Test that patterns include code examples."""
        assert "```python" in DOCUMENT_DESIGN_PATTERNS
        assert "def " in DOCUMENT_DESIGN_PATTERNS

    def test_tool_categories_prompt(self):
        """Test that tool categories prompt is defined."""
        assert DOCUMENT_TOOL_CATEGORIES
        assert "OCR" in DOCUMENT_TOOL_CATEGORIES
        assert "table_extraction" in DOCUMENT_TOOL_CATEGORIES


class TestPatternRetrieval:
    """Test pattern retrieval functions."""

    def test_get_table_extraction_pattern(self):
        """Test retrieving table extraction pattern."""
        pattern = get_pattern_for_category("table_extraction")
        assert pattern is not None
        assert "table" in pattern.lower()
        assert "```python" in pattern

    def test_get_form_extraction_pattern(self):
        """Test retrieving form extraction pattern."""
        pattern = get_pattern_for_category("form_extraction")
        assert pattern is not None
        assert "form" in pattern.lower() or "key" in pattern.lower()

    def test_get_invoice_extraction_pattern(self):
        """Test retrieving invoice extraction pattern."""
        pattern = get_pattern_for_category("invoice_extraction")
        assert pattern is not None
        assert "invoice" in pattern.lower()

    def test_get_small_text_pattern(self):
        """Test retrieving small text pattern."""
        pattern = get_pattern_for_category("small_text")
        assert pattern is not None
        assert "upscal" in pattern.lower() or "small" in pattern.lower()

    def test_get_rotated_text_pattern(self):
        """Test retrieving rotated text pattern."""
        pattern = get_pattern_for_category("rotated_text")
        assert pattern is not None
        assert "rotat" in pattern.lower() or "skew" in pattern.lower()

    def test_get_handwriting_pattern(self):
        """Test retrieving handwriting pattern."""
        pattern = get_pattern_for_category("handwriting")
        assert pattern is not None
        assert "handwrit" in pattern.lower()

    def test_get_multi_page_pattern(self):
        """Test retrieving multi-page pattern."""
        pattern = get_pattern_for_category("multi_page")
        assert pattern is not None
        assert "page" in pattern.lower()

    def test_get_quality_assessment_pattern(self):
        """Test retrieving quality assessment pattern."""
        pattern = get_pattern_for_category("quality_assessment")
        assert pattern is not None
        assert "quality" in pattern.lower() or "assess" in pattern.lower()

    def test_get_document_comparison_pattern(self):
        """Test retrieving document comparison pattern."""
        pattern = get_pattern_for_category("document_comparison")
        assert pattern is not None
        assert "compar" in pattern.lower() or "diff" in pattern.lower()

    def test_get_nonexistent_pattern(self):
        """Test handling of nonexistent category."""
        pattern = get_pattern_for_category("nonexistent_category_xyz")
        assert pattern is None

    def test_pattern_contains_description(self):
        """Test that patterns contain description section."""
        pattern = get_pattern_for_category("table_extraction")
        assert "**Description**" in pattern or "Description" in pattern

    def test_pattern_contains_when_to_use(self):
        """Test that patterns contain when to use section."""
        pattern = get_pattern_for_category("table_extraction")
        assert "**When to use**" in pattern or "When to use" in pattern


class TestTaskCategorization:
    """Test task categorization function."""

    def test_categorize_table_keywords(self):
        """Test categorization based on table keywords."""
        assert categorize_document_task("Extract the table") == "table_extraction"
        assert categorize_document_task("Find the grid values") == "table_extraction"
        assert categorize_document_task("Get row and column data") == "table_extraction"

    def test_categorize_form_keywords(self):
        """Test categorization based on form keywords."""
        assert categorize_document_task("Extract form fields") == "form_extraction"
        assert categorize_document_task("Fill in the form") == "form_extraction"
        assert categorize_document_task("Find checkbox values") == "form_extraction"

    def test_categorize_invoice_keywords(self):
        """Test categorization based on invoice keywords."""
        assert categorize_document_task("Process this invoice") == "invoice_extraction"
        assert categorize_document_task("Extract receipt data") == "invoice_extraction"
        assert categorize_document_task("Get payment details") == "invoice_extraction"

    def test_categorize_ocr_keywords(self):
        """Test categorization based on OCR keywords."""
        assert categorize_document_task("Extract text from image") == "OCR"
        assert categorize_document_task("Run OCR on this page") == "OCR"
        assert categorize_document_task("Read text from scan") == "OCR"

    def test_categorize_qa_keywords(self):
        """Test categorization based on Q&A keywords."""
        assert categorize_document_task("What color is the logo?") == "DocQA"
        assert categorize_document_task("Where is the header?") == "DocQA"
        assert categorize_document_task("Who is the author?") == "DocQA"
        assert categorize_document_task("How big is the margin?") == "DocQA"

    def test_categorize_classification_keywords(self):
        """Test categorization based on classification keywords."""
        assert categorize_document_task("Classify this document") == "classification"
        assert categorize_document_task("What type of document is this?") == "classification"
        assert categorize_document_task("Identify the category") == "classification"

    def test_categorize_ner_keywords(self):
        """Test categorization based on NER keywords."""
        assert categorize_document_task("Identify each entity in the photo") == "ner"
        assert categorize_document_task("Get all names mentioned") == "ner"
        assert categorize_document_task("NER analysis required") == "ner"

    def test_categorize_layout_keywords(self):
        """Test categorization based on layout keywords."""
        assert categorize_document_task("Analyze the layout") == "layout_analysis"
        assert categorize_document_task("Find document structure") == "layout_analysis"
        assert categorize_document_task("Identify regions and sections") == "layout_analysis"

    def test_categorize_handwriting_keywords(self):
        """Test categorization based on handwriting keywords."""
        assert categorize_document_task("Extract handwritten notes") == "handwriting"
        assert categorize_document_task("Read the cursive text") == "handwriting"

    def test_categorize_signature_keywords(self):
        """Test categorization based on signature keywords."""
        assert categorize_document_task("Find the signature") == "signature_detection"
        assert categorize_document_task("Verify the autograph") == "signature_detection"

    def test_categorize_comparison_keywords(self):
        """Test categorization based on comparison keywords."""
        assert categorize_document_task("Compare these documents") == "document_comparison"
        assert categorize_document_task("Find differences between versions") == "document_comparison"

    def test_categorize_quality_keywords(self):
        """Test categorization based on quality keywords."""
        assert categorize_document_task("Check image quality") == "quality_assessment"
        assert categorize_document_task("Is this document blurry?") == "quality_assessment"
        assert categorize_document_task("Assess the resolution") == "quality_assessment"

    def test_categorize_multipage_keywords(self):
        """Test categorization based on multi-page keywords."""
        assert categorize_document_task("Process all pages") == "multi_page"
        assert categorize_document_task("Extract from PDF document") == "multi_page"

    def test_categorize_case_insensitive(self):
        """Test that categorization is case insensitive."""
        assert categorize_document_task("EXTRACT THE TABLE") == "table_extraction"
        assert categorize_document_task("Extract The Table") == "table_extraction"

    def test_categorize_general_fallback(self):
        """Test fallback to general category."""
        assert categorize_document_task("Do something with this") == "general"
        assert categorize_document_task("Process please") == "general"


class TestToolSuggestions:
    """Test tool suggestion functions."""

    def test_tool_suggestions_defined(self):
        """Test that tool suggestions are defined for main categories."""
        expected_categories = [
            "OCR", "DocQA", "table_extraction", "form_extraction",
            "invoice_extraction", "classification", "ner",
            "layout_analysis", "handwriting", "general",
        ]

        for category in expected_categories:
            assert category in TOOL_SUGGESTIONS, f"Missing suggestions for: {category}"

    def test_get_ocr_tools(self):
        """Test getting tools for OCR category."""
        tools = get_suggested_tools("OCR")
        assert "ocr" in tools

    def test_get_table_tools(self):
        """Test getting tools for table extraction."""
        tools = get_suggested_tools("table_extraction")
        assert "detect_tables" in tools
        assert "ocr" in tools

    def test_get_form_tools(self):
        """Test getting tools for form extraction."""
        tools = get_suggested_tools("form_extraction")
        assert "detect_form_fields" in tools or "ocr" in tools

    def test_get_qa_tools(self):
        """Test getting tools for document QA."""
        tools = get_suggested_tools("DocQA")
        assert "vqa" in tools or "ocr" in tools

    def test_get_classification_tools(self):
        """Test getting tools for classification."""
        tools = get_suggested_tools("classification")
        assert "classify_document" in tools

    def test_get_general_tools(self):
        """Test getting tools for general tasks."""
        tools = get_suggested_tools("general")
        assert len(tools) > 0
        assert "ocr" in tools

    def test_get_unknown_category_tools(self):
        """Test getting tools for unknown category falls back to general."""
        tools = get_suggested_tools("unknown_category")
        assert tools == TOOL_SUGGESTIONS["general"]


class TestBuildPatternPrompt:
    """Test pattern prompt building function."""

    def test_build_prompt_includes_category(self):
        """Test that built prompt includes category."""
        prompt = build_pattern_prompt("Extract the table from this invoice")
        assert "table_extraction" in prompt.lower()

    def test_build_prompt_includes_tools(self):
        """Test that built prompt includes suggested tools."""
        prompt = build_pattern_prompt("Extract the table from this invoice")
        assert "detect_tables" in prompt.lower() or "tool" in prompt.lower()

    def test_build_prompt_includes_task(self):
        """Test that built prompt includes original task."""
        task = "Extract the table from this invoice"
        prompt = build_pattern_prompt(task)
        assert task in prompt

    def test_build_prompt_with_examples(self):
        """Test that built prompt includes examples when requested."""
        prompt = build_pattern_prompt("Extract the table", include_examples=True)
        assert "pattern" in prompt.lower()

    def test_build_prompt_without_examples(self):
        """Test that built prompt excludes examples when requested."""
        prompt = build_pattern_prompt("Extract the table", include_examples=False)
        # Should still have category and tools but potentially shorter
        assert "table_extraction" in prompt.lower()


class TestPatternCodeQuality:
    """Test that pattern code examples are well-formed."""

    def test_pattern_has_function_definitions(self):
        """Test that patterns include proper function definitions."""
        pattern = get_pattern_for_category("table_extraction")
        assert "def " in pattern
        assert "return " in pattern

    def test_pattern_has_docstrings(self):
        """Test that pattern functions have docstrings."""
        pattern = get_pattern_for_category("table_extraction")
        assert "'''" in pattern or '"""' in pattern

    def test_pattern_imports_cv2(self):
        """Test that image processing patterns import cv2."""
        pattern = get_pattern_for_category("rotated_text")
        # Rotated text pattern should use OpenCV
        assert "cv2" in pattern

    def test_pattern_imports_numpy(self):
        """Test that image processing patterns import numpy."""
        pattern = get_pattern_for_category("quality_assessment")
        assert "numpy" in pattern or "np" in pattern

    def test_pattern_handles_results(self):
        """Test that patterns return structured results."""
        pattern = get_pattern_for_category("table_extraction")
        # Should return a dict with results
        assert "return" in pattern
        assert "{" in pattern  # Returns a dict
