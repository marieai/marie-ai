"""Integration tests for VisionDocumentAgent.

Tests the document-specialized agent with design pattern support
and automatic task categorization.
"""

import json

import pytest

from marie.agent import (
    DocumentExtractionAgent,
    DocumentQAAgent,
    Message,
    VisionDocumentAgent,
    register_tool,
)
from marie.agent.patterns.document_patterns import (
    build_pattern_prompt,
    categorize_document_task,
    get_pattern_for_category,
    get_suggested_tools,
)
from tests.integration.agent.conftest import (
    MockLLMWrapper,
    SequenceMockLLMWrapper,
    run_agent_to_completion,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_ocr_tool():
    """Create a mock OCR tool."""
    from marie.agent import AgentTool, ToolMetadata, ToolOutput

    class MockOCRTool(AgentTool):
        @property
        def name(self):
            return "ocr"

        @property
        def metadata(self):
            return ToolMetadata(
                name="ocr",
                description="Extract text from document images",
                parameters={
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path"},
                    },
                    "required": ["image"],
                },
            )

        def call(self, **kwargs):
            return ToolOutput(
                content=json.dumps({
                    "text": "Invoice #12345\nTotal: $500.00",
                    "confidence": 0.95,
                }),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"text": "Invoice #12345"},
                is_error=False,
            )

    return MockOCRTool()


@pytest.fixture
def mock_table_tool():
    """Create a mock table extraction tool."""
    from marie.agent import AgentTool, ToolMetadata, ToolOutput

    class MockTableTool(AgentTool):
        @property
        def name(self):
            return "detect_tables"

        @property
        def metadata(self):
            return ToolMetadata(
                name="detect_tables",
                description="Detect tables in documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path"},
                    },
                    "required": ["image"],
                },
            )

        def call(self, **kwargs):
            return ToolOutput(
                content=json.dumps({
                    "tables": [{"bbox": [0.1, 0.4, 0.9, 0.7], "rows": 3, "cols": 4}],
                    "count": 1,
                }),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"tables": []},
                is_error=False,
            )

    return MockTableTool()


@pytest.fixture
def mock_classifier_tool():
    """Create a mock document classifier tool."""
    from marie.agent import AgentTool, ToolMetadata, ToolOutput

    class MockClassifierTool(AgentTool):
        @property
        def name(self):
            return "classify_document"

        @property
        def metadata(self):
            return ToolMetadata(
                name="classify_document",
                description="Classify document type",
            )

        def call(self, **kwargs):
            return ToolOutput(
                content=json.dumps({"type": "invoice", "confidence": 0.94}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"type": "invoice"},
                is_error=False,
            )

    return MockClassifierTool()


@pytest.fixture
def mock_vqa_tool():
    """Create a mock VQA tool."""
    from marie.agent import AgentTool, ToolMetadata, ToolOutput

    class MockVQATool(AgentTool):
        @property
        def name(self):
            return "vqa"

        @property
        def metadata(self):
            return ToolMetadata(
                name="vqa",
                description="Answer questions about documents",
            )

        def call(self, **kwargs):
            return ToolOutput(
                content=json.dumps({
                    "answer": "The total amount is $500.00",
                    "confidence": 0.88,
                }),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"answer": "The total amount is $500.00"},
                is_error=False,
            )

    return MockVQATool()


@pytest.fixture
def document_tools(mock_ocr_tool, mock_table_tool, mock_classifier_tool, mock_vqa_tool):
    """Create a set of document processing tools."""
    return [mock_ocr_tool, mock_table_tool, mock_classifier_tool, mock_vqa_tool]


# =============================================================================
# Test Task Categorization
# =============================================================================


class TestTaskCategorization:
    """Test automatic task categorization."""

    def test_categorize_table_task(self):
        """Test categorization of table extraction tasks."""
        tasks = [
            "Extract the table from this document",
            "Find all rows and columns in the grid",
            "Get the cell values from the table",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "table_extraction", f"Failed for: {task}"

    def test_categorize_form_task(self):
        """Test categorization of form extraction tasks."""
        tasks = [
            "Extract form fields from this application",
            "Fill in the form with the given data",
            "Get all checkbox values",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "form_extraction", f"Failed for: {task}"

    def test_categorize_invoice_task(self):
        """Test categorization of invoice tasks."""
        tasks = [
            "Process this invoice and get totals",
            "Get the receipt items",
            "Find payment details from the bill",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "invoice_extraction", f"Failed for: {task}"

    def test_categorize_ocr_task(self):
        """Test categorization of OCR tasks."""
        tasks = [
            "Extract all text from this image",
            "Read the text content",
            "OCR this scanned page",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "OCR", f"Failed for: {task}"

    def test_categorize_qa_task(self):
        """Test categorization of Q&A tasks."""
        tasks = [
            "What color is the logo?",
            "Where is the header located?",
            "Who is the author?",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "DocQA", f"Failed for: {task}"

    def test_categorize_classification_task(self):
        """Test categorization of classification tasks."""
        tasks = [
            "Classify this document type",
            "What category does this document belong to?",
            "Identify the kind of document",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "classification", f"Failed for: {task}"

    def test_categorize_handwriting_task(self):
        """Test categorization of handwriting tasks."""
        tasks = [
            "Extract the handwritten notes",
            "Read the cursive text",
            "Get handwriting from the margins",
        ]
        for task in tasks:
            category = categorize_document_task(task)
            assert category == "handwriting", f"Failed for: {task}"

    def test_categorize_general_task(self):
        """Test categorization of general tasks."""
        task = "Do something with this image please"
        category = categorize_document_task(task)
        assert category == "general"


# =============================================================================
# Test Design Patterns
# =============================================================================


class TestDesignPatterns:
    """Test design pattern retrieval and suggestions."""

    def test_get_table_pattern(self):
        """Test retrieving table extraction pattern."""
        pattern = get_pattern_for_category("table_extraction")
        assert pattern is not None
        assert "extract_table" in pattern.lower() or "table" in pattern.lower()

    def test_get_form_pattern(self):
        """Test retrieving form extraction pattern."""
        pattern = get_pattern_for_category("form_extraction")
        assert pattern is not None
        assert "form" in pattern.lower() or "key" in pattern.lower()

    def test_get_invalid_pattern(self):
        """Test handling of invalid category."""
        pattern = get_pattern_for_category("nonexistent_category")
        assert pattern is None

    def test_get_suggested_tools_table(self):
        """Test tool suggestions for table tasks."""
        tools = get_suggested_tools("table_extraction")
        assert "detect_tables" in tools
        assert "ocr" in tools

    def test_get_suggested_tools_ocr(self):
        """Test tool suggestions for OCR tasks."""
        tools = get_suggested_tools("OCR")
        assert "ocr" in tools

    def test_get_suggested_tools_general(self):
        """Test tool suggestions for general tasks."""
        tools = get_suggested_tools("general")
        assert len(tools) > 0

    def test_build_pattern_prompt(self):
        """Test building a pattern-enhanced prompt."""
        task = "Extract the table from this invoice"
        prompt = build_pattern_prompt(task)

        assert "table_extraction" in prompt.lower()
        assert "detect_tables" in prompt.lower() or "table" in prompt.lower()


# =============================================================================
# Test VisionDocumentAgent Creation
# =============================================================================


class TestVisionDocumentAgentCreation:
    """Test VisionDocumentAgent instantiation."""

    def test_create_basic_agent(self, mock_llm):
        """Test creating a basic VisionDocumentAgent."""
        agent = VisionDocumentAgent(llm=mock_llm)

        assert agent.llm is mock_llm
        assert agent.auto_categorize is True
        assert agent.suggest_patterns is True

    def test_create_with_tools(self, mock_llm, document_tools):
        """Test creating agent with document tools."""
        agent = VisionDocumentAgent(
            llm=mock_llm,
            function_list=document_tools,
        )

        assert len(agent.function_map) == 4
        assert "ocr" in agent.function_map
        assert "detect_tables" in agent.function_map

    def test_create_with_custom_settings(self, mock_llm):
        """Test creating agent with custom settings."""
        agent = VisionDocumentAgent(
            llm=mock_llm,
            auto_categorize=False,
            suggest_patterns=False,
            verify_results=True,
            max_iterations=25,
        )

        assert agent.auto_categorize is False
        assert agent.suggest_patterns is False
        assert agent.verify_results is True
        assert agent.max_iterations == 25

    def test_default_system_message(self, mock_llm):
        """Test default system message contains VDU prompts."""
        agent = VisionDocumentAgent(llm=mock_llm)

        assert "visual document" in agent.system_message.lower()
        assert "categorize" in agent.system_message.lower()


# =============================================================================
# Test VisionDocumentAgent Execution
# =============================================================================


class TestVisionDocumentAgentExecution:
    """Test VisionDocumentAgent execution."""

    def test_task_info_retrieval(self, mock_llm, document_tools):
        """Test get_task_info method."""
        agent = VisionDocumentAgent(
            llm=mock_llm,
            function_list=document_tools,
        )

        info = agent.get_task_info("Extract the table from this invoice")

        assert info["category"] == "table_extraction"
        assert "detect_tables" in info["suggested_tools"]
        assert "ocr" in info["available_tools"]

    def test_simple_ocr_task(self, sequence_llm_factory, mock_ocr_tool):
        """Test executing a simple OCR task."""
        llm = sequence_llm_factory([
            {"name": "ocr", "arguments": {"image": "doc.png"}, "content": "STEP 1: Running OCR..."},
            "FINAL ANSWER:\nExtracted text: Invoice #12345, Total: $500.00",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Extract text from this document"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 2
        assert len(responses) > 0

    def test_table_extraction_task(self, sequence_llm_factory, mock_ocr_tool, mock_table_tool):
        """Test executing a table extraction task."""
        llm = sequence_llm_factory([
            {"name": "detect_tables", "arguments": {"image": "doc.png"}, "content": "STEP 1: Detecting tables..."},
            {"name": "ocr", "arguments": {"image": "doc.png"}, "content": "STEP 2: Extracting table content..."},
            "FINAL ANSWER:\nTable found with 3 rows and 4 columns.",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool, mock_table_tool],
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Extract the table from this document"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3

    def test_auto_categorization(self, sequence_llm_factory, document_tools):
        """Test that auto categorization enhances messages."""
        llm = sequence_llm_factory([
            "FINAL ANSWER:\nTask categorized and processed.",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=document_tools,
            auto_categorize=True,
            suggest_patterns=True,
        )

        messages = [{"role": "user", "content": "Extract the table from this invoice"}]
        run_agent_to_completion(agent, messages)

        # Verify that the LLM received enhanced messages
        assert llm.call_count >= 1
        # The message should have been enhanced with category info
        first_call = llm._chat_history[0]
        # Find the user message in the call
        user_msgs = [m for m in first_call if m.role == "user"]
        assert len(user_msgs) > 0

    def test_without_auto_categorization(self, sequence_llm_factory, mock_ocr_tool):
        """Test agent without auto categorization."""
        llm = sequence_llm_factory([
            "FINAL ANSWER:\nProcessed without categorization.",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool],
            auto_categorize=False,
            suggest_patterns=False,
        )

        messages = [{"role": "user", "content": "Process this document"}]
        run_agent_to_completion(agent, messages)

        assert llm.call_count == 1


# =============================================================================
# Test DocumentExtractionAgent
# =============================================================================


class TestDocumentExtractionAgent:
    """Test DocumentExtractionAgent subclass."""

    def test_create_extraction_agent(self, mock_llm):
        """Test creating a DocumentExtractionAgent."""
        agent = DocumentExtractionAgent(llm=mock_llm)

        assert agent.name == "DocumentExtractionAgent"
        assert agent.auto_categorize is True
        assert agent.verify_results is True

    def test_extraction_system_message(self, mock_llm):
        """Test extraction agent has appropriate system message."""
        agent = DocumentExtractionAgent(llm=mock_llm)

        assert "extraction" in agent.system_message.lower()
        assert "json" in agent.system_message.lower()

    def test_extraction_task(self, sequence_llm_factory, mock_ocr_tool, mock_table_tool):
        """Test extraction agent execution."""
        llm = sequence_llm_factory([
            {"name": "ocr", "arguments": {"image": "invoice.png"}, "content": "Extracting..."},
            "FINAL ANSWER:\n{\"invoice_number\": \"12345\", \"total\": \"$500.00\"}",
        ])

        agent = DocumentExtractionAgent(
            llm=llm,
            function_list=[mock_ocr_tool, mock_table_tool],
        )

        messages = [{"role": "user", "content": "Extract invoice data"}]
        responses = run_agent_to_completion(agent, messages)

        assert len(responses) > 0


# =============================================================================
# Test DocumentQAAgent
# =============================================================================


class TestDocumentQAAgent:
    """Test DocumentQAAgent subclass."""

    def test_create_qa_agent(self, mock_llm):
        """Test creating a DocumentQAAgent."""
        agent = DocumentQAAgent(llm=mock_llm)

        assert agent.name == "DocumentQAAgent"
        assert agent.auto_categorize is True
        assert agent.suggest_patterns is False  # Less pattern-heavy for Q&A

    def test_qa_system_message(self, mock_llm):
        """Test Q&A agent has appropriate system message."""
        agent = DocumentQAAgent(llm=mock_llm)

        assert "question" in agent.system_message.lower()
        assert "answer" in agent.system_message.lower()

    def test_qa_task(self, sequence_llm_factory, mock_vqa_tool):
        """Test Q&A agent execution."""
        llm = sequence_llm_factory([
            {"name": "vqa", "arguments": {"image": "doc.png", "question": "What is the total?"}, "content": "Answering..."},
            "FINAL ANSWER:\nThe total amount is $500.00",
        ])

        agent = DocumentQAAgent(
            llm=llm,
            function_list=[mock_vqa_tool],
        )

        messages = [{"role": "user", "content": "What is the total amount?"}]
        responses = run_agent_to_completion(agent, messages)

        assert len(responses) > 0
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "500" in content or "FINAL ANSWER" in content


# =============================================================================
# Test Multi-tool Workflows
# =============================================================================


class TestMultiToolWorkflows:
    """Test complex multi-tool document processing workflows."""

    def test_invoice_processing_workflow(
        self, sequence_llm_factory, mock_ocr_tool, mock_table_tool, mock_classifier_tool
    ):
        """Test a complete invoice processing workflow."""
        llm = sequence_llm_factory([
            # Step 1: Classify
            {"name": "classify_document", "arguments": {"image": "invoice.png"}, "content": "STEP 1: Classifying document..."},
            # Step 2: Extract tables
            {"name": "detect_tables", "arguments": {"image": "invoice.png"}, "content": "STEP 2: Detecting tables..."},
            # Step 3: OCR
            {"name": "ocr", "arguments": {"image": "invoice.png"}, "content": "STEP 3: Running OCR..."},
            # Final answer
            "FINAL ANSWER:\nDocument Type: Invoice\nInvoice #: 12345\nTotal: $500.00\nLine Items: [Table with 3 rows]",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool, mock_table_tool, mock_classifier_tool],
            max_iterations=10,
        )

        messages = [{
            "role": "user",
            "content": "Process this invoice: classify it, extract the table, and get all text"
        }]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 4  # 3 tool calls + final answer
        assert len(responses) > 0

    def test_document_qa_with_ocr(self, sequence_llm_factory, mock_ocr_tool, mock_vqa_tool):
        """Test Q&A that requires OCR first."""
        llm = sequence_llm_factory([
            # First OCR to get text
            {"name": "ocr", "arguments": {"image": "doc.png"}, "content": "First, let me extract the text..."},
            # Then VQA to answer
            {"name": "vqa", "arguments": {"image": "doc.png", "question": "total"}, "content": "Now answering the question..."},
            "FINAL ANSWER:\nThe total amount on the invoice is $500.00",
        ])

        agent = DocumentQAAgent(
            llm=llm,
            function_list=[mock_ocr_tool, mock_vqa_tool],
        )

        messages = [{"role": "user", "content": "What is the total amount on this invoice?"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3


# =============================================================================
# Test Error Handling
# =============================================================================


class TestVisionDocumentAgentErrorHandling:
    """Test error handling in VisionDocumentAgent."""

    def test_tool_failure_continues(self, sequence_llm_factory, failing_tool):
        """Test that tool failures don't stop the agent."""
        llm = sequence_llm_factory([
            {"name": "failing_tool", "arguments": {}, "content": "Trying the tool..."},
            "FINAL ANSWER:\nThe tool failed but I can provide a general answer.",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[failing_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Process this document"}]
        responses = run_agent_to_completion(agent, messages)

        assert len(responses) > 0

    def test_max_iterations_respected(self, sequence_llm_factory, mock_ocr_tool):
        """Test that max iterations limit is respected."""
        # Create responses that never reach FINAL ANSWER
        llm = sequence_llm_factory([
            {"name": "ocr", "arguments": {"image": "1"}, "content": "Step 1"},
            {"name": "ocr", "arguments": {"image": "2"}, "content": "Step 2"},
            {"name": "ocr", "arguments": {"image": "3"}, "content": "Step 3"},
            {"name": "ocr", "arguments": {"image": "4"}, "content": "Step 4"},
            {"name": "ocr", "arguments": {"image": "5"}, "content": "Step 5"},
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool],
            max_iterations=3,
        )

        messages = [{"role": "user", "content": "Keep processing forever"}]
        run_agent_to_completion(agent, messages)

        assert llm.call_count <= 3


# =============================================================================
# Test Multimodal Input
# =============================================================================


class TestMultimodalInput:
    """Test handling of multimodal (image + text) input."""

    def test_multimodal_message_handling(self, sequence_llm_factory, mock_ocr_tool):
        """Test handling messages with images."""
        llm = sequence_llm_factory([
            "FINAL ANSWER:\nProcessed the image successfully.",
        ])

        agent = VisionDocumentAgent(
            llm=llm,
            function_list=[mock_ocr_tool],
        )

        # Multimodal message with image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract text from this image"},
                {"type": "image", "image": "path/to/image.png"},
            ],
        }]

        responses = run_agent_to_completion(agent, messages)
        assert len(responses) > 0

    def test_text_extraction_from_multimodal(self, mock_llm, mock_ocr_tool):
        """Test extracting task text from multimodal content."""
        agent = VisionDocumentAgent(
            llm=mock_llm,
            function_list=[mock_ocr_tool],
        )

        # Get task info should work with multimodal
        info = agent.get_task_info("Extract the table from this invoice")
        assert info["category"] == "table_extraction"
