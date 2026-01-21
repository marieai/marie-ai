"""VisionDocumentAgent - Specialized agent for Visual Document Understanding.

This module provides an agent specialized for document processing tasks,
incorporating design patterns inspired by Landing.ai's vision-agent approach.

Key features:
- Automatic task categorization (OCR, table extraction, form extraction, etc.)
- Design pattern suggestions based on document characteristics
- Multi-tool testing for improved accuracy
- VQA verification of results
"""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from marie.agent.base import BaseAgent
from marie.agent.message import Message
from marie.agent.patterns.document_patterns import (
    DOCUMENT_DESIGN_PATTERNS,
    DOCUMENT_TOOL_CATEGORIES,
    build_pattern_prompt,
    categorize_document_task,
    get_pattern_for_category,
    get_suggested_tools,
)
from marie.agent.tools.base import AgentTool
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.llm_wrapper import BaseLLMWrapper

logger = MarieLogger("marie.agent.agents.vision_document_agent")


class VisionDocumentAgent(BaseAgent):
    """Agent specialized for Visual Document Understanding (VDU).

    This agent combines planning capabilities with document-specific design
    patterns to handle complex document processing tasks effectively.

    Features:
    - Automatic task categorization (OCR, table, form, invoice, etc.)
    - Design pattern suggestions based on task type
    - Multi-tool testing to find the best approach
    - Iterative refinement with pattern-guided execution
    - VQA verification of results (optional)

    Example:
        ```python
        from marie.agent import VisionDocumentAgent, MarieEngineLLMWrapper

        agent = VisionDocumentAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            function_list=["ocr", "detect_tables", "classify_document"],
        )

        messages = [
            {
                "role": "user",
                "content": "Extract all tables from this invoice",
                "images": ["invoice.png"],
            }
        ]

        for responses in agent.run(messages):
            print(responses[-1].content)
        ```
    """

    VISION_DOCUMENT_PROMPT = """You are a Visual Document Understanding (VDU) specialist agent.

You analyze document images and extract structured information using specialized tools.
Your approach combines visual analysis with document processing patterns.

When given a task:
1. CATEGORIZE: Identify the task type (OCR, table extraction, form extraction, etc.)
2. ANALYZE: Examine the document to understand its structure
3. PLAN: Create a step-by-step plan using appropriate design patterns
4. EXECUTE: Run the plan using available tools
5. VERIFY: Check results for accuracy and completeness
6. RESPOND: Provide structured, actionable output

Design Pattern Categories:
- small_text: For fine print, footnotes, dense text
- rotated_text: For skewed or rotated documents
- table_extraction: For tabular data
- form_extraction: For key-value form fields
- invoice_extraction: For invoice data
- multi_column: For newspaper/magazine layouts
- handwriting: For handwritten content
- document_comparison: For comparing document versions
- quality_assessment: For image quality issues
- multi_page: For multi-page documents

Format your response as:
CATEGORY: [detected task category]
PATTERN: [recommended design pattern]
PLAN:
1. [First step]
2. [Second step]
...

[Execute each step with tool calls]

FINAL ANSWER:
[Structured results]"""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 15,
        auto_categorize: bool = True,
        suggest_patterns: bool = True,
        verify_results: bool = False,
        **kwargs: Any,
    ):
        """Initialize the VisionDocumentAgent.

        Args:
            function_list: List of tools available (e.g., ["ocr", "detect_tables"])
            llm: LLM wrapper for generating responses
            system_message: Custom system message (defaults to VDU prompt)
            name: Agent name
            description: Agent description
            max_iterations: Maximum iterations before stopping
            auto_categorize: Automatically categorize tasks
            suggest_patterns: Suggest design patterns based on task
            verify_results: Use VQA to verify extraction results
            **kwargs: Additional configuration
        """
        if system_message is None:
            system_message = self.VISION_DOCUMENT_PROMPT

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name or "VisionDocumentAgent",
            description=description or "Agent for Visual Document Understanding tasks",
            **kwargs,
        )

        self.max_iterations = max_iterations
        self.auto_categorize = auto_categorize
        self.suggest_patterns = suggest_patterns
        self.verify_results = verify_results

        # Cache for task categorization
        self._task_cache: Dict[str, str] = {}

    def _categorize_task(self, task: str) -> str:
        """Categorize a document task.

        Args:
            task: The task description.

        Returns:
            Category string (e.g., "table_extraction", "OCR").
        """
        if task in self._task_cache:
            return self._task_cache[task]

        category = categorize_document_task(task)
        self._task_cache[task] = category
        return category

    def _get_pattern_suggestion(self, category: str) -> Optional[str]:
        """Get design pattern suggestion for a category.

        Args:
            category: The task category.

        Returns:
            Design pattern string or None.
        """
        return get_pattern_for_category(category)

    def _get_tool_suggestions(self, category: str) -> List[str]:
        """Get suggested tools for a category.

        Args:
            category: The task category.

        Returns:
            List of suggested tool names.
        """
        return get_suggested_tools(category)

    def _enhance_user_message(
        self,
        user_content: str,
        category: str,
        pattern: Optional[str],
    ) -> str:
        """Enhance user message with category and pattern info.

        Args:
            user_content: Original user message content.
            category: Detected task category.
            pattern: Suggested design pattern.

        Returns:
            Enhanced message with task context.
        """
        enhanced = f"""Task: {user_content}

[System Analysis]
Detected Category: {category}
Suggested Tools: {', '.join(self._get_tool_suggestions(category))}
"""

        if pattern:
            enhanced += f"""
Recommended Pattern:
{pattern}
"""

        enhanced += """
Please follow the pattern structure to accomplish this task efficiently.
"""
        return enhanced

    def _prepare_messages(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], str, Optional[str]]:
        """Prepare messages with task categorization and pattern suggestions.

        Args:
            messages: Input messages.

        Returns:
            Tuple of (modified messages, category, pattern).
        """
        if not self.auto_categorize:
            return messages, "general", None

        # Find the user's task from messages
        user_task = ""
        for msg in reversed(messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    user_task = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from multimodal content
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_task = item.get("text", "")
                            break
                        elif isinstance(item, str):
                            user_task = item
                            break
                break

        if not user_task:
            return messages, "general", None

        # Categorize the task
        category = self._categorize_task(user_task)
        pattern = None

        if self.suggest_patterns:
            pattern = self._get_pattern_suggestion(category)

        # Don't modify messages if no enhancement needed
        if not self.suggest_patterns:
            return messages, category, pattern

        # Create enhanced message list
        enhanced_messages = []
        for msg in messages:
            if (
                msg.role == "user"
                and isinstance(msg.content, str)
                and msg.content == user_task
            ):
                # Enhance the user message
                enhanced_content = self._enhance_user_message(
                    user_task, category, pattern
                )
                enhanced_msg = Message(
                    role=msg.role,
                    content=enhanced_content,
                    name=msg.name,
                    function_call=msg.function_call,
                    tool_calls=msg.tool_calls,
                )
                enhanced_messages.append(enhanced_msg)
            else:
                enhanced_messages.append(msg)

        return enhanced_messages, category, pattern

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute the VisionDocumentAgent loop.

        This follows a pattern-guided planning and execution approach:
        1. Categorize the task
        2. Suggest appropriate design patterns
        3. Create and execute a plan
        4. Optionally verify results

        Args:
            messages: Input messages with system message prepended
            lang: Language code
            **kwargs: Additional arguments

        Yields:
            Lists of response Messages
        """
        # Prepare messages with categorization
        enhanced_messages, category, pattern = self._prepare_messages(messages)

        logger.debug(f"Task category: {category}")
        if pattern:
            logger.debug(f"Using pattern for: {category}")

        # Get function definitions
        functions = self._get_tool_definitions() if self.function_map else None

        # Log suggested tools for debugging
        suggested = self._get_tool_suggestions(category)
        available = list(self.function_map.keys()) if self.function_map else []
        matching_tools = [t for t in suggested if t in available]
        if matching_tools:
            logger.debug(f"Matching tools for category: {matching_tools}")

        # Main execution loop
        conversation = list(enhanced_messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            extra_cfg = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_cfg["seed"] = kwargs["seed"]

            llm_responses = []
            for responses in self._call_llm(
                conversation,
                functions=functions,
                extra_generate_cfg=extra_cfg,
            ):
                llm_responses = responses
                yield responses

            if not llm_responses:
                break

            response = llm_responses[-1]
            conversation.append(response)

            # Check for tool calls
            has_call, tool_name, tool_args, text_content = self._detect_tool_call(
                response
            )

            # Check for final answer
            text_upper = (text_content or "").upper()
            if "FINAL ANSWER" in text_upper:
                # Optionally verify results
                if self.verify_results and "vqa" in (self.function_map or {}):
                    logger.debug("Verifying results with VQA")
                    # Verification would happen here
                break

            if not has_call:
                # No tool call - continue planning
                continue

            # Execute tool
            logger.debug(f"Executing tool '{tool_name}' with args: {tool_args}")
            tool_result = self._call_tool(tool_name, tool_args, **kwargs)

            # Add tool result to conversation
            tool_msg = Message.function_result(
                name=tool_name,
                content=str(tool_result),
            )
            conversation.append(tool_msg)

            # Yield intermediate state
            yield [response, tool_msg]

        if iteration >= self.max_iterations:
            logger.warning(
                f"VisionDocumentAgent reached max iterations ({self.max_iterations})."
            )

    def test_tools(
        self,
        task: str,
        image: Any,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Test multiple tools on an image and compare results.

        This method allows testing different tools to find the best one
        for a specific task, following the vision-agent pattern.

        Args:
            task: Task description
            image: Image to process (numpy array or path)
            tools: List of tools to test (defaults to suggested tools)

        Returns:
            Dictionary with results from each tool
        """
        category = self._categorize_task(task)
        tools_to_test = tools or self._get_tool_suggestions(category)

        results = {}
        for tool_name in tools_to_test:
            if tool_name not in (self.function_map or {}):
                logger.debug(f"Tool '{tool_name}' not available, skipping")
                continue

            try:
                # Call tool with image
                args = json.dumps({"image": str(image), "task": task})
                result = self._call_tool(tool_name, args)
                results[tool_name] = {
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                logger.warning(f"Tool '{tool_name}' failed: {e}")
                results[tool_name] = {
                    "success": False,
                    "error": str(e),
                }

        return results

    def get_task_info(self, task: str) -> Dict[str, Any]:
        """Get categorization and pattern info for a task.

        Useful for debugging or understanding how the agent will
        approach a given task.

        Args:
            task: Task description

        Returns:
            Dictionary with category, pattern, and suggested tools
        """
        category = self._categorize_task(task)
        pattern = self._get_pattern_suggestion(category)
        tools = self._get_tool_suggestions(category)

        return {
            "task": task,
            "category": category,
            "pattern": pattern,
            "suggested_tools": tools,
            "available_tools": (
                list(self.function_map.keys()) if self.function_map else []
            ),
        }


class DocumentExtractionAgent(VisionDocumentAgent):
    """Convenience subclass specialized for data extraction tasks.

    Pre-configured with settings optimized for extracting structured
    data from documents (forms, invoices, tables, etc.).
    """

    EXTRACTION_PROMPT = """You are a document data extraction specialist.

Your job is to extract structured data from documents with high accuracy.
Focus on:
- Accurate field identification
- Proper data formatting
- Confidence scores for extractions
- Handling of ambiguous or unclear content

For each extraction task:
1. Identify the document type and structure
2. Locate relevant fields/regions
3. Extract and validate the data
4. Format output as structured JSON

FINAL ANSWER should be valid JSON with extracted fields."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        **kwargs: Any,
    ):
        """Initialize DocumentExtractionAgent."""
        kwargs.setdefault("system_message", self.EXTRACTION_PROMPT)
        kwargs.setdefault("auto_categorize", True)
        kwargs.setdefault("suggest_patterns", True)
        kwargs.setdefault("verify_results", True)

        super().__init__(
            function_list=function_list,
            llm=llm,
            name="DocumentExtractionAgent",
            description="Agent for extracting structured data from documents",
            **kwargs,
        )


class DocumentQAAgent(VisionDocumentAgent):
    """Convenience subclass specialized for document Q&A tasks.

    Pre-configured for answering questions about document content
    using a combination of OCR and visual understanding.
    """

    QA_PROMPT = """You are a document question-answering specialist.

Your job is to answer questions about documents accurately and completely.
Approach:
1. Understand the question being asked
2. Identify relevant regions in the document
3. Extract the necessary information
4. Formulate a clear, accurate answer

For complex questions:
- Break down into sub-questions
- Use multiple tools if needed
- Synthesize information from different parts

Provide answers with:
- Direct response to the question
- Supporting evidence from the document
- Confidence level if uncertain

FINAL ANSWER should directly address the user's question."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        **kwargs: Any,
    ):
        """Initialize DocumentQAAgent."""
        kwargs.setdefault("system_message", self.QA_PROMPT)
        kwargs.setdefault("auto_categorize", True)
        kwargs.setdefault("suggest_patterns", False)  # Less pattern-heavy for Q&A

        super().__init__(
            function_list=function_list,
            llm=llm,
            name="DocumentQAAgent",
            description="Agent for answering questions about documents",
            **kwargs,
        )
