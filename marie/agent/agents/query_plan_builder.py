"""QueryPlanBuilderAgent - Generates query plans from natural language.

This agent takes natural language descriptions and generates valid
QueryPlanDefinition JSON that can be rendered in the Query Plan Editor.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, List, Optional

from marie.agent.agents.assistant import ReactAgent
from marie.agent.message import Message
from marie.agent.tools.query_plan_tools import (
    get_node_defaults,
    get_node_types,
    validate_query_plan,
)
from marie.logging_core.logger import MarieLogger
from marie.schemas.query_plan import QueryPlanDefinition

logger = MarieLogger("marie.agent.agents.query_plan_builder")


QUERY_PLAN_BUILDER_SYSTEM_PROMPT = """You are an expert workflow designer for Marie AI's Query Plan system. Your task is to generate valid query plan JSON from natural language descriptions.

## Your Capabilities
1. Generate query plans with nodes and their dependencies (DAG structure)
2. Use the appropriate node types for each step
3. Ensure all plans have proper Start and End nodes
4. Validate plans before returning them

## Node Types Available
Use the `get_node_types` tool to see all available nodes. Key categories:
- Flow Control: START_END (required at beginning and end)
- AI & LLM: PROMPT, LLM, AGENT (for AI-powered processing)
- Control Flow: BRANCH (conditional), SWITCH (value-based), MERGER_ENHANCED (merge paths)
- Quality: GUARDRAIL (validate outputs with metrics)
- Human-in-the-Loop: HITL_ROUTER, HITL_APPROVAL, HITL_CORRECTION
- Annotators: ANNOTATOR_LLM, ANNOTATOR_TABLE, ANNOTATOR_EMBEDDING, ANNOTATOR_REGEX
- RAG: RAG_SEARCH, RAG_INGEST, RAG_DELETE, CONTEXT_CACHE

## Query Plan Structure
```json
{
  "nodes": [
    {
      "task_id": "uuid-string",
      "query_str": "Human-readable label",
      "dependencies": ["upstream-task-id"],
      "node_type": "COMPUTE|BRANCH|SWITCH|MERGER|GUARDRAIL",
      "definition": {
        "method": "NODE_METHOD",
        "endpoint": "executor://path",
        "params": {}
      }
    }
  ]
}
```

## Rules
1. ALWAYS start with a START_END node (no dependencies)
2. ALWAYS end with a START_END node (depends on final processing nodes)
3. Use unique UUIDs for task_id
4. Dependencies form a DAG - no cycles allowed
5. Use `get_node_defaults` to get correct default configuration for each node type
6. Use `validate_query_plan` to verify your plan before returning
7. Match node_type to the method:
   - BRANCH method → node_type "BRANCH"
   - SWITCH method → node_type "SWITCH"
   - MERGER_ENHANCED method → node_type "MERGER"
   - GUARDRAIL method → node_type "GUARDRAIL"
   - All others → node_type "COMPUTE"

## Response Format
After generating a valid plan, return it in this format:
```
REASONING: <Brief explanation of the workflow structure>

PLAN:
```json
{...your query plan JSON...}
```
```

## Examples

### Simple Document Processing
User: "OCR a document and extract text"
```json
{
  "nodes": [
    {"task_id": "start-1", "query_str": "Start", "dependencies": [], "node_type": "COMPUTE", "definition": {"method": "START_END", "endpoint": "noop"}},
    {"task_id": "ocr-2", "query_str": "OCR Document", "dependencies": ["start-1"], "node_type": "COMPUTE", "definition": {"method": "EXECUTOR_ENDPOINT", "endpoint": "ocr://extract"}},
    {"task_id": "end-3", "query_str": "End", "dependencies": ["ocr-2"], "node_type": "COMPUTE", "definition": {"method": "START_END", "endpoint": "noop"}}
  ]
}
```

### Classification with Human Review
User: "Classify documents, send low confidence ones to human review"
```json
{
  "nodes": [
    {"task_id": "start-1", "query_str": "Start", "dependencies": [], "node_type": "COMPUTE", "definition": {"method": "START_END", "endpoint": "noop"}},
    {"task_id": "classify-2", "query_str": "Classify Document", "dependencies": ["start-1"], "node_type": "COMPUTE", "definition": {"method": "PROMPT", "endpoint": ""}},
    {"task_id": "router-3", "query_str": "Route by Confidence", "dependencies": ["classify-2"], "node_type": "COMPUTE", "definition": {"method": "HITL_ROUTER", "auto_approve_threshold": 0.9, "human_review_threshold": 0.7}},
    {"task_id": "review-4", "query_str": "Human Review", "dependencies": ["router-3"], "node_type": "COMPUTE", "definition": {"method": "HITL_APPROVAL", "title": "Review Classification", "priority": "medium"}},
    {"task_id": "merge-5", "query_str": "Merge Results", "dependencies": ["router-3", "review-4"], "node_type": "MERGER", "definition": {"method": "MERGER_ENHANCED", "merge_strategy": "WAIT_ALL_ACTIVE"}},
    {"task_id": "end-6", "query_str": "End", "dependencies": ["merge-5"], "node_type": "COMPUTE", "definition": {"method": "START_END", "endpoint": "noop"}}
  ]
}
```
"""


class QueryPlanBuilderAgent(ReactAgent):
    """Agent that generates query plans from natural language descriptions.

    Uses the ReAct pattern to:
    1. Understand the user's requirements
    2. Look up available node types
    3. Generate a valid query plan
    4. Validate the plan before returning
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        system_message: Optional[str] = None,
        name: str = "query_plan_builder",
        description: str = "Generates query plans from natural language",
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        """Initialize QueryPlanBuilderAgent.

        Args:
            llm: LLM wrapper to use
            system_message: Custom system message (uses default if None)
            name: Agent name
            description: Agent description
            max_iterations: Maximum iterations for plan generation
            **kwargs: Additional arguments
        """
        if system_message is None:
            system_message = QUERY_PLAN_BUILDER_SYSTEM_PROMPT

        # Tools for plan generation
        function_list = [
            "get_node_types",
            "get_node_defaults",
            "validate_query_plan",
        ]

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            max_iterations=max_iterations,
            **kwargs,
        )

        # Conversation history for refinement
        self._conversation_history: Dict[str, List[Message]] = {}

    def generate_plan(
        self,
        description: str,
        vertical: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a query plan from a natural language description.

        Args:
            description: Natural language description of the workflow
            vertical: Optional business vertical for context
            conversation_id: Optional ID to maintain conversation context

        Returns:
            Dict with success, plan, reasoning, conversation_id, and error
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Build the user prompt
        user_prompt = f"Generate a query plan for: {description}"
        if vertical:
            user_prompt += f"\n\nContext: This is for the {vertical} industry vertical."

        # Get or create conversation history
        messages = self._conversation_history.get(conversation_id, [])
        messages.append(Message.user(content=user_prompt))

        # Run the agent
        final_response = None
        try:
            for responses in self.run(messages, **kwargs):
                if responses:
                    final_response = responses[-1]

            # Store updated conversation
            if final_response:
                messages.append(final_response)
            self._conversation_history[conversation_id] = messages

            # Parse the response
            return self._parse_response(final_response, conversation_id)

        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return {
                "success": False,
                "plan": None,
                "reasoning": None,
                "conversation_id": conversation_id,
                "error": str(e),
            }

    def refine_plan(
        self,
        current_plan: Dict[str, Any],
        feedback: str,
        conversation_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Refine an existing query plan based on user feedback.

        Args:
            current_plan: Current QueryPlanDefinition as dict
            feedback: User's refinement request
            conversation_id: Conversation ID for context

        Returns:
            Dict with success, plan, changes, conversation_id, and error
        """
        # Build refinement prompt
        plan_json = json.dumps(current_plan, indent=2)
        user_prompt = f"""Refine this query plan based on the following feedback:

Current Plan:
```json
{plan_json}
```

Feedback: {feedback}

Apply the requested changes and return the updated plan. List the changes you made."""

        # Get conversation history
        messages = self._conversation_history.get(conversation_id, [])
        messages.append(Message.user(content=user_prompt))

        # Run the agent
        final_response = None
        try:
            for responses in self.run(messages, **kwargs):
                if responses:
                    final_response = responses[-1]

            # Store updated conversation
            if final_response:
                messages.append(final_response)
            self._conversation_history[conversation_id] = messages

            # Parse the response
            result = self._parse_response(final_response, conversation_id)

            # Extract changes from reasoning
            changes = []
            if result.get("reasoning"):
                # Simple extraction of changes mentioned
                reasoning = result["reasoning"]
                if "change" in reasoning.lower() or "add" in reasoning.lower():
                    changes = [reasoning]

            result["changes"] = changes
            return result

        except Exception as e:
            logger.error(f"Error refining plan: {e}")
            return {
                "success": False,
                "plan": None,
                "changes": [],
                "conversation_id": conversation_id,
                "error": str(e),
            }

    def _parse_response(
        self,
        response: Optional[Message],
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Parse the agent's response to extract plan and reasoning.

        Args:
            response: The agent's final response message
            conversation_id: Conversation ID

        Returns:
            Dict with success, plan, reasoning, conversation_id, error
        """
        if not response or not response.text_content:
            return {
                "success": False,
                "plan": None,
                "reasoning": None,
                "conversation_id": conversation_id,
                "error": "No response from agent",
            }

        content = response.text_content

        # Extract reasoning
        reasoning = None
        if "REASONING:" in content:
            reasoning_start = content.find("REASONING:") + len("REASONING:")
            reasoning_end = content.find("PLAN:", reasoning_start)
            if reasoning_end == -1:
                reasoning_end = content.find("```json", reasoning_start)
            if reasoning_end != -1:
                reasoning = content[reasoning_start:reasoning_end].strip()

        # Extract JSON plan
        plan = None
        json_start = content.find("```json")
        if json_start != -1:
            json_start = content.find("\n", json_start) + 1
            json_end = content.find("```", json_start)
            if json_end != -1:
                try:
                    plan_json = content[json_start:json_end].strip()
                    plan = json.loads(plan_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse plan JSON: {e}")

        # Try to find inline JSON if no code block
        if plan is None:
            try:
                # Look for JSON object in content
                brace_start = content.find("{")
                if brace_start != -1:
                    # Find matching closing brace
                    depth = 0
                    for i, char in enumerate(content[brace_start:]):
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = content[brace_start : brace_start + i + 1]
                                plan = json.loads(json_str)
                                break
            except json.JSONDecodeError:
                pass

        if plan:
            # Validate the plan
            try:
                QueryPlanDefinition.model_validate(plan)
                return {
                    "success": True,
                    "plan": plan,
                    "reasoning": reasoning,
                    "conversation_id": conversation_id,
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "plan": plan,
                    "reasoning": reasoning,
                    "conversation_id": conversation_id,
                    "error": f"Plan validation failed: {e}",
                }

        return {
            "success": False,
            "plan": None,
            "reasoning": reasoning,
            "conversation_id": conversation_id,
            "error": "Could not extract plan from response",
        }

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history for a given ID.

        Args:
            conversation_id: Conversation ID to clear
        """
        if conversation_id in self._conversation_history:
            del self._conversation_history[conversation_id]
