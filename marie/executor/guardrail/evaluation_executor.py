"""
Guardrail Evaluation Executor for running custom Python evaluation functions.

This executor receives evaluation requests from GUARDRAIL nodes and returns
scores with pass/fail status. It supports a registry of custom evaluation
functions that can be called by name.

Example usage in GUARDRAIL metric configuration:
    GuardrailMetric(
        type=GuardrailMetricType.EXECUTOR,
        name="PII Check",
        threshold=1.0,
        params={
            "endpoint": "guardrail_executor://evaluate",
            "function": "check_no_pii",
            "config": {}
        }
    )
"""

import asyncio
import inspect
import json
import re
from typing import Any, Callable, Dict, Optional

from docarray import DocList
from docarray.documents import TextDoc

from marie import requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger


class GuardrailEvaluationExecutor(MarieExecutor):
    """
    Executor for running custom Python evaluation functions.

    Receives evaluation requests from GUARDRAIL nodes and returns
    scores with pass/fail status.

    The executor maintains a registry of evaluation functions that can
    be called by name. Functions can be registered either:
    1. At initialization via the `evaluation_functions` parameter
    2. Dynamically via the `register_function()` method
    3. Using the built-in functions provided by default

    Each evaluation function should have the signature:
        fn(input_data: Any, context: Dict, config: Dict) -> Dict

    And return a dict with:
        {"passed": bool, "score": float, "feedback": str}
    """

    def __init__(
        self,
        evaluation_functions: Optional[Dict[str, Callable]] = None,
        register_builtins: bool = True,
        **kwargs,
    ):
        """
        Initialize the GuardrailEvaluationExecutor.

        Args:
            evaluation_functions: Dict mapping function names to callables
            register_builtins: Whether to register built-in evaluation functions
            **kwargs: Additional arguments passed to MarieExecutor
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__)

        # Registry of evaluation functions: {"fn_name": callable}
        self.evaluation_registry: Dict[str, Callable] = {}

        # Register built-in functions
        if register_builtins:
            self._register_builtins()

        # Register user-provided functions
        if evaluation_functions:
            for name, fn in evaluation_functions.items():
                self.register_function(name, fn)

        self.logger.info(
            f"GuardrailEvaluationExecutor initialized with "
            f"{len(self.evaluation_registry)} evaluation functions"
        )

    def _register_builtins(self):
        """Register built-in evaluation functions."""
        self.register_function("check_no_pii", check_no_pii)
        self.register_function("check_json_structure", check_json_structure)
        self.register_function("check_no_profanity", check_no_profanity)
        self.register_function("check_sentiment", check_sentiment)
        self.register_function("default", default_evaluation)

    def register_function(self, name: str, fn: Callable) -> None:
        """
        Register a custom evaluation function.

        Args:
            name: Name to register the function under
            fn: Callable with signature fn(input_data, context, config) -> Dict
        """
        if not callable(fn):
            raise ValueError(f"Function {name} must be callable")

        self.evaluation_registry[name] = fn
        self.logger.debug(f"Registered evaluation function: {name}")

    def unregister_function(self, name: str) -> bool:
        """
        Unregister an evaluation function.

        Args:
            name: Name of the function to unregister

        Returns:
            True if function was unregistered, False if not found
        """
        if name in self.evaluation_registry:
            del self.evaluation_registry[name]
            self.logger.debug(f"Unregistered evaluation function: {name}")
            return True
        return False

    def list_functions(self) -> list:
        """Return list of registered function names."""
        return list(self.evaluation_registry.keys())

    @requests(on="/evaluate")
    async def evaluate(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """
        Run evaluation function on input data.

        Parameters:
            - function: Name of evaluation function to run
            - input_data: Data to evaluate
            - context: Execution context
            - config: Function-specific configuration

        Returns doc with JSON: {"passed": bool, "score": float, "feedback": str}
        """
        function_name = parameters.get("function", "default")
        input_data = parameters.get("input_data")
        context = parameters.get("context", {})
        config = parameters.get("config", {})

        self.logger.debug(f"Evaluating with function: {function_name}")

        try:
            fn = self.evaluation_registry.get(function_name)

            if fn is None:
                result = {
                    "passed": False,
                    "score": 0.0,
                    "feedback": f"Unknown evaluation function: {function_name}. "
                    f"Available: {', '.join(self.list_functions())}",
                }
            else:
                # Call the registered function (async or sync)
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(input_data, context, config)
                else:
                    result = fn(input_data, context, config)

                # Validate result structure
                if not isinstance(result, dict):
                    result = {
                        "passed": bool(result),
                        "score": 1.0 if result else 0.0,
                        "feedback": "Function returned non-dict result",
                    }

                # Ensure required fields
                result.setdefault("passed", False)
                result.setdefault("score", 0.0)
                result.setdefault("feedback", "")

        except Exception as e:
            self.logger.error(f"Evaluation error in {function_name}: {e}")
            result = {
                "passed": False,
                "score": 0.0,
                "feedback": f"Evaluation error: {str(e)}",
            }

        # Write result to doc
        if docs and len(docs) > 0:
            docs[0].text = json.dumps(result)
        else:
            doc = TextDoc(text=json.dumps(result))
            docs = DocList[TextDoc]([doc])

        return docs

    @requests(on="/list")
    async def list_available_functions(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """List all available evaluation functions."""
        functions_info = []

        for name, fn in self.evaluation_registry.items():
            info = {
                "name": name,
                "doc": fn.__doc__.strip() if fn.__doc__ else None,
                "is_async": asyncio.iscoroutinefunction(fn),
            }

            # Try to get signature
            try:
                sig = inspect.signature(fn)
                info["params"] = [p.name for p in sig.parameters.values()]
            except (ValueError, TypeError):
                info["params"] = ["input_data", "context", "config"]

            functions_info.append(info)

        result = {"functions": functions_info, "count": len(functions_info)}

        if docs and len(docs) > 0:
            docs[0].text = json.dumps(result)
        else:
            doc = TextDoc(text=json.dumps(result))
            docs = DocList[TextDoc]([doc])

        return docs


# ==============================================================================
# Built-in Evaluation Functions
# ==============================================================================


def default_evaluation(input_data: Any, context: Dict, config: Dict) -> Dict[str, Any]:
    """
    Default evaluation that always passes.
    Useful for testing or as a placeholder.
    """
    return {
        "passed": True,
        "score": 1.0,
        "feedback": "Default evaluation (always passes)",
    }


def check_no_pii(input_data: Any, context: Dict, config: Dict) -> Dict[str, Any]:
    """
    Check that output doesn't contain common PII patterns.

    Detects:
        - Social Security Numbers (XXX-XX-XXXX)
        - Credit card numbers (16 digits)
        - Email addresses (optional, via config)
        - Phone numbers (optional, via config)

    Config options:
        - check_ssn: bool (default: True)
        - check_credit_card: bool (default: True)
        - check_email: bool (default: False)
        - check_phone: bool (default: False)
    """
    text = str(input_data) if input_data is not None else ""

    pii_found = []

    # SSN pattern: XXX-XX-XXXX
    if config.get("check_ssn", True):
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        if re.search(ssn_pattern, text):
            pii_found.append("SSN")

    # Credit card: 16 digits (with optional spaces/dashes)
    if config.get("check_credit_card", True):
        cc_patterns = [
            r"\b\d{16}\b",  # 16 consecutive digits
            r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b",  # Formatted
        ]
        for pattern in cc_patterns:
            if re.search(pattern, text):
                pii_found.append("Credit Card")
                break

    # Email (optional)
    if config.get("check_email", False):
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if re.search(email_pattern, text):
            pii_found.append("Email")

    # Phone (optional)
    if config.get("check_phone", False):
        phone_patterns = [
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # US format
            r"\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b",  # (XXX) XXX-XXXX
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text):
                pii_found.append("Phone")
                break

    if pii_found:
        return {
            "passed": False,
            "score": 0.0,
            "feedback": f"PII detected: {', '.join(pii_found)}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "feedback": "No PII detected",
    }


def check_json_structure(
    input_data: Any, context: Dict, config: Dict
) -> Dict[str, Any]:
    """
    Validate that input data has required JSON structure.

    Config options:
        - required_fields: List of field names that must be present
        - required_types: Dict mapping field names to expected types
          (e.g., {"name": "str", "age": "int", "items": "list"})
    """
    required_fields = config.get("required_fields", [])
    required_types = config.get("required_types", {})

    # Handle string input (try to parse as JSON)
    if isinstance(input_data, str):
        try:
            import json

            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            return {
                "passed": False,
                "score": 0.0,
                "feedback": "Input is not valid JSON",
            }

    if not isinstance(input_data, dict):
        return {
            "passed": False,
            "score": 0.0,
            "feedback": f"Input must be a dict, got {type(input_data).__name__}",
        }

    # Check required fields
    missing_fields = [f for f in required_fields if f not in input_data]
    if missing_fields:
        return {
            "passed": False,
            "score": 0.0,
            "feedback": f"Missing required fields: {', '.join(missing_fields)}",
        }

    # Check types
    type_errors = []
    type_map = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": (int, float),
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict,
    }

    for field, expected_type in required_types.items():
        if field in input_data:
            expected = type_map.get(expected_type.lower(), str)
            if not isinstance(input_data[field], expected):
                actual = type(input_data[field]).__name__
                type_errors.append(f"{field}: expected {expected_type}, got {actual}")

    if type_errors:
        return {
            "passed": False,
            "score": 0.0,
            "feedback": f"Type errors: {'; '.join(type_errors)}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "feedback": "JSON structure is valid",
    }


def check_no_profanity(input_data: Any, context: Dict, config: Dict) -> Dict[str, Any]:
    """
    Check that output doesn't contain profanity or inappropriate content.

    This is a simple pattern-based check. For production use, consider
    integrating with a dedicated content moderation service.

    Config options:
        - additional_words: List of additional words to check
        - case_sensitive: bool (default: False)
    """
    text = str(input_data) if input_data is not None else ""

    if not config.get("case_sensitive", False):
        text = text.lower()

    # Basic word list (minimal for demonstration)
    # In production, use a comprehensive profanity filter library
    blocked_patterns = config.get("blocked_patterns", [])

    found = []
    for pattern in blocked_patterns:
        if pattern.lower() in text:
            found.append(pattern)

    if found:
        return {
            "passed": False,
            "score": 0.0,
            "feedback": f"Inappropriate content detected: {len(found)} matches",
        }

    return {
        "passed": True,
        "score": 1.0,
        "feedback": "No inappropriate content detected",
    }


def check_sentiment(input_data: Any, context: Dict, config: Dict) -> Dict[str, Any]:
    """
    Basic sentiment check using keyword matching.

    This is a simple implementation. For production use, consider
    integrating with a proper sentiment analysis model.

    Config options:
        - min_sentiment: Minimum sentiment score (-1 to 1, default: -0.5)
        - max_sentiment: Maximum sentiment score (-1 to 1, default: 1.0)
    """
    text = str(input_data).lower() if input_data is not None else ""

    # Simple keyword-based sentiment (very basic)
    positive_words = [
        "good",
        "great",
        "excellent",
        "happy",
        "positive",
        "success",
        "wonderful",
        "amazing",
    ]
    negative_words = [
        "bad",
        "terrible",
        "awful",
        "sad",
        "negative",
        "failure",
        "horrible",
        "worst",
    ]

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    total = positive_count + negative_count
    if total == 0:
        sentiment = 0.0  # Neutral
    else:
        sentiment = (positive_count - negative_count) / total

    # Normalize to 0-1 score
    score = (sentiment + 1) / 2

    min_sentiment = config.get("min_sentiment", -0.5)
    max_sentiment = config.get("max_sentiment", 1.0)

    passed = min_sentiment <= sentiment <= max_sentiment

    return {
        "passed": passed,
        "score": score,
        "feedback": f"Sentiment score: {sentiment:.2f} "
        f"(range: [{min_sentiment}, {max_sentiment}])",
    }
