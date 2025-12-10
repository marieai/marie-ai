"""
JSONPath expression evaluation for branching conditions.
Uses jsonpath-ng-ext for full JSONPath support with extensions.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from jsonpath_ng import parse as jsonpath_parse
    from jsonpath_ng.ext import parse as jsonpath_ext_parse

    JSONPATH_AVAILABLE = True
except ImportError:
    JSONPATH_AVAILABLE = False

from marie.logging_core.predefined import default_logger as logger


class JSONPathEvaluator:
    """
    Evaluates JSONPath expressions against context data.

    Supports:
    - Standard JSONPath syntax ($.field.nested)
    - Extended operators (filters, unions, regex)
    - Type-aware comparisons
    - Array operations

    Examples:
        $.metadata.document_type
        $.results[*].confidence
        $.data[?(@.score > 0.8)]
        $.metadata.tags[?(@=~"invoice.*")]
    """

    def __init__(self, use_extended: bool = True):
        """
        Initialize evaluator.

        Args:
            use_extended: Use extended parser (jsonpath-ng-ext) for advanced features
        """
        if not JSONPATH_AVAILABLE:
            raise ImportError(
                "JSONPath support requires jsonpath-ng-ext. "
                "Install with: pip install jsonpath-ng-ext"
            )

        self.use_extended = use_extended
        self._cache = {}  # Cache compiled expressions

    def evaluate(
        self, path_expression: str, context: Dict[str, Any], default: Any = None
    ) -> Any:
        """
        Evaluate JSONPath expression against context.

        Args:
            path_expression: JSONPath expression (e.g., "$.metadata.document_type")
            context: JSON data to query
            default: Default value if path not found

        Returns:
            Matched value(s). Returns list if multiple matches, single value if one match.

        Examples:
            evaluate("$.metadata.doc_type", {"metadata": {"doc_type": "invoice"}})
            # Returns: "invoice"

            evaluate("$.results[*].score", {"results": [{"score": 0.9}, {"score": 0.8}]})
            # Returns: [0.9, 0.8]
        """
        try:
            # Compile or retrieve from cache
            jsonpath_expr = self._get_compiled_expression(path_expression)

            # Find matches
            matches = jsonpath_expr.find(context)

            if not matches:
                return default

            # Extract values
            values = [match.value for match in matches]

            # Return single value if only one match, else return list
            return values[0] if len(values) == 1 else values

        except Exception as e:
            logger.error(f"JSONPath evaluation error for '{path_expression}': {e}")
            return default

    def evaluate_to_bool(self, path_expression: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate JSONPath expression and convert result to boolean.

        Useful for filter expressions that should return True/False.

        Examples:
            evaluate_to_bool("$.results[?(@.score > 0.8)]", context)
            # Returns True if any results have score > 0.8
        """
        result = self.evaluate(path_expression, context)

        if result is None:
            return False

        if isinstance(result, bool):
            return result

        if isinstance(result, (list, tuple)):
            return len(result) > 0

        if isinstance(result, (int, float)):
            return result > 0

        if isinstance(result, str):
            return len(result) > 0

        return bool(result)

    def exists(self, path_expression: str, context: Dict[str, Any]) -> bool:
        """
        Check if path exists in context (regardless of value).

        Returns True if path exists, False otherwise.
        """
        try:
            jsonpath_expr = self._get_compiled_expression(path_expression)
            matches = jsonpath_expr.find(context)
            return len(matches) > 0
        except Exception:
            return False

    def _get_compiled_expression(self, path_expression: str):
        """Get or compile JSONPath expression with caching."""
        if path_expression in self._cache:
            return self._cache[path_expression]

        try:
            if self.use_extended:
                expr = jsonpath_ext_parse(path_expression)
            else:
                expr = jsonpath_parse(path_expression)

            self._cache[path_expression] = expr
            return expr

        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{path_expression}': {e}")


class ComparisonOperator(str, Enum):
    """Operators for comparing JSONPath evaluation results"""

    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # Regex match
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


class JSONPathCondition:
    """
    Represents a condition using JSONPath expression and comparison.

    This is more flexible than simple field comparisons:
    - Can query nested data
    - Can filter arrays
    - Can use extended operators
    """

    def __init__(
        self,
        path_expression: str,
        operator: ComparisonOperator,
        value: Any = None,
        evaluator: Optional[JSONPathEvaluator] = None,
    ):
        self.path_expression = path_expression
        self.operator = (
            ComparisonOperator(operator) if isinstance(operator, str) else operator
        )
        self.value = value
        self.evaluator = evaluator or JSONPathEvaluator()

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Returns:
            True if condition matches, False otherwise
        """
        try:
            # Get value from JSONPath
            actual_value = self.evaluator.evaluate(self.path_expression, context)

            # Apply operator
            return self._apply_operator(actual_value, self.value, self.operator)

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    def _apply_operator(
        self, actual: Any, expected: Any, operator: ComparisonOperator
    ) -> bool:
        """Apply comparison operator"""

        if operator == ComparisonOperator.EXISTS:
            return actual is not None

        if operator == ComparisonOperator.NOT_EXISTS:
            return actual is None

        if operator == ComparisonOperator.IS_EMPTY:
            return not actual or (
                isinstance(actual, (list, str, dict)) and len(actual) == 0
            )

        if operator == ComparisonOperator.IS_NOT_EMPTY:
            return bool(actual) and (
                not isinstance(actual, (list, str, dict)) or len(actual) > 0
            )

        # For other operators, actual must exist
        if actual is None:
            return False

        if operator == ComparisonOperator.EQUALS:
            return actual == expected

        if operator == ComparisonOperator.NOT_EQUALS:
            return actual != expected

        # Numeric comparisons
        if operator in [
            ComparisonOperator.GREATER_THAN,
            ComparisonOperator.LESS_THAN,
            ComparisonOperator.GREATER_EQUAL,
            ComparisonOperator.LESS_EQUAL,
        ]:
            try:
                actual_num = float(actual)
                expected_num = float(expected)

                if operator == ComparisonOperator.GREATER_THAN:
                    return actual_num > expected_num
                elif operator == ComparisonOperator.LESS_THAN:
                    return actual_num < expected_num
                elif operator == ComparisonOperator.GREATER_EQUAL:
                    return actual_num >= expected_num
                elif operator == ComparisonOperator.LESS_EQUAL:
                    return actual_num <= expected_num
            except (ValueError, TypeError):
                return False

        # Collection operators
        if operator == ComparisonOperator.IN:
            return (
                actual in expected
                if isinstance(expected, (list, tuple, set, str))
                else False
            )

        if operator == ComparisonOperator.NOT_IN:
            return (
                actual not in expected
                if isinstance(expected, (list, tuple, set, str))
                else False
            )

        if operator == ComparisonOperator.CONTAINS:
            if isinstance(actual, str):
                return expected in actual
            elif isinstance(actual, (list, tuple, set)):
                return expected in actual
            return False

        if operator == ComparisonOperator.NOT_CONTAINS:
            if isinstance(actual, str):
                return expected not in actual
            elif isinstance(actual, (list, tuple, set)):
                return expected not in actual
            return True

        # Regex match
        if operator == ComparisonOperator.MATCHES:
            try:
                return bool(re.match(expected, str(actual)))
            except Exception:
                return False

        return False


class JSONPathConditionGroup:
    """
    Group of JSONPath conditions with logical operators (AND/OR).

    Allows complex boolean logic:
        (condition1 AND condition2) OR condition3
    """

    def __init__(
        self,
        conditions: List[Union[JSONPathCondition, 'JSONPathConditionGroup']],
        combinator: str = "AND",
    ):
        self.conditions = conditions
        self.combinator = combinator.upper()

        if self.combinator not in ["AND", "OR"]:
            raise ValueError(f"Invalid combinator: {combinator}. Must be AND or OR")

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions in group"""
        results = []

        for condition in self.conditions:
            result = condition.evaluate(context)
            results.append(result)

            # Short-circuit evaluation
            if self.combinator == "AND" and not result:
                return False
            elif self.combinator == "OR" and result:
                return True

        return all(results) if self.combinator == "AND" else any(results)
