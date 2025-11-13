"""
Unit tests for JSONPath evaluator functionality.

Tests cover:
- Basic JSONPath evaluation
- All comparison operators
- Condition evaluation (single and groups)
- Edge cases and error handling
"""

import pytest

from marie.query_planner.jsonpath_evaluator import (
    ComparisonOperator,
    JSONPathCondition,
    JSONPathConditionGroup,
    JSONPathEvaluator,
)


class TestJSONPathEvaluator:
    """Tests for JSONPathEvaluator class."""

    def test_evaluate_simple_path(self):
        """Test basic JSONPath evaluation."""
        evaluator = JSONPathEvaluator()
        context = {"name": "John", "age": 30}

        result = evaluator.evaluate("$.name", context)
        assert result == "John"

        result = evaluator.evaluate("$.age", context)
        assert result == 30

    def test_evaluate_nested_path(self):
        """Test nested JSONPath evaluation."""
        evaluator = JSONPathEvaluator()
        context = {
            "metadata": {
                "document_type": "invoice",
                "priority": "high",
            }
        }

        result = evaluator.evaluate("$.metadata.document_type", context)
        assert result == "invoice"

        result = evaluator.evaluate("$.metadata.priority", context)
        assert result == "high"

    def test_evaluate_array_path(self):
        """Test JSONPath with arrays."""
        evaluator = JSONPathEvaluator()
        context = {
            "items": [
                {"name": "item1", "price": 10},
                {"name": "item2", "price": 20},
            ]
        }

        result = evaluator.evaluate("$.items[0].name", context)
        assert result == "item1"

        result = evaluator.evaluate("$.items[1].price", context)
        assert result == 20

    def test_evaluate_with_default(self):
        """Test default value when path not found."""
        evaluator = JSONPathEvaluator()
        context = {"name": "John"}

        result = evaluator.evaluate("$.missing_field", context, default="default_value")
        assert result == "default_value"

    def test_evaluate_nonexistent_path(self):
        """Test nonexistent path returns None."""
        evaluator = JSONPathEvaluator()
        context = {"name": "John"}

        result = evaluator.evaluate("$.nonexistent", context)
        assert result is None

    def test_expression_caching(self):
        """Test that expressions are cached for performance."""
        evaluator = JSONPathEvaluator()
        context = {"value": 42}

        # First evaluation - should compile and cache
        result1 = evaluator.evaluate("$.value", context)
        assert result1 == 42
        assert "$.value" in evaluator._cache

        # Second evaluation - should use cache
        result2 = evaluator.evaluate("$.value", context)
        assert result2 == 42
        assert len(evaluator._cache) == 1

    def test_extended_operators(self):
        """Test extended JSONPath operators (jsonpath-ng-ext)."""
        evaluator = JSONPathEvaluator(use_extended=True)
        context = {
            "items": [
                {"name": "apple", "price": 1.0},
                {"name": "banana", "price": 0.5},
                {"name": "cherry", "price": 2.0},
            ]
        }

        # Filter items where price > 1.0 (extended syntax)
        result = evaluator.evaluate("$.items[?(@.price > 1.0)].name", context)
        assert result is not None  # Should return something


class TestComparisonOperator:
    """Tests for ComparisonOperator enum."""

    def test_all_operators_defined(self):
        """Test that all expected operators are defined."""
        expected_operators = [
            "EQUALS", "NOT_EQUALS", "GREATER_THAN", "LESS_THAN",
            "GREATER_THAN_OR_EQUAL", "LESS_THAN_OR_EQUAL",
            "IN", "NOT_IN", "CONTAINS", "NOT_CONTAINS",
            "MATCHES", "NOT_MATCHES", "EXISTS", "IS_EMPTY"
        ]

        for op in expected_operators:
            assert hasattr(ComparisonOperator, op)

    def test_operator_values(self):
        """Test operator string values."""
        assert ComparisonOperator.EQUALS == "=="
        assert ComparisonOperator.NOT_EQUALS == "!="
        assert ComparisonOperator.GREATER_THAN == ">"
        assert ComparisonOperator.LESS_THAN == "<"


class TestJSONPathCondition:
    """Tests for JSONPathCondition class."""

    def test_equals_operator(self):
        """Test EQUALS operator."""
        condition = JSONPathCondition(
            jsonpath="$.status",
            operator="==",
            value="active"
        )

        # Should match
        context = {"status": "active"}
        assert condition.evaluate(context) is True

        # Should not match
        context = {"status": "inactive"}
        assert condition.evaluate(context) is False

    def test_not_equals_operator(self):
        """Test NOT_EQUALS operator."""
        condition = JSONPathCondition(
            jsonpath="$.status",
            operator="!=",
            value="inactive"
        )

        context = {"status": "active"}
        assert condition.evaluate(context) is True

        context = {"status": "inactive"}
        assert condition.evaluate(context) is False

    def test_greater_than_operator(self):
        """Test GREATER_THAN operator."""
        condition = JSONPathCondition(
            jsonpath="$.count",
            operator=">",
            value=10
        )

        context = {"count": 20}
        assert condition.evaluate(context) is True

        context = {"count": 5}
        assert condition.evaluate(context) is False

        context = {"count": 10}
        assert condition.evaluate(context) is False

    def test_less_than_operator(self):
        """Test LESS_THAN operator."""
        condition = JSONPathCondition(
            jsonpath="$.count",
            operator="<",
            value=10
        )

        context = {"count": 5}
        assert condition.evaluate(context) is True

        context = {"count": 20}
        assert condition.evaluate(context) is False

    def test_greater_than_or_equal_operator(self):
        """Test GREATER_THAN_OR_EQUAL operator."""
        condition = JSONPathCondition(
            jsonpath="$.count",
            operator=">=",
            value=10
        )

        context = {"count": 10}
        assert condition.evaluate(context) is True

        context = {"count": 20}
        assert condition.evaluate(context) is True

        context = {"count": 5}
        assert condition.evaluate(context) is False

    def test_less_than_or_equal_operator(self):
        """Test LESS_THAN_OR_EQUAL operator."""
        condition = JSONPathCondition(
            jsonpath="$.count",
            operator="<=",
            value=10
        )

        context = {"count": 10}
        assert condition.evaluate(context) is True

        context = {"count": 5}
        assert condition.evaluate(context) is True

        context = {"count": 20}
        assert condition.evaluate(context) is False

    def test_in_operator(self):
        """Test IN operator."""
        condition = JSONPathCondition(
            jsonpath="$.status",
            operator="in",
            value=["active", "pending", "processing"]
        )

        context = {"status": "active"}
        assert condition.evaluate(context) is True

        context = {"status": "inactive"}
        assert condition.evaluate(context) is False

    def test_not_in_operator(self):
        """Test NOT_IN operator."""
        condition = JSONPathCondition(
            jsonpath="$.status",
            operator="not_in",
            value=["inactive", "deleted"]
        )

        context = {"status": "active"}
        assert condition.evaluate(context) is True

        context = {"status": "deleted"}
        assert condition.evaluate(context) is False

    def test_contains_operator(self):
        """Test CONTAINS operator."""
        condition = JSONPathCondition(
            jsonpath="$.tags",
            operator="contains",
            value="urgent"
        )

        context = {"tags": ["urgent", "important", "high-priority"]}
        assert condition.evaluate(context) is True

        context = {"tags": ["normal", "standard"]}
        assert condition.evaluate(context) is False

    def test_matches_operator(self):
        """Test MATCHES operator (regex)."""
        condition = JSONPathCondition(
            jsonpath="$.email",
            operator="matches",
            value=r".*@example\.com$"
        )

        context = {"email": "user@example.com"}
        assert condition.evaluate(context) is True

        context = {"email": "user@other.com"}
        assert condition.evaluate(context) is False

    def test_exists_operator(self):
        """Test EXISTS operator."""
        condition = JSONPathCondition(
            jsonpath="$.optional_field",
            operator="exists",
            value=True
        )

        context = {"optional_field": "value"}
        assert condition.evaluate(context) is True

        context = {"other_field": "value"}
        assert condition.evaluate(context) is False

    def test_is_empty_operator(self):
        """Test IS_EMPTY operator."""
        condition = JSONPathCondition(
            jsonpath="$.items",
            operator="is_empty",
            value=True
        )

        context = {"items": []}
        assert condition.evaluate(context) is True

        context = {"items": [1, 2, 3]}
        assert condition.evaluate(context) is False

    def test_nested_path_evaluation(self):
        """Test condition with nested paths."""
        condition = JSONPathCondition(
            jsonpath="$.metadata.document_type",
            operator="==",
            value="invoice"
        )

        context = {
            "metadata": {
                "document_type": "invoice",
                "priority": "high"
            }
        }
        assert condition.evaluate(context) is True

    def test_invalid_operator(self):
        """Test that invalid operator raises error."""
        with pytest.raises(ValueError):
            condition = JSONPathCondition(
                jsonpath="$.field",
                operator="invalid_op",
                value="test"
            )
            condition.evaluate({"field": "test"})


class TestJSONPathConditionGroup:
    """Tests for JSONPathConditionGroup class."""

    def test_and_combinator_all_true(self):
        """Test AND combinator with all conditions true."""
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.type", operator="==", value="invoice"),
                JSONPathCondition(jsonpath="$.amount", operator=">", value=100),
            ],
            combinator="AND"
        )

        context = {"type": "invoice", "amount": 150}
        assert group.evaluate(context) is True

    def test_and_combinator_one_false(self):
        """Test AND combinator with one condition false."""
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.type", operator="==", value="invoice"),
                JSONPathCondition(jsonpath="$.amount", operator=">", value=100),
            ],
            combinator="AND"
        )

        context = {"type": "invoice", "amount": 50}
        assert group.evaluate(context) is False

    def test_or_combinator_one_true(self):
        """Test OR combinator with one condition true."""
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.type", operator="==", value="invoice"),
                JSONPathCondition(jsonpath="$.type", operator="==", value="contract"),
            ],
            combinator="OR"
        )

        context = {"type": "invoice"}
        assert group.evaluate(context) is True

        context = {"type": "contract"}
        assert group.evaluate(context) is True

    def test_or_combinator_all_false(self):
        """Test OR combinator with all conditions false."""
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.type", operator="==", value="invoice"),
                JSONPathCondition(jsonpath="$.type", operator="==", value="contract"),
            ],
            combinator="OR"
        )

        context = {"type": "other"}
        assert group.evaluate(context) is False

    def test_nested_condition_groups(self):
        """Test nested condition groups (AND of ORs)."""
        # (type == invoice OR type == contract) AND amount > 100
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathConditionGroup(
                    conditions=[
                        JSONPathCondition(jsonpath="$.type", operator="==", value="invoice"),
                        JSONPathCondition(jsonpath="$.type", operator="==", value="contract"),
                    ],
                    combinator="OR"
                ),
                JSONPathCondition(jsonpath="$.amount", operator=">", value=100),
            ],
            combinator="AND"
        )

        # Invoice with amount > 100 -> True
        context = {"type": "invoice", "amount": 150}
        assert group.evaluate(context) is True

        # Contract with amount > 100 -> True
        context = {"type": "contract", "amount": 200}
        assert group.evaluate(context) is True

        # Invoice with amount <= 100 -> False
        context = {"type": "invoice", "amount": 50}
        assert group.evaluate(context) is False

        # Other type with amount > 100 -> False
        context = {"type": "other", "amount": 150}
        assert group.evaluate(context) is False

    def test_short_circuit_and(self):
        """Test that AND short-circuits on first false."""
        # This is implicitly tested by the implementation
        # If short-circuit works, later conditions won't be evaluated
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.first", operator="==", value="fail"),
                JSONPathCondition(jsonpath="$.nonexistent.path", operator="==", value="anything"),
            ],
            combinator="AND"
        )

        context = {"first": "wrong"}
        # Should not raise error even though second path is invalid
        assert group.evaluate(context) is False

    def test_short_circuit_or(self):
        """Test that OR short-circuits on first true."""
        group = JSONPathConditionGroup(
            conditions=[
                JSONPathCondition(jsonpath="$.first", operator="==", value="match"),
                JSONPathCondition(jsonpath="$.nonexistent.path", operator="==", value="anything"),
            ],
            combinator="OR"
        )

        context = {"first": "match"}
        # Should not raise error even though second path is invalid
        assert group.evaluate(context) is True

    def test_empty_condition_group(self):
        """Test empty condition group returns True."""
        group = JSONPathConditionGroup(
            conditions=[],
            combinator="AND"
        )

        context = {"any": "value"}
        assert group.evaluate(context) is True


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_null_values(self):
        """Test handling of null values."""
        condition = JSONPathCondition(
            jsonpath="$.value",
            operator="==",
            value=None
        )

        context = {"value": None}
        assert condition.evaluate(context) is True

        context = {"value": "something"}
        assert condition.evaluate(context) is False

    def test_boolean_values(self):
        """Test handling of boolean values."""
        condition = JSONPathCondition(
            jsonpath="$.is_active",
            operator="==",
            value=True
        )

        context = {"is_active": True}
        assert condition.evaluate(context) is True

        context = {"is_active": False}
        assert condition.evaluate(context) is False

    def test_numeric_comparison_types(self):
        """Test numeric comparisons with different types."""
        condition = JSONPathCondition(
            jsonpath="$.value",
            operator=">",
            value=10
        )

        # Int vs int
        context = {"value": 15}
        assert condition.evaluate(context) is True

        # Float vs int
        context = {"value": 15.5}
        assert condition.evaluate(context) is True

        # Int vs float comparison value
        condition2 = JSONPathCondition(
            jsonpath="$.value",
            operator=">",
            value=10.5
        )
        context = {"value": 11}
        assert condition2.evaluate(context) is True

    def test_string_comparison(self):
        """Test string comparisons."""
        condition = JSONPathCondition(
            jsonpath="$.name",
            operator=">",
            value="Alice"
        )

        context = {"name": "Bob"}
        assert condition.evaluate(context) is True

        context = {"name": "Aaron"}
        assert condition.evaluate(context) is False

    def test_case_sensitivity(self):
        """Test that string comparisons are case-sensitive."""
        condition = JSONPathCondition(
            jsonpath="$.status",
            operator="==",
            value="Active"
        )

        context = {"status": "Active"}
        assert condition.evaluate(context) is True

        context = {"status": "active"}
        assert condition.evaluate(context) is False

    def test_special_characters_in_path(self):
        """Test JSONPath with special characters in keys."""
        evaluator = JSONPathEvaluator()
        context = {
            "data": {
                "field-with-dash": "value1",
                "field_with_underscore": "value2"
            }
        }

        # JSONPath uses bracket notation for keys with special characters
        result = evaluator.evaluate("$.data['field-with-dash']", context)
        assert result == "value1"

        result = evaluator.evaluate("$.data.field_with_underscore", context)
        assert result == "value2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
