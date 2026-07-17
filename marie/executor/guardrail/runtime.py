import inspect
import json
import re
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Tuple

import jsonschema

from marie.query_planner.guardrail import (
    GuardrailAggregationMode,
    GuardrailEvaluationReport,
    GuardrailExecutionSpec,
    GuardrailMetric,
    GuardrailMetricType,
    GuardrailResult,
)

EvaluationFunction = Callable[
    [Any, Dict[str, Any], Dict[str, Any]],
    Dict[str, Any] | Awaitable[Dict[str, Any]],
]


class GuardrailRuntime:
    def __init__(self, evaluation_functions: Mapping[str, EvaluationFunction]):
        self._evaluation_functions = evaluation_functions

    async def evaluate(
        self,
        spec: GuardrailExecutionSpec,
        input_data: Any,
        *,
        context_data: Any = None,
        query_data: Any = None,
        full_context: Dict[str, Any] | None = None,
    ) -> GuardrailEvaluationReport:
        started_at = time.perf_counter()
        results: list[GuardrailResult] = []
        metric_context = {
            "input": input_data,
            "context": context_data,
            "query": query_data,
            "full_context": full_context or {},
        }

        for metric in spec.metrics:
            result = await self._evaluate_metric(metric, metric_context)
            if not spec.include_feedback:
                result.feedback = None
            results.append(result)
            if spec.fail_fast and not result.passed:
                break

        overall_passed, overall_score = self._aggregate_results(
            results,
            spec.metrics[: len(results)],
            spec.aggregation_mode,
            spec.pass_threshold,
        )
        return GuardrailEvaluationReport(
            overall_passed=overall_passed,
            overall_score=overall_score,
            individual_results=results,
            outcome="VALID" if overall_passed else "INVALID",
            total_execution_time_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def _evaluate_metric(
        self, metric: GuardrailMetric, context: Dict[str, Any]
    ) -> GuardrailResult:
        started_at = time.perf_counter()
        input_data = context["input"]

        if metric.type == GuardrailMetricType.REGEX_MATCH:
            score, feedback = self._eval_regex(input_data, metric.params)
        elif metric.type == GuardrailMetricType.LENGTH_CHECK:
            score, feedback = self._eval_length(input_data, metric.params)
        elif metric.type == GuardrailMetricType.JSON_SCHEMA:
            score, feedback = self._eval_json_schema(input_data, metric.params)
        elif metric.type == GuardrailMetricType.CONTAINS_KEYWORDS:
            score, feedback = self._eval_keywords(input_data, metric.params)
        elif metric.type == GuardrailMetricType.EXECUTOR:
            score, feedback = await self._eval_function(
                input_data, metric.params, context
            )
        else:
            raise NotImplementedError(
                f"Guardrail metric type '{metric.type.value}' is not implemented"
            )

        return GuardrailResult(
            metric_name=metric.name,
            passed=score >= metric.threshold,
            score=score,
            feedback=feedback,
            execution_time_ms=(time.perf_counter() - started_at) * 1000,
        )

    @staticmethod
    def _eval_regex(data: Any, params: Dict[str, Any]) -> Tuple[float, str]:
        pattern = params.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("regex_match requires params.pattern")

        try:
            matched = re.search(
                pattern,
                "" if data is None else str(data),
                re.MULTILINE | re.DOTALL,
            )
        except re.error as exc:
            raise ValueError(f"Invalid guardrail regex: {exc}") from exc

        must_not_match = bool(params.get("must_not_match", False))
        passed = matched is None if must_not_match else matched is not None
        if must_not_match:
            feedback = "Pattern was absent" if passed else "Forbidden pattern matched"
        else:
            feedback = "Pattern matched" if passed else "Pattern did not match"
        return (1.0 if passed else 0.0), feedback

    @staticmethod
    def _eval_length(data: Any, params: Dict[str, Any]) -> Tuple[float, str]:
        length = len("" if data is None else str(data))
        minimum = params.get("min", 0)
        maximum = params.get("max")
        if not isinstance(minimum, int) or minimum < 0:
            raise ValueError("length_check params.min must be a non-negative integer")
        if maximum is not None and (not isinstance(maximum, int) or maximum < minimum):
            raise ValueError("length_check params.max must be an integer >= min")

        if length < minimum:
            score = length / minimum if minimum else 0.0
            return score, f"Length {length} is below minimum {minimum}"
        if maximum is not None and length > maximum:
            score = maximum / length if length else 0.0
            return score, f"Length {length} exceeds maximum {maximum}"
        return 1.0, f"Length {length} is within configured bounds"

    @staticmethod
    def _eval_json_schema(data: Any, params: Dict[str, Any]) -> Tuple[float, str]:
        schema = params.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema requires params.schema")

        candidate = data
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                return 0.0, "Input is not valid JSON"

        try:
            jsonschema.validate(candidate, schema)
        except jsonschema.ValidationError as exc:
            return 0.0, f"Schema validation failed: {exc.message}"
        except jsonschema.SchemaError as exc:
            raise ValueError(f"Invalid guardrail JSON schema: {exc.message}") from exc
        return 1.0, "Input matches the JSON schema"

    @staticmethod
    def _eval_keywords(data: Any, params: Dict[str, Any]) -> Tuple[float, str]:
        keywords = params.get("keywords")
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) and keyword for keyword in keywords
        ):
            raise ValueError("contains_keywords requires a non-empty keyword list")

        case_sensitive = bool(params.get("case_sensitive", False))
        text = "" if data is None else str(data)
        haystack = text if case_sensitive else text.lower()
        needles = keywords if case_sensitive else [word.lower() for word in keywords]
        found_count = sum(1 for word in needles if word in haystack)
        score = found_count / len(keywords)
        return score, f"Found {found_count}/{len(keywords)} configured keywords"

    async def _eval_function(
        self,
        data: Any,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        function_name = params.get("function")
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("executor metric requires params.function")

        function = self._evaluation_functions.get(function_name)
        if function is None:
            raise ValueError(f"Unknown guardrail evaluation function: {function_name}")

        result = function(
            data,
            context.get("full_context", {}),
            params.get("config", {}),
        )
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError(
                f"Guardrail evaluation function '{function_name}' must return a dict"
            )

        score = result.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(
                f"Guardrail evaluation function '{function_name}' returned an invalid score"
            )
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Guardrail evaluation function '{function_name}' score must be between 0 and 1"
            )
        return score, str(result.get("feedback", ""))

    @staticmethod
    def _aggregate_results(
        results: list[GuardrailResult],
        metrics: list[GuardrailMetric],
        mode: GuardrailAggregationMode,
        pass_threshold: float,
    ) -> Tuple[bool, float]:
        if not results:
            return True, 1.0
        if mode == GuardrailAggregationMode.ALL:
            return all(result.passed for result in results), sum(
                result.score for result in results
            ) / len(results)
        if mode == GuardrailAggregationMode.ANY:
            return any(result.passed for result in results), max(
                result.score for result in results
            )

        total_weight = sum(metric.weight for metric in metrics)
        if total_weight <= 0:
            raise ValueError("weighted_average requires at least one positive weight")
        score = (
            sum(
                result.score * metric.weight
                for result, metric in zip(results, metrics, strict=True)
            )
            / total_weight
        )
        return score >= pass_threshold, score
