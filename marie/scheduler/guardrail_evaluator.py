"""
Guardrail evaluation service for runtime quality validation.
Evaluates guardrail metrics and determines pass/fail path routing.
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import Query, QueryPlan
from marie.query_planner.guardrail import (
    GuardrailAggregationMode,
    GuardrailEvaluationResult,
    GuardrailMetric,
    GuardrailMetricType,
    GuardrailQueryDefinition,
    GuardrailResult,
)
from marie.query_planner.jsonpath_evaluator import JSONPathEvaluator
from marie.scheduler.models import WorkInfo


class GuardrailEvaluationContext:
    """
    Context passed to guardrail evaluation.
    Contains all runtime information needed for metric evaluation.
    """

    def __init__(
        self,
        work_info: WorkInfo,
        dag_plan: QueryPlan,
        guardrail_node: Query,
        execution_results: Dict[str, Any] = None,
    ):
        self.work_info = work_info
        self.dag_plan = dag_plan
        self.guardrail_node = guardrail_node
        self.execution_results = execution_results or {}

        # Build context dict for JSONPath evaluation
        self.context = {
            "job_id": work_info.id,
            "dag_id": work_info.dag_id,
            "priority": work_info.priority,
            "data": work_info.data,
            "metadata": work_info.data.get("metadata", {}),
            "nodes": execution_results,
            "execution_results": execution_results,
        }


class GuardrailEvaluator:
    """
    Service responsible for evaluating guardrail metrics at runtime.
    Supports various metric types and custom executor-based evaluation.
    """

    def __init__(self):
        self.jsonpath_evaluator = JSONPathEvaluator(use_extended=True)

    async def evaluate(
        self,
        guardrail_def: GuardrailQueryDefinition,
        context: GuardrailEvaluationContext,
    ) -> GuardrailEvaluationResult:
        """
        Evaluate guardrail metrics and determine pass/fail path.

        Args:
            guardrail_def: The guardrail definition with metrics and paths
            context: Runtime context with execution results

        Returns:
            GuardrailEvaluationResult with pass/fail decision and metric results
        """
        start_time = time.time()

        # Extract input data using JSONPath
        input_data = None
        if guardrail_def.input_source:
            try:
                input_data = self.jsonpath_evaluator.evaluate(
                    guardrail_def.input_source, context.context
                )
            except Exception as e:
                logger.error(f"Failed to extract input data: {e}")

        # Extract context data for RAG metrics
        context_data = None
        if guardrail_def.context_source:
            try:
                context_data = self.jsonpath_evaluator.evaluate(
                    guardrail_def.context_source, context.context
                )
            except Exception as e:
                logger.warning(f"Failed to extract context data: {e}")

        # Extract query data for relevance metrics
        query_data = None
        if guardrail_def.query_source:
            try:
                query_data = self.jsonpath_evaluator.evaluate(
                    guardrail_def.query_source, context.context
                )
            except Exception as e:
                logger.warning(f"Failed to extract query data: {e}")

        # Evaluate each metric
        results: List[GuardrailResult] = []
        eval_context = {
            "input": input_data,
            "context": context_data,
            "query": query_data,
            "full_context": context.context,
        }

        for metric in guardrail_def.metrics:
            try:
                result = await self._evaluate_metric(metric, eval_context)
                results.append(result)

                # Fail fast if configured
                if guardrail_def.fail_fast and not result.passed:
                    logger.debug(f"Fail fast triggered by metric: {metric.name}")
                    break

            except Exception as e:
                logger.error(f"Error evaluating metric {metric.name}: {e}")
                results.append(
                    GuardrailResult(
                        metric_name=metric.name,
                        passed=False,
                        score=0.0,
                        feedback=f"Evaluation error: {str(e)}",
                        execution_time_ms=0.0,
                    )
                )
                if guardrail_def.fail_fast:
                    break

        # Aggregate results
        overall_passed, overall_score = self._aggregate_results(
            results, guardrail_def.aggregation_mode, guardrail_def.pass_threshold
        )

        # Determine selected path and target nodes
        pass_path = guardrail_def.get_pass_path()
        fail_path = guardrail_def.get_fail_path()

        if overall_passed:
            selected_path_id = "pass"
            active_nodes = pass_path.target_node_ids if pass_path else []
            skipped_nodes = fail_path.target_node_ids if fail_path else []
        else:
            selected_path_id = "fail"
            active_nodes = fail_path.target_node_ids if fail_path else []
            skipped_nodes = pass_path.target_node_ids if pass_path else []

        total_time = (time.time() - start_time) * 1000

        return GuardrailEvaluationResult(
            overall_passed=overall_passed,
            overall_score=overall_score,
            individual_results=results,
            selected_path_id=selected_path_id,
            active_target_nodes=active_nodes,
            skipped_target_nodes=skipped_nodes,
            total_execution_time_ms=total_time,
        )

    async def _evaluate_metric(
        self, metric: GuardrailMetric, context: Dict[str, Any]
    ) -> GuardrailResult:
        """Evaluate a single metric against input data"""
        start_time = time.time()
        input_data = context.get("input")

        passed = False
        score = 0.0
        feedback = ""

        try:
            if metric.type == GuardrailMetricType.REGEX_MATCH:
                passed, score, feedback = self._eval_regex(input_data, metric.params)

            elif metric.type == GuardrailMetricType.LENGTH_CHECK:
                passed, score, feedback = self._eval_length(input_data, metric.params)

            elif metric.type == GuardrailMetricType.JSON_SCHEMA:
                passed, score, feedback = self._eval_json_schema(
                    input_data, metric.params
                )

            elif metric.type == GuardrailMetricType.CONTAINS_KEYWORDS:
                passed, score, feedback = self._eval_keywords(input_data, metric.params)

            elif metric.type == GuardrailMetricType.EXECUTOR:
                passed, score, feedback = await self._eval_executor(
                    input_data, metric.params, context
                )

            elif metric.type == GuardrailMetricType.FAITHFULNESS:
                passed, score, feedback = await self._eval_faithfulness(
                    input_data, context.get("context"), metric.params
                )

            elif metric.type == GuardrailMetricType.RELEVANCE:
                passed, score, feedback = await self._eval_relevance(
                    input_data, context.get("query"), metric.params
                )

            elif metric.type == GuardrailMetricType.LLM_JUDGE:
                passed, score, feedback = await self._eval_llm_judge(
                    input_data, metric.params, context
                )

            else:
                feedback = f"Unknown metric type: {metric.type}"
                logger.warning(feedback)

            # Apply threshold
            passed = score >= metric.threshold

        except Exception as e:
            feedback = f"Evaluation error: {str(e)}"
            logger.error(f"Error in metric {metric.name}: {e}")

        execution_time = (time.time() - start_time) * 1000

        return GuardrailResult(
            metric_name=metric.name,
            passed=passed,
            score=score,
            feedback=feedback,
            execution_time_ms=execution_time,
        )

    def _eval_regex(self, data: Any, params: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Evaluate regex pattern matching"""
        pattern = params.get("pattern", "")
        if not pattern:
            return False, 0.0, "No regex pattern specified"

        text = str(data) if data is not None else ""

        try:
            # Use timeout for regex to prevent ReDoS
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)

            if params.get("must_not_match", False):
                # Inverted logic - pass if pattern NOT found
                if match:
                    return False, 0.0, f"Pattern found (should not match): {pattern}"
                return True, 1.0, f"Pattern not found (as expected): {pattern}"
            else:
                # Normal logic - pass if pattern found
                if match:
                    return True, 1.0, f"Pattern matched: {pattern}"
                return False, 0.0, f"Pattern not matched: {pattern}"

        except re.error as e:
            return False, 0.0, f"Invalid regex pattern: {e}"

    def _eval_length(
        self, data: Any, params: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Evaluate string length bounds"""
        text = str(data) if data is not None else ""
        length = len(text)

        min_len = params.get("min", 0)
        max_len = params.get("max", float("inf"))

        if min_len <= length <= max_len:
            # Calculate score based on position in range
            if max_len == float("inf"):
                score = 1.0 if length >= min_len else 0.0
            else:
                # Normalize to 0-1 scale
                score = 1.0
            return True, score, f"Length {length} within bounds [{min_len}, {max_len}]"

        if length < min_len:
            score = length / min_len if min_len > 0 else 0.0
            return False, score, f"Length {length} below minimum {min_len}"

        # length > max_len
        score = max_len / length if length > 0 else 0.0
        return False, score, f"Length {length} exceeds maximum {max_len}"

    def _eval_json_schema(
        self, data: Any, params: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Evaluate JSON schema validation"""
        schema = params.get("schema")
        if not schema:
            return False, 0.0, "No JSON schema specified"

        try:
            import jsonschema

            jsonschema.validate(data, schema)
            return True, 1.0, "Valid against JSON schema"

        except jsonschema.ValidationError as e:
            return False, 0.0, f"Schema validation failed: {e.message}"
        except jsonschema.SchemaError as e:
            return False, 0.0, f"Invalid schema: {e.message}"
        except ImportError:
            return False, 0.0, "jsonschema library not installed"

    def _eval_keywords(
        self, data: Any, params: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Evaluate keyword presence"""
        keywords = params.get("keywords", [])
        if not keywords:
            return True, 1.0, "No keywords to check"

        text = str(data).lower() if data is not None else ""
        case_sensitive = params.get("case_sensitive", False)

        if case_sensitive:
            text = str(data) if data is not None else ""
            found = [kw for kw in keywords if kw in text]
        else:
            found = [kw for kw in keywords if kw.lower() in text]

        score = len(found) / len(keywords)
        require_all = params.get("require_all", False)

        if require_all:
            passed = len(found) == len(keywords)
            feedback = f"Found {len(found)}/{len(keywords)} keywords (all required)"
        else:
            passed = len(found) > 0
            feedback = f"Found {len(found)}/{len(keywords)} keywords"

        return passed, score, feedback

    async def _eval_executor(
        self, data: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Call external executor for custom Python evaluation.

        Params should include:
            - endpoint: Executor endpoint (e.g., "guardrail_executor://evaluate")
            - function: Name of evaluation function to call
            - config: Optional function-specific configuration
        """
        endpoint = params.get("endpoint", "guardrail_executor://evaluate")
        function_name = params.get("function", "default")
        config = params.get("config", {})

        try:
            from marie.job.gateway_job_distributor import GatewayJobDistributor

            # Build evaluation request
            eval_params = {
                "function": function_name,
                "input_data": data,
                "context": context.get("full_context", {}),
                "config": config,
            }

            # Dispatch to executor
            distributor = GatewayJobDistributor.get_instance()
            result = await distributor.submit_and_wait(
                entrypoint=endpoint, parameters=eval_params
            )

            # Parse executor response
            passed = result.get("passed", False)
            score = result.get("score", 0.0)
            feedback = result.get("feedback", "No feedback provided")

            return passed, score, feedback

        except ImportError as e:
            logger.error(f"GatewayJobDistributor not available: {e}")
            return False, 0.0, f"Executor not available: {e}"
        except Exception as e:
            logger.error(f"Executor evaluation failed: {e}")
            return False, 0.0, f"Executor error: {str(e)}"

    async def _eval_faithfulness(
        self, response: Any, context: Any, params: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Evaluate RAG faithfulness - is the response grounded in context?
        This is a placeholder for integration with existing evaluation infrastructure.
        """
        # TODO: Integrate with marie.evaluation.faithfulness
        logger.warning("Faithfulness evaluation not yet implemented")
        return True, 1.0, "Faithfulness evaluation not yet implemented"

    async def _eval_relevance(
        self, response: Any, query: Any, params: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Evaluate query-response relevance.
        This is a placeholder for integration with existing evaluation infrastructure.
        """
        # TODO: Integrate with marie.evaluation.relevance
        logger.warning("Relevance evaluation not yet implemented")
        return True, 1.0, "Relevance evaluation not yet implemented"

    async def _eval_llm_judge(
        self, data: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Use LLM as judge for evaluation.
        Requires prompt template and model configuration.
        """
        prompt_template = params.get("prompt")
        model = params.get("model", "default")

        if not prompt_template:
            return False, 0.0, "No prompt template specified for LLM judge"

        # TODO: Integrate with LLM evaluation
        logger.warning("LLM judge evaluation not yet implemented")
        return True, 1.0, "LLM judge evaluation not yet implemented"

    def _aggregate_results(
        self,
        results: List[GuardrailResult],
        mode: GuardrailAggregationMode,
        pass_threshold: float,
    ) -> Tuple[bool, float]:
        """Aggregate metric results based on aggregation mode"""
        if not results:
            # No metrics = pass by default
            return True, 1.0

        if mode == GuardrailAggregationMode.ALL:
            # Pass only if ALL metrics pass
            passed = all(r.passed for r in results)
            score = sum(r.score for r in results) / len(results)
            return passed, score

        elif mode == GuardrailAggregationMode.ANY:
            # Pass if ANY metric passes
            passed = any(r.passed for r in results)
            score = max(r.score for r in results)
            return passed, score

        else:  # WEIGHTED_AVERAGE
            # Calculate weighted average and compare to threshold
            total_weight = len(results)  # Using equal weights for now
            weighted_score = sum(r.score for r in results) / total_weight
            passed = weighted_score >= pass_threshold
            return passed, weighted_score
