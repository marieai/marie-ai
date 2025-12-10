"""
Branch evaluation service for runtime condition evaluation.
Evaluates branch conditions and determines which paths to activate.
"""

import asyncio
import importlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import Query, QueryPlan
from marie.query_planner.branching import (
    BranchCondition,
    BranchConditionGroup,
    BranchEvaluationMode,
    BranchPath,
    BranchQueryDefinition,
    PythonBranchQueryDefinition,
    SwitchQueryDefinition,
)
from marie.query_planner.jsonpath_evaluator import (
    ComparisonOperator,
    JSONPathCondition,
    JSONPathConditionGroup,
    JSONPathEvaluator,
)
from marie.scheduler.models import WorkInfo


class BranchEvaluationContext:
    """
    Context passed to branch evaluation functions.
    Contains all runtime information needed for decision making.
    """

    def __init__(
        self,
        work_info: WorkInfo,
        dag_plan: QueryPlan,
        branch_node: Query,
        execution_results: Dict[str, Any] = None,
    ):
        self.work_info = work_info
        self.dag_plan = dag_plan
        self.branch_node = branch_node
        self.execution_results = execution_results or {}

        # Build convenient context dict
        self.context = {
            "job_id": work_info.id,
            "dag_id": work_info.dag_id,
            "priority": work_info.priority,
            "data": work_info.data,
            "metadata": work_info.data.get("metadata", {}),
            "execution_results": execution_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_field(self, field_path: str) -> Any:
        """Get field value using JSONPath-like syntax"""
        if field_path.startswith("$."):
            field_path = field_path[2:]

        parts = field_path.split(".")
        value = self.context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value


class BranchEvaluator:
    """
    Service responsible for evaluating branch conditions at runtime.
    Supports both JSONPath conditions and arbitrary Python functions.
    """

    def __init__(self):
        self.jsonpath_evaluator = JSONPathEvaluator(use_extended=True)
        self._function_cache: Dict[str, Any] = {}

    async def evaluate_branch(
        self, branch_def: BranchQueryDefinition, context: BranchEvaluationContext
    ) -> List[str]:
        """
        Evaluate branch and return list of path_ids to activate.

        Returns:
            List of path_ids to activate (empty if no match and no default)
        """
        if isinstance(branch_def, PythonBranchQueryDefinition):
            return await self._evaluate_python_branch(branch_def, context)

        # Standard branch with per-path conditions
        matching_paths = []
        sorted_paths = sorted(branch_def.paths, key=lambda p: p.priority, reverse=True)

        for path in sorted_paths:
            try:
                if await self._evaluate_path_condition(path, context):
                    matching_paths.append(path.path_id)

                    if branch_def.evaluation_mode == BranchEvaluationMode.FIRST_MATCH:
                        break
                    elif (
                        branch_def.evaluation_mode
                        == BranchEvaluationMode.PRIORITY_MATCH
                    ):
                        break  # Already sorted by priority
            except Exception as e:
                logger.error(f"Error evaluating path {path.path_id}: {e}")
                continue

        # Fallback to default
        if not matching_paths and branch_def.default_path_id:
            matching_paths.append(branch_def.default_path_id)

        return matching_paths

    async def _evaluate_python_branch(
        self, branch_def: PythonBranchQueryDefinition, context: BranchEvaluationContext
    ) -> List[str]:
        """Execute Python function to determine paths"""
        func = self._load_function(branch_def.branch_function)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, func, context.context),
                timeout=branch_def.function_timeout,
            )

            # Normalize result to list
            if isinstance(result, str):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                logger.error(f"Branch function returned invalid type: {type(result)}")
                return []

        except asyncio.TimeoutError:
            logger.error(
                f"Branch function timed out after {branch_def.function_timeout}s"
            )
            return []
        except Exception as e:
            logger.error(f"Branch function failed: {e}")
            return []

    async def _evaluate_path_condition(
        self, path: BranchPath, context: BranchEvaluationContext
    ) -> bool:
        """Evaluate a single path's condition"""

        # Python function condition
        if path.condition_function:
            func = self._load_function(path.condition_function)
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, func, context.context
                )
                return bool(result)
            except Exception as e:
                logger.error(f"Condition function failed for path {path.path_id}: {e}")
                return False

        # JSONPath condition
        elif path.condition:
            return self._evaluate_jsonpath_condition(path.condition, context.context)

        # No condition = always True (default path)
        return True

    def _evaluate_jsonpath_condition(
        self,
        condition: Union[BranchCondition, BranchConditionGroup],
        context: Dict[str, Any],
    ) -> bool:
        """Evaluate JSONPath condition or condition group"""

        if isinstance(condition, BranchConditionGroup):
            # Evaluate group
            results = []
            for cond in condition.conditions:
                result = self._evaluate_jsonpath_condition(cond, context)
                results.append(result)

                # Short-circuit
                if condition.combinator == "AND" and not result:
                    return False
                elif condition.combinator == "OR" and result:
                    return True

            return all(results) if condition.combinator == "AND" else any(results)

        else:
            # Single condition - convert to JSONPathCondition
            jsonpath_cond = JSONPathCondition(
                path_expression=condition.jsonpath,
                operator=condition.operator,
                value=condition.value,
                evaluator=self.jsonpath_evaluator,
            )
            return jsonpath_cond.evaluate(context)

    def _load_function(self, function_path: str) -> Any:
        """Load and cache Python function by module path"""
        if function_path in self._function_cache:
            return self._function_cache[function_path]

        try:
            module_path, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)

            self._function_cache[function_path] = func
            return func

        except Exception as e:
            raise ImportError(f"Failed to load function {function_path}: {e}")

    async def evaluate_switch(
        self, switch_def: SwitchQueryDefinition, context: BranchEvaluationContext
    ) -> Optional[List[str]]:
        """Evaluate SWITCH node - simpler than BRANCH"""
        field_value = self.jsonpath_evaluator.evaluate(
            switch_def.switch_field, context.context
        )

        if field_value in switch_def.cases:
            return switch_def.cases[field_value]

        return switch_def.default_case
