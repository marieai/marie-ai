"""
Integration tests for GUARDRAIL node execution in the scheduler.

Tests the full flow of GUARDRAIL nodes including:
- Query plan generation with guardrail nodes
- Path selection and skip cascade
- Integration with the scheduler pipeline
"""

import pytest

from marie.job.job_manager import generate_job_id
from marie.query_planner.base import PlannerInfo, QueryPlanRegistry
from marie.query_planner.guardrail import (
    GuardrailMetric,
    GuardrailMetricType,
    GuardrailPath,
    GuardrailQueryDefinition,
)

# Import mock plans to register them
from tests.integration.scheduler.mock_plans.guardrail import (
    query_planner_mock_guardrail_executor_metric,
    query_planner_mock_guardrail_multi_metric,
    query_planner_mock_guardrail_retry_loop,
    query_planner_mock_guardrail_simple,
)


class TestGuardrailQueryPlanGeneration:
    """Tests for GUARDRAIL query plan generation."""

    def test_simple_guardrail_plan_structure(self):
        """Test that simple guardrail plan has correct structure."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        assert plan is not None
        assert len(plan.nodes) == 5  # START, PROCESS, GUARDRAIL, SUCCESS, ERROR

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        assert len(guardrail_nodes) == 1

        guardrail = guardrail_nodes[0]
        assert guardrail.definition.method == "GUARDRAIL"
        assert len(guardrail.definition.metrics) == 2
        assert len(guardrail.definition.paths) == 2

    def test_retry_loop_plan_structure(self):
        """Test that retry loop plan has correct structure."""
        planner_info = PlannerInfo(
            name="mock_guardrail_retry_loop",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_retry_loop(planner_info)

        assert plan is not None
        # START, LLM_1, GUARDRAIL_1, END_SUCCESS, LLM_2, GUARDRAIL_2, END_RETRY
        assert len(plan.nodes) == 7

        # Should have 2 guardrail nodes
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        assert len(guardrail_nodes) == 2

    def test_executor_metric_plan_structure(self):
        """Test that executor metric plan has correct structure."""
        planner_info = PlannerInfo(
            name="mock_guardrail_executor_metric",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_executor_metric(planner_info)

        assert plan is not None
        # START, EXTRACT, GUARDRAIL, STORE, REVIEW
        assert len(plan.nodes) == 5

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Should have executor metric
        executor_metrics = [m for m in guardrail.definition.metrics
                           if m.type == GuardrailMetricType.EXECUTOR]
        assert len(executor_metrics) == 1
        assert "endpoint" in executor_metrics[0].params

    def test_multi_metric_plan_structure(self):
        """Test that multi-metric plan has correct structure."""
        planner_info = PlannerInfo(
            name="mock_guardrail_multi_metric",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_multi_metric(planner_info)

        assert plan is not None
        # START, PROCESS, GUARDRAIL, END, ERROR
        assert len(plan.nodes) == 5

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Should have 4 metrics with weighted_average aggregation
        assert len(guardrail.definition.metrics) == 4
        assert guardrail.definition.aggregation_mode == "weighted_average"
        assert guardrail.definition.fail_fast is True


class TestGuardrailPlanRegistry:
    """Tests for GUARDRAIL plan registration."""

    def test_guardrail_plans_registered(self):
        """Test that all guardrail plans are registered in the registry."""
        plan_names = [
            "mock_guardrail_simple",
            "mock_guardrail_retry_loop",
            "mock_guardrail_executor_metric",
            "mock_guardrail_multi_metric",
        ]

        for name in plan_names:
            planner = QueryPlanRegistry.get(name)
            assert planner is not None, f"Plan {name} not registered"

    def test_generate_plan_from_registry(self):
        """Test generating a plan from the registry."""
        planner_func = QueryPlanRegistry.get("mock_guardrail_simple")
        assert planner_func is not None

        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = planner_func(planner_info)

        assert plan is not None
        assert len(plan.nodes) > 0


class TestGuardrailPathConfiguration:
    """Tests for GUARDRAIL path configuration."""

    def test_pass_path_has_correct_targets(self):
        """Test that pass path has correct target node IDs."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Get pass path
        pass_path = next((p for p in guardrail.definition.paths if p.path_id == "pass"), None)
        assert pass_path is not None
        assert len(pass_path.target_node_ids) > 0

        # Verify target nodes exist in plan
        all_node_ids = {n.task_id for n in plan.nodes}
        for target_id in pass_path.target_node_ids:
            assert target_id in all_node_ids, f"Pass path target {target_id} not in plan"

    def test_fail_path_has_correct_targets(self):
        """Test that fail path has correct target node IDs."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Get fail path
        fail_path = next((p for p in guardrail.definition.paths if p.path_id == "fail"), None)
        assert fail_path is not None
        assert len(fail_path.target_node_ids) > 0

        # Verify target nodes exist in plan
        all_node_ids = {n.task_id for n in plan.nodes}
        for target_id in fail_path.target_node_ids:
            assert target_id in all_node_ids, f"Fail path target {target_id} not in plan"

    def test_paths_are_mutually_exclusive(self):
        """Test that pass and fail paths have different targets."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        # Find the guardrail node
        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        pass_path = next((p for p in guardrail.definition.paths if p.path_id == "pass"), None)
        fail_path = next((p for p in guardrail.definition.paths if p.path_id == "fail"), None)

        pass_targets = set(pass_path.target_node_ids)
        fail_targets = set(fail_path.target_node_ids)

        # For simple guardrail, pass and fail should be different nodes
        assert pass_targets.isdisjoint(fail_targets), "Pass and fail paths should not share targets"


class TestGuardrailMetricConfiguration:
    """Tests for GUARDRAIL metric configuration."""

    def test_metrics_have_required_fields(self):
        """Test that all metrics have required fields."""
        planner_info = PlannerInfo(
            name="mock_guardrail_multi_metric",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_multi_metric(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        for metric in guardrail.definition.metrics:
            assert metric.type is not None
            assert metric.name is not None
            assert metric.threshold is not None
            assert metric.weight is not None
            assert 0 <= metric.threshold <= 1.0
            assert metric.weight > 0

    def test_metric_params_are_valid(self):
        """Test that metric params are properly configured."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Check LENGTH_CHECK metric has min/max
        length_metrics = [m for m in guardrail.definition.metrics
                         if m.type == GuardrailMetricType.LENGTH_CHECK]
        for metric in length_metrics:
            assert "min" in metric.params
            assert "max" in metric.params
            assert metric.params["min"] >= 0
            assert metric.params["max"] > metric.params["min"]

        # Check REGEX_MATCH metric has pattern
        regex_metrics = [m for m in guardrail.definition.metrics
                        if m.type == GuardrailMetricType.REGEX_MATCH]
        for metric in regex_metrics:
            assert "pattern" in metric.params
            assert isinstance(metric.params["pattern"], str)


class TestGuardrailDependencies:
    """Tests for GUARDRAIL node dependencies."""

    def test_guardrail_depends_on_upstream(self):
        """Test that guardrail node has correct upstream dependencies."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Guardrail should depend on at least one upstream node
        assert len(guardrail.dependencies) > 0

        # Dependencies should be valid node IDs in the plan
        all_node_ids = {n.task_id for n in plan.nodes}
        for dep_id in guardrail.dependencies:
            assert dep_id in all_node_ids, f"Dependency {dep_id} not in plan"

    def test_downstream_nodes_depend_on_guardrail(self):
        """Test that downstream nodes depend on the guardrail."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Get pass and fail path target IDs
        pass_targets = []
        fail_targets = []
        for path in guardrail.definition.paths:
            if path.path_id == "pass":
                pass_targets.extend(path.target_node_ids)
            elif path.path_id == "fail":
                fail_targets.extend(path.target_node_ids)

        # All target nodes should depend on the guardrail
        for node in plan.nodes:
            if node.task_id in pass_targets or node.task_id in fail_targets:
                assert guardrail.task_id in node.dependencies, \
                    f"Node {node.task_id} should depend on guardrail {guardrail.task_id}"


class TestGuardrailSerialization:
    """Tests for GUARDRAIL plan serialization."""

    def test_plan_can_be_serialized(self):
        """Test that guardrail plan can be serialized to JSON."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        # Should be able to serialize to dict
        plan_dict = plan.model_dump()
        assert plan_dict is not None
        assert "nodes" in plan_dict

    def test_plan_can_be_deserialized(self):
        """Test that guardrail plan can be deserialized from JSON."""
        from marie.query_planner.base import QueryPlan

        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        original_plan = query_planner_mock_guardrail_simple(planner_info)

        # Serialize and deserialize
        plan_dict = original_plan.model_dump()
        restored_plan = QueryPlan.model_validate(plan_dict)

        # Verify restored plan matches original
        assert len(restored_plan.nodes) == len(original_plan.nodes)

        original_ids = {n.task_id for n in original_plan.nodes}
        restored_ids = {n.task_id for n in restored_plan.nodes}
        assert original_ids == restored_ids


class TestGuardrailInputSource:
    """Tests for GUARDRAIL input source configuration."""

    def test_input_source_is_valid_jsonpath(self):
        """Test that input source is a valid JSONPath expression."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        input_source = guardrail.definition.input_source
        assert input_source is not None
        assert input_source.startswith("$"), "Input source should be a JSONPath starting with $"

    def test_input_source_references_upstream_node(self):
        """Test that input source references an upstream node's output."""
        planner_info = PlannerInfo(
            name="mock_guardrail_simple",
            base_id=generate_job_id(),
        )
        plan = query_planner_mock_guardrail_simple(planner_info)

        guardrail_nodes = [n for n in plan.nodes if hasattr(n, 'definition') and
                          isinstance(n.definition, GuardrailQueryDefinition)]
        guardrail = guardrail_nodes[0]

        # Input source should reference an upstream node ID
        input_source = guardrail.definition.input_source
        upstream_ids = guardrail.dependencies

        # Check if any upstream ID is referenced in the input source
        references_upstream = any(dep_id in input_source for dep_id in upstream_ids)
        assert references_upstream, f"Input source {input_source} should reference upstream node"
