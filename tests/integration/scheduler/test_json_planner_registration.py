"""
Test JSON-based query planner registration.

This test demonstrates how Marie Studio can publish query plan templates
to the Marie-AI gateway via JSON without requiring Python code.
"""

import json
import tempfile
from pathlib import Path

import pytest

from marie.query_planner import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    Query,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
)


@pytest.fixture
def sample_query_plan_json():
    """Create a sample query plan as JSON (similar to what Marie Studio would send)"""
    plan = QueryPlan(
        nodes=[
            Query(
                task_id="01930d8c-0000-7000-8000-000000000000",
                query_str="START: Initialize processing",
                dependencies=[],
                node_type=QueryType.COMPUTE,
                definition=NoopQueryDefinition(),
            ),
            Query(
                task_id="01930d8c-0001-7000-8000-000000000000",
                query_str="Extract document data",
                dependencies=["01930d8c-0000-7000-8000-000000000000"],
                node_type=QueryType.COMPUTE,
                definition=ExecutorEndpointQueryDefinition(
                    method="EXECUTOR_ENDPOINT",
                    endpoint="/extract",
                    params={"layout": "invoice"},
                ),
            ),
            Query(
                task_id="01930d8c-0002-7000-8000-000000000000",
                query_str="END: Complete processing",
                dependencies=["01930d8c-0001-7000-8000-000000000000"],
                node_type=QueryType.COMPUTE,
                definition=NoopQueryDefinition(),
            ),
        ]
    )

    # Convert to JSON dict (this is what Marie Studio would send)
    return plan.model_dump()


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry before and after each test"""
    # Store original state
    original_plans = QueryPlanRegistry._plans.copy()
    original_metadata = QueryPlanRegistry._metadata.copy()

    yield

    # Restore original state
    QueryPlanRegistry._plans = original_plans
    QueryPlanRegistry._metadata = original_metadata


def test_register_json_planner(sample_query_plan_json):
    """Test registering a planner from JSON definition"""
    success = QueryPlanRegistry.register_from_json(
        name="test_json_planner",
        plan_definition=sample_query_plan_json,
        description="Test planner for invoice processing",
        version="1.0.0",
        tags=["invoice", "extraction"],
        category="document_processing",
    )

    assert success is True

    # Verify planner is registered
    assert "test_json_planner" in QueryPlanRegistry.list_planners()

    # Verify metadata
    metadata = QueryPlanRegistry.get_metadata("test_json_planner")
    assert metadata is not None
    assert metadata.name == "test_json_planner"
    assert metadata.description == "Test planner for invoice processing"
    assert metadata.version == "1.0.0"
    assert metadata.tags == ["invoice", "extraction"]
    assert metadata.category == "document_processing"
    assert metadata.source_type == "json"
    assert metadata.plan_definition == sample_query_plan_json


def test_get_json_planner(sample_query_plan_json):
    """Test retrieving a registered JSON planner"""
    QueryPlanRegistry.register_from_json(
        name="test_retrieve",
        plan_definition=sample_query_plan_json,
        description="Test retrieval",
    )

    # Get the planner function
    planner_func = QueryPlanRegistry.get("test_retrieve")
    assert planner_func is not None

    # Call the planner function (it should return the QueryPlan)
    from marie.job.job_manager import generate_job_id
    from marie.query_planner import PlannerInfo

    planner_info = PlannerInfo(name="test_retrieve", base_id=generate_job_id())
    result = planner_func(planner_info)

    assert isinstance(result, QueryPlan)
    assert len(result.nodes) == 3


def test_unregister_json_planner(sample_query_plan_json):
    """Test unregistering a JSON planner"""
    QueryPlanRegistry.register_from_json(
        name="test_unregister", plan_definition=sample_query_plan_json
    )

    assert "test_unregister" in QueryPlanRegistry.list_planners()

    # Unregister
    success = QueryPlanRegistry.unregister("test_unregister")
    assert success is True

    # Verify it's gone
    assert "test_unregister" not in QueryPlanRegistry.list_planners()
    assert QueryPlanRegistry.get_metadata("test_unregister") is None


def test_list_planners_with_metadata(sample_query_plan_json):
    """Test listing all planners with metadata"""
    # Register a JSON planner
    QueryPlanRegistry.register_from_json(
        name="test_list_json",
        plan_definition=sample_query_plan_json,
        description="JSON planner",
    )

    planners = QueryPlanRegistry.list_planners_with_metadata()

    # Find our planner
    test_planner = next((p for p in planners if p["name"] == "test_list_json"), None)

    assert test_planner is not None
    assert test_planner["description"] == "JSON planner"
    assert test_planner["source_type"] == "json"


def test_persist_and_load_json_planners(sample_query_plan_json):
    """Test persisting JSON planners to storage and loading them back"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set storage path
        QueryPlanRegistry.set_storage_path(tmpdir)

        # Register a planner (should auto-save)
        QueryPlanRegistry.register_from_json(
            name="test_persist",
            plan_definition=sample_query_plan_json,
            description="Persisted planner",
            version="2.0.0",
            tags=["test"],
        )

        # Verify file was created
        planner_file = Path(tmpdir) / "test_persist.json"
        assert planner_file.exists()

        # Read and verify content
        with open(planner_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["name"] == "test_persist"
        assert saved_data["description"] == "Persisted planner"
        assert saved_data["version"] == "2.0.0"

        # Clear registry
        QueryPlanRegistry._plans.clear()
        QueryPlanRegistry._metadata.clear()

        # Load from storage
        result = QueryPlanRegistry.load_json_planners_from_storage()

        assert result["loaded"] == 1
        assert "test_persist" in result["planners"]
        assert "test_persist" in QueryPlanRegistry.list_planners()


def test_duplicate_planner_registration(sample_query_plan_json):
    """Test that registering a duplicate planner fails"""
    QueryPlanRegistry.register_from_json(
        name="test_duplicate", plan_definition=sample_query_plan_json
    )

    # Try to register again with same name
    success = QueryPlanRegistry.register_from_json(
        name="test_duplicate", plan_definition=sample_query_plan_json
    )

    assert success is False


def test_invalid_plan_definition():
    """Test that invalid plan definition fails validation"""
    invalid_plan = {"nodes": [{"invalid": "structure"}]}

    success = QueryPlanRegistry.register_from_json(
        name="test_invalid", plan_definition=invalid_plan
    )

    assert success is False
    assert "test_invalid" not in QueryPlanRegistry.list_planners()


if __name__ == "__main__":
    # Run a simple demo
    print("=" * 60)
    print("JSON Planner Registration Demo")
    print("=" * 60)

    # Create sample plan
    plan = QueryPlan(
        nodes=[
            Query(
                task_id="01930d8c-0000-7000-8000-000000000000",
                query_str="START",
                dependencies=[],
                node_type=QueryType.COMPUTE,
                definition=NoopQueryDefinition(),
            ),
            Query(
                task_id="01930d8c-0001-7000-8000-000000000000",
                query_str="Process",
                dependencies=["01930d8c-0000-7000-8000-000000000000"],
                node_type=QueryType.COMPUTE,
                definition=ExecutorEndpointQueryDefinition(
                    method="EXECUTOR_ENDPOINT", endpoint="/process", params={}
                ),
            ),
        ]
    )

    plan_json = plan.model_dump()

    # Register from JSON
    print("\n1. Registering planner from JSON...")
    success = QueryPlanRegistry.register_from_json(
        name="demo_planner",
        plan_definition=plan_json,
        description="Demo planner for testing",
        tags=["demo", "test"],
    )
    print(f"   Registration successful: {success}")

    # List all planners
    print("\n2. Listing all planners...")
    planners = QueryPlanRegistry.list_planners_with_metadata()
    for planner in planners:
        print(f"   - {planner['name']} ({planner['source_type']})")

    # Get metadata
    print("\n3. Getting planner metadata...")
    metadata = QueryPlanRegistry.get_metadata("demo_planner")
    print(f"   Name: {metadata.name}")
    print(f"   Description: {metadata.description}")
    print(f"   Tags: {metadata.tags}")
    print(f"   Nodes: {len(metadata.plan_definition['nodes'])}")

    # Unregister
    print("\n4. Unregistering planner...")
    success = QueryPlanRegistry.unregister("demo_planner")
    print(f"   Unregistration successful: {success}")

    print("\n" + "=" * 60)
