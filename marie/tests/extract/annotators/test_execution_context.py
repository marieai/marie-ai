"""
Tests for ExecutionContext (simplified API).
"""

from typing import Set

import pytest

from marie.extract.annotators.execution_context import (
    ExecutionContext,
    ExecutionContextConfig,
    ExecutionContextEvaluator,
)


class MockRunContext:
    """Mock RunContext for testing."""

    def __init__(self, results: dict = None):
        self._results = results or {}

    def get(self, key: str, from_task: str = None, default=None):
        if from_task and from_task in self._results:
            return self._results[from_task].get(key, default)
        return default


class TestExecutionContextConfig:
    """Tests for ExecutionContextConfig."""

    def test_from_config_empty(self):
        """Test creating ExecutionContextConfig with empty config."""
        config = ExecutionContextConfig.from_config({})
        assert config.depends_on == []

    def test_from_config_simple_list(self):
        """Test creating ExecutionContextConfig with simple list format."""
        config = ExecutionContextConfig.from_config(
            {"depends_on": ["tables", "claims", "ocr"]}
        )
        assert config.depends_on == ["tables", "claims", "ocr"]

    def test_from_config_dict_format(self):
        """Test creating ExecutionContextConfig with dict format (backward compatible)."""
        config = ExecutionContextConfig.from_config(
            {
                "depends_on": [
                    {"task_id": "tables"},
                    {"task_id": "claims"},
                ]
            }
        )
        assert config.depends_on == ["tables", "claims"]

    def test_from_config_mixed_format(self):
        """Test creating ExecutionContextConfig with mixed format."""
        config = ExecutionContextConfig.from_config(
            {
                "depends_on": [
                    "tables",
                    {"task_id": "claims"},
                    "ocr",
                ]
            }
        )
        assert config.depends_on == ["tables", "claims", "ocr"]

    def test_from_config_empty_task_id_ignored(self):
        """Test that empty task_ids are ignored."""
        config = ExecutionContextConfig.from_config(
            {
                "depends_on": [
                    {"task_id": "tables"},
                    {"task_id": ""},  # Empty - should be ignored
                    {"other_key": "value"},  # No task_id - should be ignored
                ]
            }
        )
        assert config.depends_on == ["tables"]


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_load_upstream_data_success(self):
        """Test loading upstream data successfully."""
        upstream_results = {
            "tables": {
                "ANNOTATOR_RESULTS": {
                    "task_id": "tables",
                    "pages": {1: {"data": {"some": "data"}}},
                }
            }
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})

        exec_ctx = ExecutionContext(config, run_context)

        assert exec_ctx.has_upstream_data("tables")
        assert exec_ctx.get_upstream_data("tables") is not None
        assert exec_ctx.loaded_count == 1
        assert exec_ctx.dependency_count == 1

    def test_load_upstream_data_missing(self):
        """Test handling missing upstream data."""
        run_context = MockRunContext({})  # No results
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})

        exec_ctx = ExecutionContext(config, run_context)

        assert not exec_ctx.has_upstream_data("tables")
        assert exec_ctx.get_upstream_data("tables") is None
        assert exec_ctx.loaded_count == 0
        assert exec_ctx.dependency_count == 1

    def test_load_multiple_upstream_tasks(self):
        """Test loading data from multiple upstream tasks."""
        upstream_results = {
            "tables": {
                "ANNOTATOR_RESULTS": {"task_id": "tables", "data": "tables_data"}
            },
            "claims": {
                "ANNOTATOR_RESULTS": {"task_id": "claims", "data": "claims_data"}
            },
            "ocr": {"ANNOTATOR_RESULTS": {"task_id": "ocr", "data": "ocr_data"}},
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config(
            {"depends_on": ["tables", "claims", "ocr"]}
        )

        exec_ctx = ExecutionContext(config, run_context)

        assert exec_ctx.dependency_count == 3
        assert exec_ctx.loaded_count == 3
        assert exec_ctx.has_upstream_data("tables")
        assert exec_ctx.has_upstream_data("claims")
        assert exec_ctx.has_upstream_data("ocr")

    def test_partial_load(self):
        """Test partial loading when some upstream tasks are missing."""
        upstream_results = {
            "tables": {
                "ANNOTATOR_RESULTS": {"task_id": "tables", "data": "tables_data"}
            },
            # "claims" is missing
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config(
            {"depends_on": ["tables", "claims"]}
        )

        exec_ctx = ExecutionContext(config, run_context)

        assert exec_ctx.dependency_count == 2
        assert exec_ctx.loaded_count == 1
        assert exec_ctx.has_upstream_data("tables")
        assert not exec_ctx.has_upstream_data("claims")

    def test_get_all_upstream_data(self):
        """Test getting all upstream data at once."""
        upstream_results = {
            "tables": {"ANNOTATOR_RESULTS": {"task_id": "tables", "value": 1}},
            "claims": {"ANNOTATOR_RESULTS": {"task_id": "claims", "value": 2}},
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config(
            {"depends_on": ["tables", "claims"]}
        )

        exec_ctx = ExecutionContext(config, run_context)
        all_data = exec_ctx.get_all_upstream_data()

        assert len(all_data) == 2
        assert "tables" in all_data
        assert "claims" in all_data
        assert all_data["tables"]["task_id"] == "tables"
        assert all_data["claims"]["task_id"] == "claims"

    def test_get_all_upstream_data_returns_copy(self):
        """Test that get_all_upstream_data returns a copy."""
        upstream_results = {
            "tables": {"ANNOTATOR_RESULTS": {"task_id": "tables"}},
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})

        exec_ctx = ExecutionContext(config, run_context)
        all_data = exec_ctx.get_all_upstream_data()

        # Modifying returned dict shouldn't affect internal state
        all_data["new_key"] = "new_value"
        assert "new_key" not in exec_ctx.get_all_upstream_data()

    def test_no_dependencies(self):
        """Test ExecutionContext with no dependencies."""
        run_context = MockRunContext({})
        config = ExecutionContextConfig.from_config({})

        exec_ctx = ExecutionContext(config, run_context)

        assert exec_ctx.dependency_count == 0
        assert exec_ctx.loaded_count == 0
        assert exec_ctx.get_all_upstream_data() == {}

    def test_get_upstream_data_for_unconfigured_task(self):
        """Test getting data for a task not in depends_on."""
        upstream_results = {
            "tables": {"ANNOTATOR_RESULTS": {"task_id": "tables"}},
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})

        exec_ctx = ExecutionContext(config, run_context)

        # "claims" was not configured as a dependency
        assert exec_ctx.get_upstream_data("claims") is None
        assert not exec_ctx.has_upstream_data("claims")


class MockDocument:
    """Mock UnstructuredDocument for testing."""

    def __init__(self, page_count: int = 5):
        self.page_count = page_count


class TestExecutionContextEvaluator:
    """Tests for ExecutionContextEvaluator abstract class."""

    def test_evaluator_is_abstract(self):
        """Test that ExecutionContextEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExecutionContextEvaluator()

    def test_custom_evaluator_implementation(self):
        """Test implementing a custom evaluator."""

        class SimpleEvaluator(ExecutionContextEvaluator):
            """Simple evaluator that returns specific pages."""

            def __init__(self, eligible: Set[int], should_skip_value: bool = False):
                self._eligible = eligible
                self._should_skip = should_skip_value

            def get_eligible_pages(self, document) -> Set[int]:
                return self._eligible

            def should_skip(self, document) -> bool:
                return self._should_skip

        evaluator = SimpleEvaluator({0, 2, 4}, should_skip_value=False)
        document = MockDocument(page_count=5)

        assert evaluator.get_eligible_pages(document) == {0, 2, 4}
        assert evaluator.should_skip(document) is False

    def test_evaluator_with_execution_context(self):
        """Test evaluator that uses ExecutionContext data."""

        class DataBasedEvaluator(ExecutionContextEvaluator):
            """Evaluator that uses upstream data to determine eligible pages."""

            def __init__(self, exec_ctx: ExecutionContext):
                self.ctx = exec_ctx

            def get_eligible_pages(self, document) -> Set[int]:
                tables_data = self.ctx.get_upstream_data("tables")
                if not tables_data:
                    return set(range(document.page_count))

                # Example: return pages that have data in the upstream results
                pages = tables_data.get("pages", {})
                return {int(p) - 1 for p in pages.keys()}  # Convert to 0-indexed

            def should_skip(self, document) -> bool:
                return len(self.get_eligible_pages(document)) == 0

        # Setup upstream data
        upstream_results = {
            "tables": {
                "ANNOTATOR_RESULTS": {
                    "task_id": "tables",
                    "pages": {1: {"data": "page1"}, 3: {"data": "page3"}},
                }
            }
        }
        run_context = MockRunContext(upstream_results)
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})
        exec_ctx = ExecutionContext(config, run_context)

        evaluator = DataBasedEvaluator(exec_ctx)
        document = MockDocument(page_count=5)

        # Pages 1 and 3 in upstream (1-indexed) -> pages 0 and 2 (0-indexed)
        assert evaluator.get_eligible_pages(document) == {0, 2}
        assert evaluator.should_skip(document) is False

    def test_evaluator_should_skip_when_no_data(self):
        """Test evaluator skips when no upstream data."""

        class StrictEvaluator(ExecutionContextEvaluator):
            """Evaluator that skips if no upstream data."""

            def __init__(self, exec_ctx: ExecutionContext):
                self.ctx = exec_ctx

            def get_eligible_pages(self, document) -> Set[int]:
                if not self.ctx.has_upstream_data("tables"):
                    return set()
                return set(range(document.page_count))

            def should_skip(self, document) -> bool:
                return len(self.get_eligible_pages(document)) == 0

        run_context = MockRunContext({})  # No data
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})
        exec_ctx = ExecutionContext(config, run_context)

        evaluator = StrictEvaluator(exec_ctx)
        document = MockDocument(page_count=5)

        assert evaluator.get_eligible_pages(document) == set()
        assert evaluator.should_skip(document) is True
