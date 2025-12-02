"""Tests for asset decorators."""

import pytest

from marie.assets import (
    asset,
    get_asset_spec,
    get_asset_specs,
    graph_asset,
    is_asset,
    is_graph_asset,
    is_multi_asset,
    is_observable_asset,
    multi_asset,
    observable_source_asset,
)
from marie.assets.models import AssetKind, AssetSpec


class TestAssetDecorator:
    """Test @asset decorator."""

    def test_basic_asset(self):
        """Test basic @asset usage."""

        @asset
        def my_asset():
            return "data"

        assert is_asset(my_asset)
        spec = get_asset_spec(my_asset)
        assert spec is not None
        assert spec.key == "my_asset"
        assert spec.is_primary is True

    def test_asset_with_key(self):
        """Test @asset with explicit key."""

        @asset(key="ocr/text")
        def extract_text():
            return "text"

        spec = get_asset_spec(extract_text)
        assert spec.key == "ocr/text"

    def test_asset_with_key_prefix(self):
        """Test @asset with key_prefix."""

        @asset(key_prefix="ocr")
        def text():
            return "text"

        spec = get_asset_spec(text)
        assert spec.key == "ocr/text"

    def test_asset_with_list_key_prefix(self):
        """Test @asset with list key_prefix."""

        @asset(key_prefix=["ocr", "v2"])
        def text():
            return "text"

        spec = get_asset_spec(text)
        assert spec.key == "ocr/v2/text"

    def test_asset_with_metadata(self):
        """Test @asset with metadata."""

        @asset(
            key="ocr/text",
            kind=AssetKind.TEXT,
            kinds=["craft_ocr", "torch"],
            metadata={"model": "craft-v2"},
            description="Extracted text",
            group_name="ocr",
            code_version="1.0.0",
            owners=["team@company.com"],
            tags={"priority": "high"},
            compute_kind="python",
        )
        def extract_text():
            return "text"

        spec = get_asset_spec(extract_text)
        assert spec.key == "ocr/text"
        assert spec.kind == AssetKind.TEXT
        assert "craft_ocr" in spec.kinds
        assert "torch" in spec.kinds
        assert spec.metadata["model"] == "craft-v2"
        assert spec.metadata["group_name"] == "ocr"
        assert spec.metadata["code_version"] == "1.0.0"
        assert spec.metadata["owners"] == ["team@company.com"]
        assert spec.metadata["tags"]["priority"] == "high"
        assert spec.metadata["compute_kind"] == "python"
        assert spec.description == "Extracted text"

    def test_asset_with_deps(self):
        """Test @asset with dependencies."""

        @asset(deps=["upstream1", "upstream2"])
        def downstream():
            return "data"

        assert downstream.__asset_deps__ == ["upstream1", "upstream2"]

    def test_asset_with_is_required(self):
        """Test @asset with is_required flag."""

        @asset(is_required=False)
        def optional_asset():
            return "data"

        spec = get_asset_spec(optional_asset)
        assert spec.is_required is False


class TestMultiAssetDecorator:
    """Test @multi_asset decorator."""

    def test_multi_asset_with_specs(self):
        """Test @multi_asset with spec dicts."""

        @multi_asset(
            specs=[
                {"key": "ocr/text", "kind": "text", "is_primary": True},
                {"key": "ocr/bboxes", "kind": "bbox"},
            ]
        )
        def ocr_extraction():
            return {"ocr/text": "text", "ocr/bboxes": [[0, 0, 10, 10]]}

        assert is_multi_asset(ocr_extraction)
        specs = get_asset_specs(ocr_extraction)
        assert len(specs) == 2
        assert specs[0].key == "ocr/text"
        assert specs[0].kind == AssetKind.TEXT
        assert specs[0].is_primary is True
        assert specs[1].key == "ocr/bboxes"
        assert specs[1].kind == AssetKind.BBOX

    def test_multi_asset_with_key_prefix(self):
        """Test @multi_asset with key_prefix."""

        @multi_asset(
            key_prefix="ocr",
            specs=[
                {"key": "text", "kind": "text"},
                {"key": "bboxes", "kind": "bbox"},
            ],
        )
        def extraction():
            return {}

        specs = get_asset_specs(extraction)
        assert specs[0].key == "ocr/text"
        assert specs[1].key == "ocr/bboxes"

    def test_multi_asset_with_shared_metadata(self):
        """Test @multi_asset with shared metadata."""

        @multi_asset(
            specs=[
                {"key": "asset1", "kind": "json"},
                {"key": "asset2", "kind": "json"},
            ],
            group_name="test_group",
            code_version="2.0.0",
            owners=["team@company.com"],
            tags={"env": "prod"},
        )
        def multi_op():
            return {}

        specs = get_asset_specs(multi_op)
        for spec in specs:
            assert spec.metadata["group_name"] == "test_group"
            assert spec.metadata["code_version"] == "2.0.0"
            assert spec.metadata["owners"] == ["team@company.com"]
            assert spec.metadata["tags"]["env"] == "prod"

    def test_multi_asset_with_asset_specs(self):
        """Test @multi_asset with AssetSpec objects."""

        spec1 = AssetSpec(key="asset1", kind=AssetKind.JSON)
        spec2 = AssetSpec(key="asset2", kind=AssetKind.TEXT)

        @multi_asset(assets=[spec1, spec2])
        def multi_op():
            return {}

        specs = get_asset_specs(multi_op)
        assert len(specs) == 2
        assert specs[0].key == "asset1"
        assert specs[1].key == "asset2"

    def test_multi_asset_requires_specs_or_assets(self):
        """Test @multi_asset raises error without specs or assets."""

        with pytest.raises(ValueError, match="Either 'specs' or 'assets' must be provided"):

            @multi_asset
            def bad_multi():
                return {}


class TestGraphAssetDecorator:
    """Test @graph_asset decorator."""

    def test_graph_asset_basic(self):
        """Test basic @graph_asset usage."""

        @graph_asset
        def my_graph():
            return "result"

        assert is_graph_asset(my_graph)
        assert my_graph.__asset_key__ == "my_graph"

    def test_graph_asset_with_key(self):
        """Test @graph_asset with explicit key."""

        @graph_asset(key="reports/summary")
        def summary():
            return "report"

        assert summary.__asset_key__ == "reports/summary"

    def test_graph_asset_with_key_prefix(self):
        """Test @graph_asset with key_prefix."""

        @graph_asset(key_prefix="reports")
        def weekly():
            return "report"

        assert weekly.__asset_key__ == "reports/weekly"


class TestObservableSourceAsset:
    """Test @observable_source_asset decorator."""

    def test_observable_asset_basic(self):
        """Test basic @observable_source_asset usage."""

        @observable_source_asset(key="s3/documents")
        def observe_s3():
            return {"file_count": 42}

        assert is_observable_asset(observe_s3)
        assert observe_s3.__asset_key__ == "s3/documents"

    def test_observable_asset_with_metadata(self):
        """Test @observable_source_asset with metadata."""

        @observable_source_asset(
            key="db/customers",
            description="Customer database",
            metadata={"database": "postgres", "table": "customers"},
        )
        def observe_db():
            return {"row_count": 1000}

        assert observe_db.__asset_description__ == "Customer database"
        assert observe_db.__asset_metadata__["database"] == "postgres"


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_asset_detection(self):
        """Test is_asset() detection."""

        @asset
        def my_asset():
            return "data"

        def regular_function():
            return "data"

        assert is_asset(my_asset) is True
        assert is_asset(regular_function) is False

    def test_is_multi_asset_detection(self):
        """Test is_multi_asset() detection."""

        @multi_asset(specs=[{"key": "a", "kind": "json"}])
        def multi():
            return {}

        @asset
        def single():
            return "data"

        assert is_multi_asset(multi) is True
        assert is_multi_asset(single) is False

    def test_get_asset_spec_returns_none_for_non_asset(self):
        """Test get_asset_spec() returns None for non-decorated functions."""

        def regular_function():
            return "data"

        assert get_asset_spec(regular_function) is None

    def test_get_asset_specs_returns_none_for_non_multi_asset(self):
        """Test get_asset_specs() returns None for non-multi-asset functions."""

        @asset
        def single_asset():
            return "data"

        assert get_asset_specs(single_asset) is None
