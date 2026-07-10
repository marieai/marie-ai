"""
Unit tests for run_params delivery into the kb_indexing planner's node params.

Verifies that metadata (including nested run_params) submitted with a job
flows through PlannerInfo into the EXTRACT and EMBED node definitions, and
that an empty metadata dict preserves today's static defaults.
"""

from marie.query_planner.base import PlannerInfo
from marie.query_planner.kb_indexing_planner import query_planner_kb_indexing


def _plan(metadata):
    info = PlannerInfo(name="kb_indexing", base_id="0" * 32, metadata=metadata)
    return query_planner_kb_indexing(info)


def test_run_params_flow_into_nodes():
    md = {
        "source_id": "s1",
        "index_name": "i1",
        "run_params": {
            "parse_mode": "agent",
            "multimodal": True,
            "layout_options": {"precise_bounding_boxes": True},
            "cache_options": {"invalidate": False, "disabled": False},
        },
    }
    plan = _plan(md)
    extract = next(n for n in plan.nodes if "EXTRACT" in n.query_str)
    embed = next(n for n in plan.nodes if "EMBED" in n.query_str)
    assert extract.definition.endpoint == "extract_executor://document/extract"
    assert extract.definition.params["parse_mode"] == "agent"
    assert extract.definition.params["layout_options"]["precise_bounding_boxes"] is True
    assert extract.definition.params["cache_options"]["invalidate"] is False
    assert embed.definition.params["source_id"] == "s1"
    assert embed.definition.params["index_name"] == "i1"
    assert embed.definition.params["multimodal"] is True


def test_empty_metadata_keeps_legacy_defaults():
    plan = _plan({})
    extract = next(n for n in plan.nodes if "EXTRACT" in n.query_str)
    embed = next(n for n in plan.nodes if "EMBED" in n.query_str)
    assert extract.definition.endpoint == "extract_executor://document/extract"
    assert "parse_mode" not in extract.definition.params
    assert "layout_options" not in extract.definition.params
    assert "cache_options" not in extract.definition.params
    assert "source_id" not in embed.definition.params
    assert "index_name" not in embed.definition.params
    assert "multimodal" not in embed.definition.params
    assert embed.definition.params["node_type"] == "document"


def test_none_metadata_keeps_legacy_defaults():
    plan = _plan(None)
    extract = next(n for n in plan.nodes if "EXTRACT" in n.query_str)
    assert extract.definition.endpoint == "extract_executor://document/extract"
