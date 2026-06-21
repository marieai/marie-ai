"""Tests for LLMAnnotator refinement passes.

Covers:
- ProcessingKey parsing
- RefinementContextProvider variable injection
- compute_fingerprints for all known formats
- PassValidationReport construction via _validate_pass_outputs
- _compare_pass_reports acceptance / rejection logic
- _engine_for_pass routing
- _promote_pass_atomically correctness and recovery
- _has_completed_results skip logic
- Integration-style tests with mocked LLM
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from marie.extract.annotators.refinement_provider import (
    PassValidationReport,
    ProcessingKey,
    RefinementContextProvider,
    compute_fingerprints,
)

# ======================================================================
# ProcessingKey
# ======================================================================


class TestProcessingKey:
    def test_from_filename_page_only(self):
        key = ProcessingKey.from_filename("frame_0001.json")
        assert key == ProcessingKey(page_number=1, unit_index=None)

    def test_from_filename_with_unit(self):
        key = ProcessingKey.from_filename("frame_0001_t0.json")
        assert key == ProcessingKey(page_number=1, unit_index=0)

    def test_from_filename_multidigit_page(self):
        key = ProcessingKey.from_filename("frame_0012.json")
        assert key == ProcessingKey(page_number=12, unit_index=None)

    def test_from_filename_multidigit_unit(self):
        key = ProcessingKey.from_filename("frame_0002_t1.json")
        assert key == ProcessingKey(page_number=2, unit_index=1)

    def test_from_filename_invalid_raises(self):
        with pytest.raises(ValueError):
            ProcessingKey.from_filename("unknown_file.json")

    def test_from_filename_no_extension_raises(self):
        with pytest.raises(ValueError):
            ProcessingKey.from_filename("frame_0001.png")

    def test_frozen(self):
        key = ProcessingKey(1, 0)
        with pytest.raises(AttributeError):
            key.page_number = 2  # type: ignore[misc]

    def test_hashable_for_dict_keys(self):
        d = {ProcessingKey(1, None): "a", ProcessingKey(1, 0): "b"}
        assert len(d) == 2


# ======================================================================
# RefinementContextProvider
# ======================================================================


class TestRefinementContextProvider:
    @staticmethod
    def _make_doc(page_count: int = 3):
        doc = MagicMock()
        doc.page_count = page_count
        return doc

    def test_eligible_pages_returns_all(self):
        provider = RefinementContextProvider({}, "test-ann")
        doc = self._make_doc(5)
        assert provider.get_eligible_pages(doc) == {1, 2, 3, 4, 5}

    def test_get_variables_with_payload(self):
        payload = '{"extractions": [{"label": "total", "value": "100"}]}'
        results = {ProcessingKey(1, None): payload}
        provider = RefinementContextProvider(results, "test-ann")
        doc = self._make_doc()

        variables = provider.get_variables(doc, page_number=1, unit=None)
        assert "PREVIOUS_EXTRACTION" in variables
        assert payload in variables["PREVIOUS_EXTRACTION"]
        assert "Previous Extraction Results" in variables["PREVIOUS_EXTRACTION"]

    def test_get_variables_empty_when_no_match(self):
        provider = RefinementContextProvider({}, "test-ann")
        doc = self._make_doc()

        variables = provider.get_variables(doc, page_number=1, unit=None)
        assert variables["PREVIOUS_EXTRACTION"] == ""

    def test_get_variables_with_unit(self):
        payload = '{"extractions": []}'
        results = {ProcessingKey(1, 0): payload}
        provider = RefinementContextProvider(results, "test-ann")
        doc = self._make_doc()
        unit = MagicMock()
        unit.index = 0

        variables = provider.get_variables(doc, page_number=1, unit=unit)
        assert payload in variables["PREVIOUS_EXTRACTION"]


# ======================================================================
# compute_fingerprints
# ======================================================================


class TestComputeFingerprints:
    def test_standard_extraction_result(self):
        data = {
            "extractions": [
                {"label": "total", "line_number": 5, "value": "100"},
                {"label": "date", "line_number": 2, "value": "2024-01-01"},
            ]
        }
        fp = compute_fingerprints(data)
        assert fp == {("total", 5), ("date", 2)}

    def test_simplified_no_label(self):
        data = {
            "extractions": [
                {"line_number": 10, "value": "some text here"},
                {"line_number": 20, "value": "more text"},
            ]
        }
        fp = compute_fingerprints(data)
        assert fp == {(10, "some text here"), (20, "more text")}

    def test_simplified_truncates_value(self):
        data = {
            "extractions": [
                {"line_number": 1, "value": "x" * 100},
            ]
        }
        fp = compute_fingerprints(data)
        (entry,) = fp
        assert entry == (1, "x" * 50)

    def test_canonical_mapping(self):
        data = {
            "canonical_mapping": {"col_a": "Column A", "col_b": "Column B"},
            "unmapped": [],
        }
        fp = compute_fingerprints(data)
        assert fp == {("col_a",), ("col_b",)}

    def test_custom_table_format_falls_back(self):
        """Table format with name/rows/columns has extractions list but no label/line_number."""
        data = {
            "extractions": [
                {"name": "table1", "rows": 5, "columns": 3},
            ],
            "reasoning": {"text": "..."},
        }
        # No label and no line_number → falls through to fallback
        fp = compute_fingerprints(data)
        # Fallback: top-level keys
        assert fp == {("extractions",), ("reasoning",)}

    def test_fallback_top_level_keys(self):
        data = {"foo": 1, "bar": 2}
        fp = compute_fingerprints(data)
        assert fp == {("foo",), ("bar",)}

    def test_empty_extractions_uses_fallback(self):
        data = {"extractions": [], "metadata": {}}
        fp = compute_fingerprints(data)
        # Empty list → fallback
        assert fp == {("extractions",), ("metadata",)}


# ======================================================================
# Helpers: build a minimal LLMAnnotator for method-level tests
# ======================================================================


def _make_annotator(
    tmp_path,
    refine_passes: int = 0,
    expect_output: str = "json",
    pass_temperatures: Optional[List[float]] = None,
    pass_models: Optional[List[str]] = None,
    refinement_validation: Optional[dict] = None,
    **annotator_kwargs: Any,
):
    """Create a minimal LLMAnnotator with mocked dependencies."""
    # Create required directories
    working_dir = str(tmp_path / "work")
    os.makedirs(os.path.join(working_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "agent-output", "test-ann"), exist_ok=True)

    prompt_dir = str(tmp_path / "prompts")
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = os.path.join(prompt_dir, "test.j2")
    with open(prompt_path, "w") as f:
        f.write("Extract data. PREVIOUS_EXTRACTION OCR_DATA")

    model_config: Dict[str, Any] = {
        "model_name": "test-model",
        "prompt_path": "./test.j2",
        "expect_output": expect_output,
        "temperature": 0.0,
        "refine_passes": refine_passes,
    }
    if pass_temperatures is not None:
        model_config["pass_temperatures"] = pass_temperatures
    if pass_models is not None:
        model_config["pass_models"] = pass_models
    if refinement_validation is not None:
        model_config["refinement_validation"] = refinement_validation

    annotator_conf = {
        "name": "test-ann",
        "annotator_type": "llm",
        "model_config": model_config,
    }
    layout_conf = {"layout_id": "999"}

    with patch(
        "marie.extract.annotators.llm_annotator.route_llm_engine"
    ) as mock_engine, patch(
        "marie.extract.annotators.llm_annotator.ContextProviderManager"
    ) as mock_cpm:
        mock_engine.return_value = MagicMock(spec=["close"])
        mock_cpm_instance = MagicMock()
        mock_cpm_instance.has_providers.return_value = False
        mock_cpm.return_value = mock_cpm_instance

        ann = None
        # Import inside patch scope
        from marie.extract.annotators.llm_annotator import LLMAnnotator

        ann = LLMAnnotator(
            working_dir=working_dir,
            annotator_conf=annotator_conf,
            layout_conf=layout_conf,
            prompt_dir=prompt_dir,
            **annotator_kwargs,
        )
    return ann


# ======================================================================
# _validate_pass_outputs
# ======================================================================


class TestValidatePassOutputs:
    def test_rejects_malformed_json(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        pass_dir = str(tmp_path / "pass")
        os.makedirs(pass_dir)
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            f.write("{bad json")

        report = ann._validate_pass_outputs(pass_dir)
        assert not report.json_valid
        assert report.errors

    def test_handles_extraction_result_format(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        pass_dir = str(tmp_path / "pass")
        os.makedirs(pass_dir)
        data = {
            "extractions": [
                {"label": "total", "line_number": 5, "value": "100"},
                {"label": "date", "line_number": 2, "value": "2024-01-01"},
            ]
        }
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump(data, f)

        report = ann._validate_pass_outputs(pass_dir)
        assert report.json_valid
        assert report.file_count == 1
        assert report.total_element_count == 2
        assert ProcessingKey(1, None) in report.processing_keys
        fp = report.fingerprints_by_key[ProcessingKey(1, None)]
        assert ("total", 5) in fp
        assert ("date", 2) in fp

    def test_handles_mapping_format(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        pass_dir = str(tmp_path / "pass")
        os.makedirs(pass_dir)
        data = {"canonical_mapping": {"col_a": "A"}, "unmapped": []}
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump(data, f)

        report = ann._validate_pass_outputs(pass_dir)
        assert report.json_valid
        assert report.total_element_count == 2  # 2 top-level keys (no extractions)

    def test_nonexistent_dir(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        report = ann._validate_pass_outputs("/nonexistent/path")
        assert not report.json_valid

    def test_non_json_output_is_permissive(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1, expect_output="markdown")
        pass_dir = str(tmp_path / "pass")
        os.makedirs(pass_dir)
        with open(os.path.join(pass_dir, "frame_0001.md"), "w") as f:
            f.write("# Table\n| a | b |")

        report = ann._validate_pass_outputs(pass_dir)
        assert report.json_valid
        assert report.file_count == 1

    def test_skips_success_marker_file(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        pass_dir = str(tmp_path / "pass")
        os.makedirs(pass_dir)
        data = {"extractions": [{"label": "x", "line_number": 1, "value": "y"}]}
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump(data, f)
        with open(os.path.join(pass_dir, "_SUCCESS.yaml"), "w") as f:
            f.write("run_id: abc\n")

        report = ann._validate_pass_outputs(pass_dir)
        assert report.file_count == 1  # _SUCCESS.yaml excluded


# ======================================================================
# _compare_pass_reports
# ======================================================================


class TestComparePassReports:
    def _make_report(self, **kwargs) -> PassValidationReport:
        defaults = {
            "json_valid": True,
            "processing_keys": {ProcessingKey(1, None)},
            "file_count": 1,
            "total_element_count": 10,
            "fingerprints_by_key": {
                ProcessingKey(1, None): {("a", 1), ("b", 2)}
            },
            "total_json_size": 1000,
            "errors": [],
        }
        defaults.update(kwargs)
        return PassValidationReport(**defaults)

    def test_accepts_valid_refinement(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report()
        curr = self._make_report(total_element_count=12)
        assert ann._compare_pass_reports(prev, curr) is True

    def test_rejects_invalid_json(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report()
        curr = self._make_report(json_valid=False)
        assert ann._compare_pass_reports(prev, curr) is False

    def test_rejects_missing_processing_keys(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report(
            processing_keys={ProcessingKey(1, None), ProcessingKey(2, None)}
        )
        curr = self._make_report(
            processing_keys={ProcessingKey(1, None)}
        )
        assert ann._compare_pass_reports(prev, curr) is False

    def test_allows_different_keys_when_not_required(self, tmp_path):
        ann = _make_annotator(
            tmp_path,
            refine_passes=1,
            refinement_validation={"require_same_units": False, "max_segment_drop_ratio": 0.2},
        )
        prev = self._make_report(
            processing_keys={ProcessingKey(1, None), ProcessingKey(2, None)}
        )
        curr = self._make_report(processing_keys={ProcessingKey(1, None)})
        # Keys differ but not required — passes that gate
        # But element count drops from 10 to 10, which is fine
        assert ann._compare_pass_reports(prev, curr) is True

    def test_rejects_excessive_element_drop(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report(total_element_count=100)
        curr = self._make_report(total_element_count=70)  # 30% drop
        assert ann._compare_pass_reports(prev, curr) is False

    def test_allows_small_element_drop(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report(total_element_count=100)
        curr = self._make_report(total_element_count=85)  # 15% drop
        assert ann._compare_pass_reports(prev, curr) is True

    def test_rejects_catastrophic_size_shrink(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report(total_json_size=10000)
        curr = self._make_report(total_json_size=2000)  # 20% of original
        assert ann._compare_pass_reports(prev, curr) is False

    def test_rejects_fingerprint_regression(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        prev = self._make_report(
            fingerprints_by_key={
                ProcessingKey(1, None): {("a", 1), ("b", 2), ("c", 3), ("d", 4)}
            }
        )
        # Only 1/4 retained = 25% < 50%
        curr = self._make_report(
            fingerprints_by_key={
                ProcessingKey(1, None): {("a", 1), ("x", 5), ("y", 6)}
            }
        )
        assert ann._compare_pass_reports(prev, curr) is False


# ======================================================================
# _engine_for_pass
# ======================================================================


class TestEngineForPass:
    def test_returns_default_when_no_pass_models(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        assert ann._engine_for_pass(0) is ann.engine
        assert ann._engine_for_pass(1) is ann.engine

    def test_returns_correct_engine_per_pass(self, tmp_path):
        ann = _make_annotator(
            tmp_path,
            refine_passes=1,
            pass_models=["model_a", "model_b"],
        )
        with patch(
            "marie.extract.annotators.llm_annotator.route_llm_engine"
        ) as mock_route:
            engine_a = MagicMock()
            engine_b = MagicMock()
            mock_route.side_effect = lambda name, mm: (
                engine_a if name == "model_a" else engine_b
            )

            assert ann._engine_for_pass(0) is engine_a
            assert ann._engine_for_pass(1) is engine_b

    def test_falls_back_when_index_out_of_range(self, tmp_path):
        ann = _make_annotator(
            tmp_path,
            refine_passes=2,
            pass_models=["model_a", "model_b", "model_c"],
        )
        # Index 5 is out of range
        assert ann._engine_for_pass(5) is ann.engine


class TestSpanMetadata:
    def test_includes_queue_pool_id_when_provided(self, tmp_path):
        ann = _make_annotator(tmp_path, pool_id="document-small")
        ann.engine = MagicMock()
        ann.engine.model_string = "test-model"

        metadata = ann._build_span_metadata()

        assert metadata["pool_id"] == "document-small"


# ======================================================================
# _promote_pass_atomically
# ======================================================================


class TestPromotePassAtomically:
    def test_produces_live_dir_with_winning_artifacts(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        pass_dir = str(tmp_path / "pass_winner")
        os.makedirs(pass_dir)
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump({"extractions": []}, f)
        with open(os.path.join(pass_dir, "frame_0001.png"), "wb") as f:
            f.write(b"PNG")

        ann._promote_pass_atomically(pass_dir, "run123")

        live = ann.output_dir
        assert os.path.isfile(os.path.join(live, "frame_0001.json"))
        assert os.path.isfile(os.path.join(live, "frame_0001.png"))
        assert os.path.isfile(os.path.join(live, "_SUCCESS.yaml"))

        with open(os.path.join(live, "_SUCCESS.yaml")) as f:
            marker = yaml.safe_load(f)
        assert marker["run_id"] == "run123"
        assert marker["file_count"] == 2

    def test_stale_files_removed(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)

        # Pre-populate live dir with stale file
        live = ann.output_dir
        with open(os.path.join(live, "old_stale.json"), "w") as f:
            f.write("{}")

        pass_dir = str(tmp_path / "pass_new")
        os.makedirs(pass_dir)
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump({"extractions": []}, f)

        ann._promote_pass_atomically(pass_dir, "run456")

        live_files = os.listdir(live)
        assert "old_stale.json" not in live_files
        assert "frame_0001.json" in live_files

    def test_failed_promotion_recoverable(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        live = ann.output_dir

        # Put a valid file in live
        with open(os.path.join(live, "frame_0001.json"), "w") as f:
            json.dump({"extractions": [{"label": "original"}]}, f)

        pass_dir = str(tmp_path / "pass_bad")
        os.makedirs(pass_dir)
        with open(os.path.join(pass_dir, "frame_0001.json"), "w") as f:
            json.dump({"extractions": [{"label": "new"}]}, f)

        # Simulate rename failure by making staging → live rename fail
        original_rename = os.rename

        def mock_rename(src, dst):
            if dst == live and "staging" in src:
                raise OSError("Simulated rename failure")
            return original_rename(src, dst)

        with patch("os.rename", side_effect=mock_rename):
            with pytest.raises(OSError):
                ann._promote_pass_atomically(pass_dir, "runfail")

        # Live dir should still exist (recovered from backup)
        assert os.path.isdir(live)


# ======================================================================
# _has_completed_results
# ======================================================================


class TestHasCompletedResults:
    def test_false_for_empty_dir(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        assert ann._has_completed_results(ann.output_dir) is False

    def test_false_without_success_marker(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        with open(os.path.join(ann.output_dir, "frame_0001.json"), "w") as f:
            f.write("{}")
        assert ann._has_completed_results(ann.output_dir) is False

    def test_true_with_marker_and_results(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        with open(os.path.join(ann.output_dir, "frame_0001.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(ann.output_dir, "_SUCCESS.yaml"), "w") as f:
            f.write("run_id: abc\n")
        assert ann._has_completed_results(ann.output_dir) is True

    def test_legacy_mode_any_content_is_complete(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=0)
        with open(os.path.join(ann.output_dir, "frame_0001.json"), "w") as f:
            f.write("{}")
        assert ann._has_completed_results(ann.output_dir) is True

    def test_false_for_nonexistent_dir(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        assert ann._has_completed_results("/nonexistent") is False


# ======================================================================
# Recovery
# ======================================================================


class TestRecovery:
    def test_recover_cleans_partial_state(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        # Put partial files without marker
        with open(os.path.join(ann.output_dir, "frame_0001.json"), "w") as f:
            f.write("{}")

        ann._recover_or_reset_live_output()
        # Should be cleaned
        assert os.listdir(ann.output_dir) == []

    def test_recover_preserves_completed_state(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        with open(os.path.join(ann.output_dir, "frame_0001.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(ann.output_dir, "_SUCCESS.yaml"), "w") as f:
            f.write("run_id: abc\n")

        ann._recover_or_reset_live_output()
        # Should be preserved
        assert "_SUCCESS.yaml" in os.listdir(ann.output_dir)


# ======================================================================
# Integration-style tests with mocked LLM
# ======================================================================


class TestRefinementIntegration:
    @staticmethod
    def _make_doc(page_count: int = 1):
        doc = MagicMock()
        doc.page_count = page_count
        doc.source_metadata = {"ocr": [{"meta": {"page": 0}, "words": []}]}
        return doc

    @pytest.mark.asyncio
    async def test_pass0_output_injected_into_pass1(self, tmp_path):
        """Verify that pass 0 results appear in the refinement provider."""
        ann = _make_annotator(tmp_path, refine_passes=1)

        pass0_data = {"extractions": [{"label": "a", "line_number": 1, "value": "v"}]}
        pass1_data = {"extractions": [{"label": "a", "line_number": 1, "value": "v_refined"}]}

        call_count = {"n": 0}
        captured_contexts: list = []

        async def mock_extraction(
            frames_dir, output_dir, prompt_text, document,
            completion_params, context_manager, engine=None,
        ):
            idx = call_count["n"]
            call_count["n"] += 1

            # Write appropriate output
            data = pass0_data if idx == 0 else pass1_data
            with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                json.dump(data, f)

            if context_manager and context_manager.has_providers():
                captured_contexts.append(context_manager)

        ann._arun_single_extraction = mock_extraction

        doc = self._make_doc()
        await ann._arun_refine_passes(doc)

        # Pass 1 should have had a context manager with refinement provider
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        has_refine = any(
            isinstance(p, RefinementContextProvider) for p in ctx.providers
        )
        assert has_refine

    @pytest.mark.asyncio
    async def test_valid_refinement_replaces_pass0(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)

        pass0_data = {"extractions": [{"label": "a", "line_number": 1, "value": "old"}]}
        pass1_data = {
            "extractions": [
                {"label": "a", "line_number": 1, "value": "new"},
                {"label": "b", "line_number": 2, "value": "added"},
            ]
        }

        call_count = {"n": 0}

        async def mock_extraction(
            frames_dir, output_dir, prompt_text, document,
            completion_params, context_manager, engine=None,
        ):
            idx = call_count["n"]
            call_count["n"] += 1
            data = pass0_data if idx == 0 else pass1_data
            with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                json.dump(data, f)

        ann._arun_single_extraction = mock_extraction

        doc = self._make_doc()
        await ann._arun_refine_passes(doc)

        # Live dir should have pass 1's data
        live_json = os.path.join(ann.output_dir, "frame_0001.json")
        with open(live_json) as f:
            result = json.load(f)
        assert len(result["extractions"]) == 2
        assert result["extractions"][1]["label"] == "b"

    @pytest.mark.asyncio
    async def test_regressed_refinement_promotes_pass0(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)

        pass0_data = {
            "extractions": [
                {"label": "a", "line_number": 1, "value": "v1"},
                {"label": "b", "line_number": 2, "value": "v2"},
                {"label": "c", "line_number": 3, "value": "v3"},
                {"label": "d", "line_number": 4, "value": "v4"},
                {"label": "e", "line_number": 5, "value": "v5"},
            ]
        }
        # Severe regression: only 1 extraction
        pass1_data = {
            "extractions": [{"label": "a", "line_number": 1, "value": "v1"}]
        }

        call_count = {"n": 0}

        async def mock_extraction(
            frames_dir, output_dir, prompt_text, document,
            completion_params, context_manager, engine=None,
        ):
            idx = call_count["n"]
            call_count["n"] += 1
            data = pass0_data if idx == 0 else pass1_data
            with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                json.dump(data, f)

        ann._arun_single_extraction = mock_extraction

        doc = self._make_doc()
        await ann._arun_refine_passes(doc)

        # Should have promoted pass 0's data (5 extractions, not 1)
        live_json = os.path.join(ann.output_dir, "frame_0001.json")
        with open(live_json) as f:
            result = json.load(f)
        assert len(result["extractions"]) == 5

    @pytest.mark.asyncio
    async def test_exception_in_later_pass_promotes_last_good(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=2)

        pass0_data = {"extractions": [{"label": "a", "line_number": 1, "value": "v"}]}

        call_count = {"n": 0}

        async def mock_extraction(
            frames_dir, output_dir, prompt_text, document,
            completion_params, context_manager, engine=None,
        ):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx == 0:
                with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                    json.dump(pass0_data, f)
            elif idx == 1:
                # Pass 1 succeeds with same data
                with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                    json.dump(pass0_data, f)
            else:
                # Pass 2 raises
                raise RuntimeError("LLM failure")

        ann._arun_single_extraction = mock_extraction

        doc = self._make_doc()
        await ann._arun_refine_passes(doc)

        # Should still have valid output from pass 1
        live_json = os.path.join(ann.output_dir, "frame_0001.json")
        assert os.path.isfile(live_json)
        assert os.path.isfile(os.path.join(ann.output_dir, "_SUCCESS.yaml"))

    @pytest.mark.asyncio
    async def test_per_pass_model_routes_correctly(self, tmp_path):
        ann = _make_annotator(
            tmp_path,
            refine_passes=1,
            pass_models=["fast_model", "accurate_model"],
        )

        engines_used: list = []

        async def mock_extraction(
            frames_dir, output_dir, prompt_text, document,
            completion_params, context_manager, engine=None,
        ):
            engines_used.append(engine)
            data = {"extractions": [{"label": "a", "line_number": 1, "value": "v"}]}
            with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
                json.dump(data, f)

        ann._arun_single_extraction = mock_extraction

        engine_fast = MagicMock()
        engine_accurate = MagicMock()

        with patch(
            "marie.extract.annotators.llm_annotator.route_llm_engine"
        ) as mock_route:
            mock_route.side_effect = lambda name, mm: (
                engine_fast if name == "fast_model" else engine_accurate
            )

            doc = self._make_doc()
            await ann._arun_refine_passes(doc)

        assert len(engines_used) == 2
        assert engines_used[0] is engine_fast
        assert engines_used[1] is engine_accurate


# ======================================================================
# Config validation
# ======================================================================


class TestConfigValidation:
    def test_negative_refine_passes_raises(self, tmp_path):
        with pytest.raises(ValueError, match="refine_passes must be >= 0"):
            _make_annotator(tmp_path, refine_passes=-1)

    def test_short_pass_temperatures_raises(self, tmp_path):
        with pytest.raises(ValueError, match="pass_temperatures"):
            _make_annotator(tmp_path, refine_passes=2, pass_temperatures=[0.0])

    def test_short_pass_models_raises(self, tmp_path):
        with pytest.raises(ValueError, match="pass_models"):
            _make_annotator(tmp_path, refine_passes=2, pass_models=["m1"])


# ======================================================================
# _completion_params_for_pass
# ======================================================================


class TestCompletionParamsForPass:
    def test_default_temperature(self, tmp_path):
        ann = _make_annotator(tmp_path, refine_passes=1)
        params = ann._completion_params_for_pass(0)
        assert params["temperature"] == 0.0

    def test_per_pass_temperature_override(self, tmp_path):
        ann = _make_annotator(
            tmp_path, refine_passes=1, pass_temperatures=[0.0, 0.3]
        )
        assert ann._completion_params_for_pass(0)["temperature"] == 0.0
        assert ann._completion_params_for_pass(1)["temperature"] == 0.3

    def test_does_not_mutate_original(self, tmp_path):
        ann = _make_annotator(
            tmp_path, refine_passes=1, pass_temperatures=[0.0, 0.5]
        )
        params = ann._completion_params_for_pass(1)
        params["temperature"] = 999
        # Original should be untouched
        assert ann.completion_params["temperature"] == 0.0
