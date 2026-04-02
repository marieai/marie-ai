"""Unit tests for PromptTemplate."""

import logging
import os
import textwrap

import pytest

from marie.prompt.errors import PromptLoadError, PromptRenderError, PromptTemplateError
from marie.prompt.template import PromptTemplate

# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestCreation:
    def test_from_string(self):
        tmpl = PromptTemplate("Hello {{ name }}")
        assert tmpl.source == "Hello {{ name }}"
        assert tmpl.file_path is None

    def test_from_string_preserves_whitespace(self):
        raw = "  leading\n  trailing  "
        tmpl = PromptTemplate(raw)
        assert tmpl.source == raw

    def test_empty_string_allowed(self):
        tmpl = PromptTemplate("")
        assert tmpl.source == ""

    def test_none_source_raises(self):
        with pytest.raises(PromptLoadError):
            PromptTemplate(None)

    def test_from_file(self, tmp_path):
        p = tmp_path / "test.j2"
        p.write_text("  Hello {{ name }}  \n")
        tmpl = PromptTemplate.from_file(str(p))
        assert tmpl.source == "Hello {{ name }}"
        assert tmpl.file_path == str(p)

    def test_from_file_strips_whitespace(self, tmp_path):
        p = tmp_path / "test.j2"
        p.write_text("\n\n  content here  \n\n")
        tmpl = PromptTemplate.from_file(str(p))
        assert tmpl.source == "content here"

    def test_from_file_not_found(self):
        with pytest.raises(PromptLoadError, match="not found"):
            PromptTemplate.from_file("/nonexistent/path.j2")

    def test_name_property(self):
        tmpl = PromptTemplate("hello", name="test-prompt")
        assert tmpl.name == "test-prompt"


# ---------------------------------------------------------------------------
# Jinja2 rendering
# ---------------------------------------------------------------------------


class TestJinja2Rendering:
    def test_simple_substitution(self):
        tmpl = PromptTemplate("Hello {{ name }}, you are {{ age }}.")
        result = tmpl.render({"name": "Alice", "age": "30"})
        assert result == "Hello Alice, you are 30."

    def test_missing_jinja2_var_renders_empty(self):
        tmpl = PromptTemplate("Hello {{ name }}!")
        result = tmpl.render({})
        assert result == "Hello !"

    def test_conditional_syntax(self):
        tmpl = PromptTemplate("{% if show %}visible{% endif %}")
        assert tmpl.render({"show": "yes"}) == "visible"
        assert tmpl.render({}) == ""


# ---------------------------------------------------------------------------
# Bare variable rendering
# ---------------------------------------------------------------------------


class TestBareVarRendering:
    def test_simple_bare_var(self):
        tmpl = PromptTemplate("Data: OCR_DATA end")
        result = tmpl.render({"OCR_DATA": "hello world"})
        assert result == "Data: hello world end"

    def test_overlapping_bare_vars(self):
        """FILTERED_OCR_DATA must be replaced independently of OCR_DATA."""
        tmpl = PromptTemplate("filtered=FILTERED_OCR_DATA full=OCR_DATA")
        result = tmpl.render({
            "FILTERED_OCR_DATA": "filtered_content",
            "OCR_DATA": "full_content",
        })
        assert result == "filtered=filtered_content full=full_content"

    def test_overlapping_bare_vars_order(self):
        """Longest-key-first prevents OCR_DATA from corrupting FILTERED_OCR_DATA."""
        tmpl = PromptTemplate("FILTERED_OCR_DATA and OCR_DATA")
        result = tmpl.render({
            "FILTERED_OCR_DATA": "[filtered]",
            "OCR_DATA": "[full]",
        })
        assert result == "[filtered] and [full]"


# ---------------------------------------------------------------------------
# Dual syntax
# ---------------------------------------------------------------------------


class TestDualSyntax:
    def test_jinja2_and_bare_in_same_template(self):
        tmpl = PromptTemplate("{{ greeting }} OCR_DATA end")
        result = tmpl.render({"greeting": "Hello", "OCR_DATA": "data here"})
        assert result == "Hello data here end"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_used_when_var_missing(self):
        tmpl = PromptTemplate(
            "{{ greeting }} {{ name }}",
            defaults={"greeting": "Hi", "name": "World"},
        )
        result = tmpl.render({})
        assert result == "Hi World"

    def test_supplied_vars_override_defaults(self):
        tmpl = PromptTemplate(
            "{{ greeting }}",
            defaults={"greeting": "Hi"},
        )
        result = tmpl.render({"greeting": "Hello"})
        assert result == "Hello"

    def test_defaults_for_bare_vars(self):
        tmpl = PromptTemplate(
            "OCR_DATA here",
            defaults={"OCR_DATA": "default_data"},
        )
        result = tmpl.render({})
        assert result == "default_data here"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class TestFunctions:
    def test_computed_variable(self):
        tmpl = PromptTemplate(
            "{{ computed }}",
            functions={"computed": lambda vars: vars.get("a", "") + "_suffix"},
        )
        result = tmpl.render({"a": "value"})
        assert result == "value_suffix"

    def test_function_conflict_raises(self):
        tmpl = PromptTemplate(
            "{{ x }}",
            functions={"x": lambda vars: "computed"},
        )
        with pytest.raises(PromptRenderError, match="conflicts"):
            tmpl.render({"x": "supplied"})


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


class TestFork:
    def test_fork_independent_copy(self):
        original = PromptTemplate("{{ x }}", defaults={"x": "original"})
        forked = original.fork(defaults={"x": "forked"})
        assert original.render() == "original"
        assert forked.render() == "forked"

    def test_fork_merges_defaults(self):
        original = PromptTemplate("{{ a }} {{ b }}", defaults={"a": "1"})
        forked = original.fork(defaults={"b": "2"})
        assert forked.render() == "1 2"

    def test_fork_new_source(self):
        original = PromptTemplate("old")
        forked = original.fork(source="new")
        assert forked.source == "new"
        assert original.source == "old"

    def test_fork_preserves_file_path(self):
        original = PromptTemplate("x", file_path="/some/path.j2")
        forked = original.fork(defaults={"y": "z"})
        assert forked.file_path == "/some/path.j2"

    def test_fork_new_name(self):
        original = PromptTemplate("x", name="orig")
        forked = original.fork(name="forked")
        assert forked.name == "forked"
        assert original.name == "orig"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_jinja2_variables(self):
        tmpl = PromptTemplate("{{ a }} {{ b }} text")
        assert tmpl.jinja2_variables == frozenset({"a", "b"})

    def test_bare_variables(self):
        tmpl = PromptTemplate("OCR_DATA and FILTERED_OCR_DATA here")
        assert "OCR_DATA" in tmpl.bare_variables
        assert "FILTERED_OCR_DATA" in tmpl.bare_variables

    def test_expected_variables_union(self):
        tmpl = PromptTemplate("{{ j2_var }} BARE_VAR")
        assert tmpl.expected_variables == frozenset({"j2_var", "BARE_VAR"})

    def test_ignore_list_filters_common_tokens(self):
        tmpl = PromptTemplate("Output JSON format with HTML tags")
        assert "JSON" not in tmpl.bare_variables
        assert "HTML" not in tmpl.bare_variables

    def test_known_pipeline_variables_detected(self):
        known_vars = [
            "OCR_DATA",
            "OCR_TEXT",
            "FILTERED_OCR_DATA",
            "INJECTED_TEXT",
            "PREVIOUS_EXTRACTION",
            "TABLE_CONTEXT_ALL",
            "TABLE_COUNT",
            "HAS_TABLES",
            "TABLE_HEADER",
            "TABLE_ROWS",
            "CLAIM_CONTEXT",
            "PAGE_NUMBER",
            "DOCUMENT_TYPE",
            "SCHEMA_DEFINITION",
            "FIELD_DEFINITIONS",
            "EXTRACTION_RULES",
        ]
        source = " ".join(known_vars)
        tmpl = PromptTemplate(source)
        for var in known_vars:
            assert var in tmpl.bare_variables, f"{var} not detected as bare variable"


# ---------------------------------------------------------------------------
# Warning on unresolved
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_unresolved_bare_var_warns(self, caplog):
        tmpl = PromptTemplate("OCR_DATA here")
        with caplog.at_level(logging.WARNING, logger="marie.prompt.template"):
            tmpl.render({})
        assert any("OCR_DATA" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# from_file_with_fallback
# ---------------------------------------------------------------------------


class TestFromFileWithFallback:
    def _setup_dirs(self, tmp_path, tid="123"):
        """Create a typical config directory structure."""
        base_dir = tmp_path / "config" / "extract" / "base"
        tid_dir = tmp_path / "config" / "extract" / f"TID-{tid}" / "annotator"
        base_dir.mkdir(parents=True)
        tid_dir.mkdir(parents=True)
        return base_dir, tid_dir

    def test_tid_specific_found(self, tmp_path):
        base_dir, tid_dir = self._setup_dirs(tmp_path)
        (tid_dir / "test.j2").write_text("TID content")
        tmpl = PromptTemplate.from_file_with_fallback(
            "test.j2",
            config_dir=str(tmp_path / "config"),
            layout_id="123",
        )
        assert tmpl.source == "TID content"

    def test_fallback_to_base(self, tmp_path):
        base_dir, tid_dir = self._setup_dirs(tmp_path)
        (base_dir / "test.j2").write_text("Base content")
        tmpl = PromptTemplate.from_file_with_fallback(
            "test.j2",
            config_dir=str(tmp_path / "config"),
            layout_id="123",
        )
        assert tmpl.source == "Base content"

    def test_neither_found_raises(self, tmp_path):
        self._setup_dirs(tmp_path)
        with pytest.raises(PromptLoadError, match="not found"):
            PromptTemplate.from_file_with_fallback(
                "missing.j2",
                config_dir=str(tmp_path / "config"),
                layout_id="123",
            )

    def test_prompt_dir_mode(self, tmp_path):
        # Simulate: prompt_dir = .../TID-X/annotator/
        tid_dir = tmp_path / "extract" / "TID-99" / "annotator"
        tid_dir.mkdir(parents=True)
        (tid_dir / "prompt.j2").write_text("prompt_dir content")

        tmpl = PromptTemplate.from_file_with_fallback(
            "prompt.j2",
            prompt_dir=str(tid_dir),
        )
        assert tmpl.source == "prompt_dir content"

    def test_prompt_dir_fallback_to_base(self, tmp_path):
        # prompt_dir = .../TID-X/annotator/ -> go up 2 levels to find base/
        tid_dir = tmp_path / "extract" / "TID-99" / "annotator"
        base_dir = tmp_path / "extract" / "base"
        tid_dir.mkdir(parents=True)
        base_dir.mkdir(parents=True)
        (base_dir / "prompt.j2").write_text("base fallback")

        tmpl = PromptTemplate.from_file_with_fallback(
            "prompt.j2",
            prompt_dir=str(tid_dir),
        )
        assert tmpl.source == "base fallback"

    def test_path_traversal_sanitized(self, tmp_path):
        base_dir, tid_dir = self._setup_dirs(tmp_path)
        # Even with traversal attempt, should look for "passwd" not "../../etc/passwd"
        with pytest.raises(PromptLoadError, match="not found"):
            PromptTemplate.from_file_with_fallback(
                "../../etc/passwd",
                config_dir=str(tmp_path / "config"),
                layout_id="123",
            )

    def test_path_traversal_basename_only(self):
        assert PromptTemplate._sanitize_filename("../../etc/passwd") == "passwd"
        assert PromptTemplate._sanitize_filename("../evil.j2") == "evil.j2"
        assert PromptTemplate._sanitize_filename("safe.j2") == "safe.j2"

    def test_empty_filename_raises(self):
        with pytest.raises(PromptLoadError, match="empty"):
            PromptTemplate.from_file_with_fallback(
                "",
                config_dir="/tmp",
                layout_id="1",
            )

    def test_none_filename_raises(self):
        with pytest.raises(PromptLoadError, match="empty"):
            PromptTemplate.from_file_with_fallback(
                None,
                config_dir="/tmp",
                layout_id="1",
            )

    def test_tried_paths_in_error(self, tmp_path):
        self._setup_dirs(tmp_path)
        with pytest.raises(PromptLoadError) as exc_info:
            PromptTemplate.from_file_with_fallback(
                "missing.j2",
                config_dir=str(tmp_path / "config"),
                layout_id="123",
            )
        error_msg = str(exc_info.value)
        assert "TID-123" in error_msg
        assert "base" in error_msg

    def test_defaults_passed_through(self, tmp_path):
        base_dir, _ = self._setup_dirs(tmp_path)
        (base_dir / "test.j2").write_text("{{ greeting }}")
        tmpl = PromptTemplate.from_file_with_fallback(
            "test.j2",
            config_dir=str(tmp_path / "config"),
            layout_id="123",
            defaults={"greeting": "Hello"},
        )
        assert tmpl.render() == "Hello"


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_render_error_is_template_error(self):
        assert issubclass(PromptRenderError, PromptTemplateError)

    def test_load_error_is_template_error(self):
        assert issubclass(PromptLoadError, PromptTemplateError)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_includes_name(self):
        tmpl = PromptTemplate("x", name="test")
        r = repr(tmpl)
        assert "test" in r

    def test_repr_includes_file_path(self):
        tmpl = PromptTemplate("x", file_path="/a/b.j2")
        r = repr(tmpl)
        assert "/a/b.j2" in r
