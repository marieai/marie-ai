import importlib.util
import textwrap
from pathlib import Path

import pytest

_UTIL_PATH = Path(__file__).parents[4] / "marie" / "executor" / "extract" / "util.py"
_SPEC = importlib.util.spec_from_file_location("extract_util_under_test", _UTIL_PATH)
_UTIL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_UTIL)

load_layout_config = _UTIL.load_layout_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_layout_config_merges_includes_before_tid_override(tmp_path):
    base = tmp_path / "base"
    layout = tmp_path / "TID-1" / "annotator"

    _write(base / "field-config.yml", "grounding: {key-value: []}\nannotators: {}\n")
    _write(
        base / "base-config.yml",
        """
        annotators:
          base-only:
            enabled: true
        grounding:
          key-value:
            - CLAIM NUMBER
        """,
    )
    _write(
        base / "annotators" / "example-package" / "config.yml",
        """
        annotators:
          package-labels:
            enabled: false
          package-labels-aggregated:
            enabled: false
        grounding:
          key-value:
            - PACKAGE FIELD
        """,
    )
    _write(
        layout / "config.yml",
        """
        includes:
          - ./annotators/example-package/config.yml
        annotators:
          package-labels:
            enabled: true
        grounding:
          key-value:
            - PATIENT NAME
        """,
    )

    conf = load_layout_config(str(base), str(layout))

    assert conf.annotators["base-only"].enabled is True
    assert conf.annotators["package-labels"].enabled is True
    assert conf.annotators["package-labels-aggregated"].enabled is False
    assert conf.grounding["key-value"] == [
        "CLAIM_NUMBER",
        "PACKAGE_FIELD",
        "PATIENT_NAME",
    ]


@pytest.mark.parametrize(
    "include_path", ["/tmp/config.yml", "../config.yml", "x/../../config.yml"]
)
def test_layout_config_rejects_unsafe_include_paths(tmp_path, include_path):
    base = tmp_path / "base"
    layout = tmp_path / "TID-1" / "annotator"
    _write(base / "field-config.yml", "grounding: {key-value: []}\n")
    _write(base / "base-config.yml", "annotators: {}\n")
    _write(layout / "config.yml", f"includes:\n  - {include_path}\n")

    with pytest.raises(ValueError, match="Unsafe include path"):
        load_layout_config(str(base), str(layout))
