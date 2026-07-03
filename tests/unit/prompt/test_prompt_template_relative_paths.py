from pathlib import Path

import pytest

from marie.prompt.template import PromptLoadError, PromptTemplate


def test_prompt_fallback_preserves_relative_subpath(tmp_path):
    prompt_dir = tmp_path / "extract" / "TID-1" / "annotator"
    base_dir = tmp_path / "extract" / "base" / "annotators" / "example-package"
    base_dir.mkdir(parents=True)
    (base_dir / "extract.j2").write_text("hello {{ NAME }}", encoding="utf-8")

    template = PromptTemplate.from_file_with_fallback(
        "./annotators/example-package/extract.j2",
        prompt_dir=str(prompt_dir),
    )

    assert template.render({"NAME": "PACKAGE"}) == "hello PACKAGE"


def test_prompt_fallback_keeps_existing_dot_slash_prompt_compatibility(tmp_path):
    prompt_dir = tmp_path / "extract" / "TID-1" / "annotator"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "tables-refine.j2").write_text(
        "refine {{ VALUE }}", encoding="utf-8"
    )

    template = PromptTemplate.from_file_with_fallback(
        "./tables-refine.j2",
        prompt_dir=str(prompt_dir),
    )

    assert template.render({"VALUE": "ok"}) == "refine ok"


@pytest.mark.parametrize("prompt_path", ["/tmp/x.j2", "../x.j2", "a/../../x.j2"])
def test_prompt_fallback_rejects_unsafe_relative_subpath(tmp_path, prompt_path):
    prompt_dir = tmp_path / "extract" / "TID-1" / "annotator"

    with pytest.raises(PromptLoadError, match="Unsafe prompt path"):
        PromptTemplate.from_file_with_fallback(prompt_path, prompt_dir=str(prompt_dir))
