from pathlib import Path

import pytest
import yaml

from marie.extension.manifest import ExtensionPackage

FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_manifest_parses() -> None:
    data = yaml.safe_load(
        (FIXTURES / "minimal-tool" / "marie-extension.yaml").read_text()
    )

    manifest = ExtensionPackage.model_validate(data)

    assert manifest.kind == "ExtensionPackage"
    assert manifest.providers[0].tools[0].invocation_schema.required == ["text"]


def test_invalid_kind_rejected() -> None:
    data = yaml.safe_load(
        (FIXTURES / "minimal-tool" / "marie-extension.yaml").read_text()
    )
    data["kind"] = "Plugin"

    with pytest.raises(ValueError, match="kind must be ExtensionPackage"):
        ExtensionPackage.model_validate(data)
