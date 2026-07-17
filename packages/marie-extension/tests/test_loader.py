import re
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from marie.extension import PackageLoadError, load_package

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_directory_package() -> None:
    package = load_package(FIXTURES / "minimal-tool")

    assert package.manifest.metadata.id == "ext.test.minimal-tool"
    assert package.manifest.providers[0].tools[0].name == "echo"
    assert package.digest.startswith("sha256:")


def test_load_model_provider_metadata() -> None:
    package = load_package(FIXTURES / "minimal-model")
    provider = package.manifest.providers[0]

    assert provider.type == "model_provider"
    assert provider.models[0].model_type == "llm"
    assert provider.models[0].features == ["tool-call", "stream-tool-call"]


def test_reject_declared_path_traversal() -> None:
    with pytest.raises(PackageLoadError, match="unsafe package path"):
        load_package(FIXTURES / "invalid-traversal")


def test_load_zip_package(tmp_path: Path) -> None:
    archive = tmp_path / "extension.zip"
    with ZipFile(archive, "w") as zf:
        zf.write(
            FIXTURES / "minimal-tool" / "marie-extension.yaml",
            "marie-extension.yaml",
        )

    package = load_package(archive)

    assert package.manifest.metadata.name == "minimal-tool"
    assert package.manifest_path == "marie-extension.yaml"


def test_reject_zip_without_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "missing.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("README.md", "missing manifest")

    with pytest.raises(
        PackageLoadError, match=re.escape("missing marie-extension.yaml")
    ):
        load_package(archive)


def test_reject_zip_with_multiple_manifests(tmp_path: Path) -> None:
    archive = tmp_path / "multiple.zip"
    manifest = (FIXTURES / "minimal-tool" / "marie-extension.yaml").read_text()
    with ZipFile(archive, "w") as zf:
        zf.writestr("a/marie-extension.yaml", manifest)
        zf.writestr("b/marie-extension.yaml", manifest)

    with pytest.raises(
        PackageLoadError, match=re.escape("multiple marie-extension.yaml")
    ):
        load_package(archive)


def test_reject_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "traversal.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("../marie-extension.yaml", "bad")

    with pytest.raises(PackageLoadError, match="unsafe package path"):
        load_package(archive)


def test_reject_zip_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = ZipInfo("marie-extension.yaml")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with ZipFile(archive, "w") as zf:
        zf.writestr(info, "target")

    with pytest.raises(PackageLoadError, match="archive symlink"):
        load_package(archive)
