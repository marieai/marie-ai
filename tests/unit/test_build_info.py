import json

from marie import build_info as build_info_module
from marie._version import __version__
from marie.build_info import format_build_identity, get_build_info, write_build_info
from marie.helper import get_full_version


def test_build_info_falls_back_for_missing_explicit_file(tmp_path) -> None:
    info = get_build_info(tmp_path / "missing.json")

    assert info == {
        "version": __version__,
        "git_commit": "unknown",
        "git_commit_short": "unknown",
        "build_time": "unknown",
        "build_number": "unknown",
        "image": "unknown",
        "image_digest": "unknown",
    }


def test_build_info_uses_git_for_source_checkout(tmp_path, monkeypatch) -> None:
    commit = "4b7f26d3d2927c7d4e61e147e9652fdb636f90ae"
    monkeypatch.setattr(
        build_info_module, "DEFAULT_BUILD_INFO_PATH", tmp_path / "system.json"
    )
    monkeypatch.setattr(
        build_info_module, "PACKAGE_BUILD_INFO_PATH", tmp_path / "package.json"
    )
    monkeypatch.setattr(build_info_module, "_source_git_commit", lambda: commit)

    info = get_build_info()

    assert info["version"] == __version__
    assert info["git_commit"] == commit
    assert info["git_commit_short"] == "4b7f26d3d292"
    assert info["build_number"] == "source"


def test_build_info_is_shared_by_startup_and_version_endpoint(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "build-info.json"
    commit = "4b7f26d3d2927c7d4e61e147e9652fdb636f90ae"
    write_build_info(
        path,
        version="5.0.3",
        git_commit=commit,
        build_time="2026-07-26T16:42:18Z",
        build_number="1842",
        image="marieai/marie-gateway:5.0.3-cpu",
    )
    monkeypatch.setenv("MARIE_BUILD_INFO_PATH", str(path))
    monkeypatch.setenv("MARIE_IMAGE_DIGEST", "sha256:bb2c")

    info = get_build_info()
    version_info, _ = get_full_version()

    assert json.loads(path.read_text(encoding="utf-8"))["git_commit"] == commit
    assert info["git_commit_short"] == "4b7f26d3d292"
    assert info["image_digest"] == "sha256:bb2c"
    assert version_info["git-commit"] == commit
    assert version_info["image-digest"] == "sha256:bb2c"
    assert format_build_identity("gateway") == (
        "Marie-AI build service=gateway version=5.0.3 "
        "commit=4b7f26d3d292 build=1842 "
        "built_at=2026-07-26T16:42:18Z "
        "image=marieai/marie-gateway:5.0.3-cpu "
        "image_digest=sha256:bb2c"
    )
