import subprocess
import sys

from marie.cli import _get_run_args, main
from marie.cli.autocomplete import ac_table

from marie.build_info import write_build_info


def test_cli_public_entrypoint() -> None:
    assert callable(main)
    assert "gateway" in ac_table["commands"]


def test_python_module_entrypoint_displays_gateway_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "marie", "gateway", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Start a Gateway" in result.stdout


def test_cli_startup_displays_build_identity(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "build-info.json"
    write_build_info(
        path,
        version="5.0.4",
        git_commit="4b7f26d3d2927c7d4e61e147e9652fdb636f90ae",
        build_time="2026-07-26T16:42:18Z",
        build_number="1842",
        image="marieai/marie-gateway:5.0.4-cpu",
    )
    monkeypatch.setenv("MARIE_BUILD_INFO_PATH", str(path))
    monkeypatch.setattr(sys, "argv", ["marie", "gateway"])

    args = _get_run_args()

    assert args.cli == "gateway"
    assert (
        "Marie-AI build service=gateway version=5.0.4 commit=4b7f26d3d292 build=1842"
    ) in capsys.readouterr().out
