import subprocess
import sys

from marie.cli import main
from marie.cli.autocomplete import ac_table


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
