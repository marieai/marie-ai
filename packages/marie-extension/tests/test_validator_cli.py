from pathlib import Path

from marie_extension.cli import main
from marie_extension.validator import validate_package

FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_package_success() -> None:
    result = validate_package(FIXTURES / "minimal-tool")

    assert result.ok is True
    assert result.package is not None


def test_validate_package_error() -> None:
    result = validate_package(FIXTURES / "invalid-traversal")

    assert result.ok is False
    assert "unsafe package path" in result.errors[0]


def test_cli_validate_success(capsys) -> None:
    code = main(["validate", "--path", str(FIXTURES / "minimal-tool")])
    captured = capsys.readouterr()

    assert code == 0
    assert "valid: ext.test.minimal-tool" in captured.out


def test_cli_validate_failure(capsys) -> None:
    code = main(["validate", "--path", str(FIXTURES / "invalid-traversal")])
    captured = capsys.readouterr()

    assert code == 1
    assert "unsafe package path" in captured.err
