from dataclasses import dataclass, field
from pathlib import Path

from marie.extension.errors import ExtensionError
from marie.extension.loader import LoadedPackage, load_package


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    package: LoadedPackage | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)


def validate_package(path: str | Path) -> ValidationResult:
    try:
        return ValidationResult(ok=True, package=load_package(path))
    except ExtensionError as exc:
        return ValidationResult(ok=False, errors=(str(exc),))
