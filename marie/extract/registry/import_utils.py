import importlib
import pkgutil
import sys
from typing import Iterable, List, Optional, Set, Tuple


def import_submodules(
    package_name: str,
    include_prefixes: Optional[Iterable[str]] = (
        "parsers",
        "validators",
        "template_builders",
        "processing_visitors",
    ),
    seen: Optional[Set[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Import `package_name` and selected submodules; return (loaded, failed)."""
    loaded: List[str] = []
    failed: List[Tuple[str, str]] = []

    if seen is None:
        seen = set(sys.modules.keys())

    try:
        root_pkg = importlib.import_module(package_name)
        loaded.append(package_name)
    except Exception as e:
        failed.append((package_name, f"root import failed: {e!r}"))
        return loaded, failed

    try:
        paths = root_pkg.__path__  # type: ignore[attr-defined]
    except AttributeError:
        return loaded, failed

    allowed: Optional[Set[str]] = None
    if include_prefixes:
        base = root_pkg.__name__
        allowed = {f"{base}.{p}" for p in include_prefixes}

    for _, modname, _ in pkgutil.walk_packages(paths, prefix=f"{root_pkg.__name__}."):
        if allowed is not None and not any(
            modname == a or modname.startswith(a + ".") for a in allowed
        ):
            continue
        if modname in seen:
            continue
        try:
            importlib.import_module(modname)
            loaded.append(modname)
            seen.add(modname)
        except Exception as e:
            failed.append((modname, repr(e)))

    return loaded, failed
