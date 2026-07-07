"""Blueprint manifest registry for the sandbox gateway.

Resolves a blueprint identifier to its parsed YAML manifest.  The lookup
order is:

1. ``MARIE_BLUEPRINTS_DIR`` environment variable (overrides everything).
2. Bundled ``builtin/`` directory shipped with this package.

Both locations are scanned for ``<blueprint_id>.yaml`` and ``<blueprint_id>.yml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from marie.logging_core.logger import MarieLogger

_logger = MarieLogger('marie.sandbox.blueprints.registry')

# Bundled blueprints shipped with the package (populated by devs; empty by default).
_BUILTIN_DIR = Path(__file__).parent / 'builtin'

_YAML_EXTENSIONS = ('.yaml', '.yml')


class BlueprintRegistry:
    """Resolve blueprint IDs to parsed manifest dicts.

    Args:
        blueprints_dir: Explicit directory to search.  ``None`` means read
            ``MARIE_BLUEPRINTS_DIR`` or fall back to the bundled ``builtin/``
            directory.
    """

    def __init__(self, blueprints_dir: str | None = None) -> None:
        self._dir: Path = (
            Path(blueprints_dir) if blueprints_dir else self._resolve_dir()
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def lookup(self, blueprint_id: str) -> dict[str, Any] | None:
        """Return the parsed manifest dict for *blueprint_id*, or ``None``.

        Args:
            blueprint_id: Blueprint identifier as sent by the Studio seam
                (e.g. ``'ner-vlm-ocr-entity-extraction'``).

        Returns:
            Parsed YAML dict, or ``None`` when the blueprint is not found.
        """
        for ext in _YAML_EXTENSIONS:
            path = self._dir / f'{blueprint_id}{ext}'
            if path.is_file():
                return self._load(path, blueprint_id)
        _logger.warning(
            f'Blueprint not found: {blueprint_id!r} '
            f'(searched {self._dir} with extensions {_YAML_EXTENSIONS})'
        )
        return None

    @staticmethod
    def _resolve_dir() -> Path:
        env = os.environ.get('MARIE_BLUEPRINTS_DIR', '').strip()
        if env:
            return Path(env)
        return _BUILTIN_DIR

    @staticmethod
    def _load(path: Path, blueprint_id: str) -> dict[str, Any] | None:
        try:
            import yaml  # pyyaml — already a transitive dependency

            with open(path, encoding='utf-8') as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                _logger.error(
                    f'Blueprint manifest at {path} is not a YAML mapping; skipping'
                )
                return None
            return data
        except Exception as exc:
            _logger.error(
                f'Failed to load blueprint {blueprint_id!r} from {path}: {exc}'
            )
            return None
