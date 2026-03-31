"""Core PromptTemplate class with Jinja2 rendering and bare-var fallback."""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Dict, FrozenSet, Optional

from jinja2 import BaseLoader, Environment, Undefined, meta

from marie.prompt.errors import PromptLoadError, PromptRenderError

logger = logging.getLogger(__name__)

# Bare-variable pattern: uppercase identifiers with at least 2 chars
_BARE_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\b")

# Tokens that look like bare vars but aren't prompt placeholders
_BARE_VAR_IGNORE = frozenset(
    {
        "JSON",
        "HTML",
        "XML",
        "CSV",
        "UTF",
        "ASCII",
        "HTTP",
        "HTTPS",
        "URL",
        "URI",
        "API",
        "LLM",
        "OCR",
        "PDF",
        "PNG",
        "JPEG",
        "TIFF",
        "SQL",
        "ID",
        "OK",
        "NULL",
        "TRUE",
        "FALSE",
        "NONE",
        "NAN",
        "NA",
        "IF",
        "ELSE",
        "AND",
        "OR",
        "NOT",
        "FOR",
        "IN",
        "IS",
        "THE",
        "OF",
        "TO",
        "DO",
        "NO",
        "ON",
        "AT",
        "BY",
        "AS",
        "AN",
        "SO",
        "BE",
        "IT",
        "WE",
        "HE",
        "UP",
        "ALL",
        "ANY",
        "GET",
        "SET",
        "RUN",
        "END",
        "NEW",
        "OUT",
        "USE",
        "TAG",
        "KEY",
        "ROW",
        "MAP",
        "LOG",
    }
)


class _SilentUndefined(Undefined):
    """Jinja2 Undefined that returns empty string instead of raising."""

    def __str__(self) -> str:
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self) -> bool:
        return False


_ENV = Environment(
    loader=BaseLoader(),
    undefined=_SilentUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


class PromptTemplate:
    """Prompt template with Jinja2 rendering, bare-var fallback, and introspection.

    Rendering pipeline:
      1. Merge defaults -> supplied variables -> function outputs
      2. Jinja2 render ``{{ var }}`` expressions
      3. ``str.replace()`` fallback for remaining bare ``VAR`` placeholders
         (longest-key-first to prevent substring corruption)
      4. Log warnings for expected variables that remain unresolved
    """

    __slots__ = (
        "_source",
        "_file_path",
        "_defaults",
        "_functions",
        "_name",
        "_jinja2_vars",
        "_bare_vars",
        "_compiled",
    )

    def __init__(
        self,
        source: str,
        *,
        file_path: Optional[str] = None,
        defaults: Optional[Dict[str, str]] = None,
        functions: Optional[Dict[str, Callable[[Dict[str, str]], str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        if not source and source != "":
            raise PromptLoadError("Template source must not be None")
        self._source = source
        self._file_path = file_path
        self._defaults = dict(defaults) if defaults else {}
        self._functions = dict(functions) if functions else {}
        self._name = name

        # Parse Jinja2 AST once
        try:
            ast = _ENV.parse(source)
            self._jinja2_vars: FrozenSet[str] = frozenset(
                meta.find_undeclared_variables(ast)
            )
            self._compiled = _ENV.from_string(source)
        except Exception as exc:
            raise PromptRenderError(f"Failed to parse Jinja2 template: {exc}") from exc

        # Detect bare variables
        self._bare_vars: FrozenSet[str] = frozenset(
            m
            for m in _BARE_VAR_RE.findall(source)
            if m not in _BARE_VAR_IGNORE and m not in self._jinja2_vars
        )

    # -- Properties ----------------------------------------------------------

    @property
    def source(self) -> str:
        return self._source

    @property
    def file_path(self) -> Optional[str]:
        return self._file_path

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def defaults(self) -> Dict[str, str]:
        return dict(self._defaults)

    @property
    def jinja2_variables(self) -> FrozenSet[str]:
        return self._jinja2_vars

    @property
    def bare_variables(self) -> FrozenSet[str]:
        return self._bare_vars

    @property
    def expected_variables(self) -> FrozenSet[str]:
        return self._jinja2_vars | self._bare_vars

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def from_str(
        cls,
        source: str,
        *,
        defaults: Optional[Dict[str, str]] = None,
        functions: Optional[Dict[str, Callable[[Dict[str, str]], str]]] = None,
        name: Optional[str] = None,
    ) -> PromptTemplate:
        """Create a template from a raw string (no file, no stripping)."""
        return cls(source, defaults=defaults, functions=functions, name=name)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        *,
        defaults: Optional[Dict[str, str]] = None,
        functions: Optional[Dict[str, Callable[[Dict[str, str]], str]]] = None,
        name: Optional[str] = None,
    ) -> PromptTemplate:
        """Load a template from disk, stripping leading/trailing whitespace."""
        try:
            with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
                source = f.read().strip()
        except FileNotFoundError as exc:
            raise PromptLoadError(f"Prompt file not found: {file_path}") from exc
        except OSError as exc:
            raise PromptLoadError(
                f"Failed to read prompt file {file_path}: {exc}"
            ) from exc
        return cls(
            source,
            file_path=file_path,
            defaults=defaults,
            functions=functions,
            name=name,
        )

    @classmethod
    def from_file_with_fallback(
        cls,
        prompt_filename: str,
        *,
        prompt_dir: Optional[str] = None,
        layout_id: Optional[str] = None,
        config_dir: Optional[str] = None,
        defaults: Optional[Dict[str, str]] = None,
        functions: Optional[Dict[str, Callable]] = None,
        name: Optional[str] = None,
    ) -> PromptTemplate:
        """Load a template with TID-specific -> base fallback resolution.

        Path 1 (prompt_dir provided):
          1. {prompt_dir}/{filename}
          2. {dirname(dirname(prompt_dir))}/base/{filename}

        Path 2 (no prompt_dir, uses config_dir + layout_id):
          1. {config_dir}/extract/TID-{layout_id}/annotator/{filename}
          2. {config_dir}/extract/base/{filename}

        The filename is sanitized via ``os.path.basename()`` to prevent path
        traversal — callers can pass the raw config value directly.
        """
        safe_name = cls._sanitize_filename(prompt_filename)
        if not safe_name:
            raise PromptLoadError("prompt_filename is empty or None")

        candidates: list[str] = []

        if prompt_dir:
            # Path 1: prompt_dir mode
            candidates.append(os.path.join(prompt_dir, safe_name))
            # Go up 2 levels from prompt_dir to find base/
            base_dir = os.path.join(
                os.path.dirname(os.path.dirname(prompt_dir)), "base"
            )
            candidates.append(os.path.join(base_dir, safe_name))
        elif config_dir and layout_id is not None:
            # Path 2: production mode
            candidates.append(
                os.path.join(
                    config_dir,
                    "extract",
                    f"TID-{layout_id}",
                    "annotator",
                    safe_name,
                )
            )
            candidates.append(os.path.join(config_dir, "extract", "base", safe_name))
        else:
            raise PromptLoadError(
                "Either prompt_dir or (config_dir + layout_id) must be provided"
            )

        for path in candidates:
            if os.path.exists(path):
                logger.info("Loading prompt from: %s", path)
                return cls.from_file(
                    path, defaults=defaults, functions=functions, name=name
                )

        raise PromptLoadError(
            f"Prompt file '{safe_name}' not found. Tried:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    # -- Rendering -----------------------------------------------------------

    def render(self, variables: Optional[Dict[str, str]] = None) -> str:
        """Render the template with the given variables.

        Pipeline:
          1. Merge defaults -> supplied variables -> function outputs
          2. Jinja2 render
          3. Bare-var str.replace fallback (longest-key-first)
          4. Warn about unresolved expected variables
        """
        merged = dict(self._defaults)
        if variables:
            merged.update(variables)

        # Compute function outputs
        for fn_name, fn in self._functions.items():
            if fn_name in merged:
                raise PromptRenderError(
                    f"Function '{fn_name}' conflicts with supplied variable"
                )
            try:
                merged[fn_name] = fn(merged)
            except Exception as exc:
                raise PromptRenderError(f"Function '{fn_name}' raised: {exc}") from exc

        # Step 1: Jinja2 render
        try:
            text = self._compiled.render(**merged)
        except Exception as exc:
            raise PromptRenderError(f"Jinja2 rendering failed: {exc}") from exc

        # Step 2: Bare-var fallback
        text = self._render_bare(text, merged)

        # Step 3: Warn about unresolved variables
        self._warn_unresolved(text, merged)

        return text

    @staticmethod
    def _render_bare(text: str, variables: Dict[str, str]) -> str:
        """Replace remaining bare VAR placeholders, longest-key-first.

        Only processes keys that look like bare variables (uppercase identifiers
        with at least 2 characters) to avoid corrupting text with short keys.
        """
        bare_keys = [k for k in variables if _BARE_VAR_RE.fullmatch(k)]
        for var_name in sorted(bare_keys, key=len, reverse=True):
            text = text.replace(var_name, variables[var_name])
        return text

    def _warn_unresolved(self, text: str, supplied: Dict[str, str]) -> None:
        """Log warnings for expected variables that are still in the rendered text."""
        for var in self._bare_vars:
            if var not in supplied and var in text:
                logger.warning(
                    "Template%s: bare variable '%s' was not resolved",
                    f" '{self._name}'" if self._name else "",
                    var,
                )

    # -- Derivation ----------------------------------------------------------

    def fork(
        self,
        *,
        source: Optional[str] = None,
        defaults: Optional[Dict[str, str]] = None,
        functions: Optional[Dict[str, Callable[[Dict[str, str]], str]]] = None,
        name: Optional[str] = None,
    ) -> PromptTemplate:
        """Return a new PromptTemplate with merged overrides."""
        new_defaults = dict(self._defaults)
        if defaults:
            new_defaults.update(defaults)

        new_functions = dict(self._functions)
        if functions:
            new_functions.update(functions)

        return PromptTemplate(
            source=source if source is not None else self._source,
            file_path=self._file_path,
            defaults=new_defaults,
            functions=new_functions,
            name=name if name is not None else self._name,
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _sanitize_filename(filename: str) -> Optional[str]:
        """Strip directory components to prevent path traversal."""
        return os.path.basename(filename) if filename else None

    def __repr__(self) -> str:
        parts = [f"PromptTemplate(name={self._name!r}"]
        if self._file_path:
            parts.append(f"file={self._file_path!r}")
        parts.append(f"vars={sorted(self.expected_variables)}")
        return ", ".join(parts) + ")"
