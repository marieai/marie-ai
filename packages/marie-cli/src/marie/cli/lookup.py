"""Public CLI access to Marie's command lookup service."""

from marie.parsers.lookup import _build_lookup_table as _build_lookup_table
from marie.parsers.lookup import lookup_and_print as lookup_and_print

__all__ = ["_build_lookup_table", "lookup_and_print"]
