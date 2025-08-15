import logging
from typing import Any, Callable, Mapping


class EnsureFieldsFilter(logging.Filter):
    """
    Ensures specific attributes exist on LogRecord. If missing, sets defaults.
    Value can be a constant or a zero-arg callable (lazy evaluation).
    """

    def __init__(self, defaults: Mapping[str, Any]):
        super().__init__()
        self._defaults = dict(defaults)

    def filter(self, record: logging.LogRecord) -> bool:
        for key, val in self._defaults.items():
            if not hasattr(record, key):
                setattr(record, key, val() if callable(val) else val)
        return True
