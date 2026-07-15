"""Sample module for code extraction tests."""

import json
from pathlib import Path

DEFAULT_CURRENCY = "USD"


class InvoiceParser:
    """Parse invoice documents into line items."""

    def __init__(self, currency: str = "USD") -> None:
        self.currency = currency

    def parse(self, path: Path) -> dict:
        """Read one invoice file and return its line items."""
        return json.loads(path.read_text())


def total_amount(items: list[dict]) -> float:
    """Sum the amount field across line items."""
    return sum(item["amount"] for item in items)
