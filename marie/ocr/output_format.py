from enum import Enum


class OutputFormat(Enum):
    """Output format for the document"""

    JSON = "json"  # Render document as JSON output
    PDF = "pdf"  # Render document as PDF
    TEXT = "text"  # Render document as plain TEXT
    ASSETS = "assets"  # Render and return all available assets

    @staticmethod
    def from_value(value: str):
        if value is None:
            return OutputFormat.JSON
        for data in OutputFormat:
            if data.value == value.lower():
                return data
        return OutputFormat.JSON
