"""File validation utilities."""

import os
from pathlib import Path
from typing import Optional

from .constants import MB, SUPPORTED_FORMATS


class ValidationError(Exception):
    """Raised when file validation fails."""

    pass


def validate_file(
    file_path: str, max_size_mb: Optional[int] = None, check_format: bool = True
) -> None:
    """
    Validate that a file exists, is readable, and meets size/format requirements.

    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum file size in MB (default: None = no limit)
        check_format: Whether to check file format against supported formats

    Raises:
        ValidationError: If validation fails
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    # Check if it's a file (not directory)
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    # Check readability
    if not os.access(path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")

    # Check size
    file_size = path.stat().st_size
    if max_size_mb is not None:
        max_size_bytes = max_size_mb * MB
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File size ({file_size / MB:.2f}MB) exceeds maximum "
                f"allowed size ({max_size_mb}MB): {file_path}"
            )

    # Check format
    if check_format:
        file_ext = path.suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )


def validate_ref_id(ref_id: str) -> None:
    """
    Validate reference ID format.

    Args:
        ref_id: Reference ID to validate

    Raises:
        ValidationError: If ref_id is invalid
    """
    if not ref_id or not ref_id.strip():
        raise ValidationError("ref_id cannot be empty")

    if len(ref_id) > 255:
        raise ValidationError("ref_id cannot exceed 255 characters")


def validate_template_id(template_id: str) -> None:
    """
    Validate template ID format.

    Args:
        template_id: Template ID to validate

    Raises:
        ValidationError: If template_id is invalid
    """
    if not template_id or not template_id.strip():
        raise ValidationError("template_id cannot be empty")

    # Template IDs should be alphanumeric
    if not template_id.replace("_", "").replace("-", "").isalnum():
        raise ValidationError(
            "template_id must contain only alphanumeric characters, underscores, and hyphens"
        )
