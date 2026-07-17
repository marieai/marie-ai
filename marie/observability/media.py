"""Resolve Marie request media into durable telemetry references."""

from typing import Any

from marie.utils.asset_util import s3_asset_path


def resolve_media_reference(context: Any) -> tuple[str, str]:
    """Return the source asset URL and resolution mode for a request context."""
    return (
        s3_asset_path(
            ref_id=str(context.ref_id),
            ref_type=str(context.ref_type),
            include_filename=True,
        ),
        "s3_asset_path",
    )
