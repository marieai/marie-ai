"""Synchronous S3 helpers; async callers must run them in a worker thread."""

import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ..config import Config
from .validators import ValidationError, validate_ref_id


def s3_asset_path(ref_id: str, ref_type: str, include_filename: bool = True) -> str:
    """Generate paths using the backend asset convention."""
    validate_ref_id(ref_id)
    normalized_type = ref_type.replace('/', '_').lower()
    if (
        not normalized_type
        or normalized_type in {'.', '..'}
        or any(char in normalized_type for char in ('\\', '\x00'))
    ):
        raise ValidationError('Invalid ref_type')
    filename = os.path.basename(ref_id)
    prefix = os.path.splitext(filename)[0].lower()
    if not prefix or prefix in {'.', '..'}:
        raise ValidationError('Invalid ref_id filename')
    root = f's3://{Config.S3_BUCKET}/{normalized_type}/{prefix}'
    return f'{root}/{filename}' if include_filename else root


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    if not s3_path.startswith('s3://'):
        raise ValueError('S3 path must start with s3://')
    parts = s3_path[5:].split('/', 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError('S3 path must contain a bucket and key')
    return parts[0], parts[1]


def upload_to_s3(local_path: str, s3_path: str, overwrite: bool = True) -> bool:
    """Upload a file; propagate storage failures, return False if overwrite is disabled."""
    bucket, key = _parse_s3_path(s3_path)
    if not overwrite and s3_exists(s3_path):
        return False
    _get_s3_client().upload_file(local_path, bucket, key)
    return True


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download atomically so failures leave an existing result intact."""
    bucket, key = _parse_s3_path(s3_path)
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(
        dir=destination.parent, prefix='.marie-mcp-'
    )
    os.close(handle)
    try:
        _get_s3_client().download_file(bucket, key, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return True


def s3_exists(s3_path: str) -> bool:
    """Return False only when the requested object is absent."""
    bucket, key = _parse_s3_path(s3_path)
    try:
        _get_s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get('Error', {}).get('Code') in {
            '404',
            'NoSuchKey',
            'NotFound',
        }:
            return False
        raise
    return True


def _get_s3_client() -> Any:
    return boto3.client(
        's3',
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        region_name=Config.AWS_REGION,
    )
