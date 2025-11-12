"""S3 utilities for document upload and download."""

import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from ..config import Config


def s3_asset_path(ref_id: str, ref_type: str, include_filename: bool = True) -> str:
    """
    Generate S3 path following Marie's convention.

    Pattern: s3://{bucket}/{ref_type}/{ref_id}/{filename}

    Args:
        ref_id: Document reference ID
        ref_type: Document type/category
        include_filename: Whether to include filename in path

    Returns:
        S3 URI path
    """
    bucket = Config.S3_BUCKET
    base_path = f"s3://{bucket}/{ref_type}/{ref_id}"

    if include_filename:
        # Extract filename from ref_id if it looks like a filename
        filename = os.path.basename(ref_id)
        return f"{base_path}/{filename}"

    return base_path


def upload_to_s3(local_path: str, s3_path: str, overwrite: bool = True) -> bool:
    """
    Upload file to S3.

    Args:
        local_path: Local file path
        s3_path: S3 URI (s3://bucket/key)
        overwrite: Whether to overwrite existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse S3 URI
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path (must start with s3://): {s3_path}")

        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        if not key:
            raise ValueError(f"Invalid S3 path (missing key): {s3_path}")

        # Check if file exists
        if not overwrite:
            s3_client = _get_s3_client()
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                print(f"File already exists at {s3_path} and overwrite=False")
                return False
            except ClientError as e:
                # File doesn't exist, proceed with upload
                if e.response["Error"]["Code"] != "404":
                    raise

        # Upload using boto3
        s3_client = _get_s3_client()
        s3_client.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to {s3_path}")
        return True

    except ClientError as e:
        print(f"S3 upload error: {e}")
        return False
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download file from S3.

    Args:
        s3_path: S3 URI (s3://bucket/key)
        local_path: Local destination path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse S3 URI
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path (must start with s3://): {s3_path}")

        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        if not key:
            raise ValueError(f"Invalid S3 path (missing key): {s3_path}")

        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Download using boto3
        s3_client = _get_s3_client()
        s3_client.download_file(bucket, key, local_path)
        print(f"Downloaded {s3_path} to {local_path}")
        return True

    except ClientError as e:
        print(f"S3 download error: {e}")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def s3_exists(s3_path: str) -> bool:
    """
    Check if file exists in S3.

    Args:
        s3_path: S3 URI (s3://bucket/key)

    Returns:
        True if file exists, False otherwise
    """
    try:
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        s3_client = _get_s3_client()
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
    except Exception as e:
        print(f"Error checking S3 file: {e}")
        return False


def _get_s3_client():
    """Get configured S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        region_name=Config.AWS_REGION,
    )
