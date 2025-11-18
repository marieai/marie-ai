"""Configuration management for Marie MCP Server."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for Marie MCP Server."""

    # Marie Gateway Connection
    MARIE_BASE_URL = os.getenv("MARIE_BASE_URL", "http://localhost:5000")
    MARIE_API_KEY = os.getenv("MARIE_API_KEY")  # REQUIRED

    # AWS S3 Configuration (for document uploads)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET", "marie")

    # Request Settings
    REQUEST_TIMEOUT = int(os.getenv("MARIE_REQUEST_TIMEOUT", "300"))  # 5 minutes
    MAX_FILE_SIZE_MB = int(os.getenv("MARIE_MAX_FILE_SIZE_MB", "50"))

    # Retry Settings
    MAX_RETRIES = int(os.getenv("MARIE_MAX_RETRIES", "3"))
    RETRY_MIN_WAIT = int(os.getenv("MARIE_RETRY_MIN_WAIT", "1"))  # seconds
    RETRY_MAX_WAIT = int(os.getenv("MARIE_RETRY_MAX_WAIT", "10"))  # seconds

    # Output Settings
    OUTPUT_DIR = Path(
        os.getenv("MARIE_OUTPUT_DIR", "~/.marie-mcp/outputs")
    ).expanduser()

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.MARIE_API_KEY:
            raise ValueError(
                "MARIE_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )

        if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
            print(
                "WARNING: AWS credentials not set. S3 uploads will fail. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment."
            )

        # Create output directory
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {cls.OUTPUT_DIR}")

    @classmethod
    def summary(cls) -> str:
        """Get configuration summary."""
        return f"""
Marie MCP Server Configuration:
  Gateway URL: {cls.MARIE_BASE_URL}
  API Key: {'*' * 10}{cls.MARIE_API_KEY[-4:] if cls.MARIE_API_KEY else 'NOT SET'}
  S3 Bucket: {cls.S3_BUCKET}
  AWS Region: {cls.AWS_REGION}
  Max File Size: {cls.MAX_FILE_SIZE_MB}MB
  Request Timeout: {cls.REQUEST_TIMEOUT}s
  Output Dir: {cls.OUTPUT_DIR}
"""
