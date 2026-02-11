"""
Data Sink sensor implementations.

Data Sink sensors monitor cloud storage services for new files
and trigger jobs to process them.

Supported providers:
- S3 (AWS S3 and S3-compatible: MinIO, DigitalOcean Spaces, Backblaze B2)

Future providers:
- GCS (Google Cloud Storage)
- Google Drive
- Dropbox
- Azure Blob Storage
"""

from marie.sensors.definitions.data_sink.base import (
    DataSinkProvider,
    DataSinkSensor,
    FileObject,
)
from marie.sensors.definitions.data_sink.s3_sensor import S3DataSinkSensor

__all__ = [
    "DataSinkProvider",
    "DataSinkSensor",
    "FileObject",
    "S3DataSinkSensor",
]
