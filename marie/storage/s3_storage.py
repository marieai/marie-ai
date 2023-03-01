import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

import io

from marie.excepts import BadConfigSource, raise_exception
from marie.storage import PathHandler

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

StrOrBytesPath = Union[str, Path, os.PathLike]

from marie.logging.predefined import default_logger as logger
from marie.logging.logger import MarieLogger


def is_file_like(obj) -> bool:
    """
    Check if the object is a file-like object.
    For objects to be considered file-like, they must
    be an iterator AND have either a `read` and/or `write`
    method as an attribute.
    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.
    Parameters
    ----------
    obj : The object to check
    Returns
    -------
    is_file_like : bool
        Whether `obj` has file-like properties.
    Examples
    --------
    >>> import io
    >>> buffer = io.StringIO("data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    return bool(hasattr(obj, "__iter__"))


class S3Url(object):
    """
    >>> s = S3Url("s3://bucket/hello/world")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world'
    >>> s.url
    's3://bucket/hello/world'

    >>> s = S3Url("s3://bucket/hello/world?qwe1=3#ddd")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world?qwe1=3#ddd'
    >>> s.url
    's3://bucket/hello/world?qwe1=3#ddd'

    >>> s = S3Url("s3://bucket/hello/world#foo?bar=2")
    >>> s.key
    'hello/world#foo?bar=2'
    >>> s.url
    's3://bucket/hello/world#foo?bar=2'
    """

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            return self._parsed.path.lstrip("/")

    @property
    def url(self):
        return self._parsed.geturl()


class S3StorageHandler(PathHandler):
    """
    S3 Storage Handler that handled s3:// paths.
    This implementation uses boto3 to access S3.

    Example configuration:

    .. code-block:: python

        handler = S3StorageHandler(
            config={
                "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
                "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
                "S3_STORAGE_BUCKET_NAME": "marie",
                "S3_ENDPOINT_URL": "http://localhost:8000",
                "S3_ADDRESSING_STYLE": "path",
            }
        )

        StorageManager.register_handler(handler)
        StorageManager.ensure_connection("s3://")

    Example configuration from yaml:

    .. code-block:: yaml

         storage:
          # S3 configuration. Will be used only if value of backend is "s3"
          s3:
            enabled: True
            metadata_only: False # If True, only metadata will be stored in the storage backend
            endpoint: ${{ ENV.S3_ENDPOINT_URL }}
            access_key_id: ${{ ENV.S3_ACCESS_KEY_ID }}
            secret_access_key: ${{ ENV.S3_SECRET_ACCESS_KEY }}
            bucket: ${{ ENV.S3_STORAGE_BUCKET_NAME }}
            region: ${{ ENV.S3_REGION }}
            insecure: True
            addressing_style: path


    .. code-block:: python

        handler = S3StorageHandler(config=storage_config["s3"], prefix="S3_")
        StorageManager.register_handler(handler=handler)
        StorageManager.ensure_connection("s3://")



    The following environment variables are used to configure the S3 client:

    - S3_REGION: The region to connect to. Defaults to "us-east-1".
    - S3_ACCESS_KEY_ID: The access key to use.
    - S3_SECRET_ACCESS_KEY: The secret access key to use.
    - S3_SESSION_TOKEN: The session token to use.
    - S3_STORAGE_BUCKET_NAME: The name of the bucket to use.
    - S3_ADDRESSING_STYLE: The addressing style to use. Defaults to "auto".
    - S3_ENDPOINT_URL: The endpoint URL to use.
    - S3_KEY_PREFIX: The key prefix to use.
    - S3_BUCKET_AUTH: Whether to use bucket authentication. Defaults to True.
    - S3_MAX_AGE_SECONDS: The max age in seconds. Defaults to 1 hour.
    - S3_PUBLIC_URL: The public URL to use.
    - S3_REDUCED_REDUNDANCY: Whether to use reduced redundancy. Defaults to False.
    - S3_CONTENT_DISPOSITION: The content disposition to use.
    - S3_CONTENT_ENCODING: The content encoding to use.
    - S3_CONTENT_LANGUAGE: The content language to use.
    - S3_CONTENT_TYPE: The content type to use.
    - S3_CONNECT_TIMEOUT: The connect timeout to use.
    - S3_READ_TIMEOUT: The read timeout to use.
    - S3_FILE_OVERWRITE: Whether to overwrite files. Defaults to True.

    Possible S3 providers (cloud or self-hosted):

    Self-hosted:
    - cloudserver: http://scality.github.io/S3/
    - mariadb: https://mariadb.com/kb/en/using-the-s3-storage-engine/
    - Minio: https://min.io/
    - Ceph: https://ceph.io/

    Cloud providers:
    - AWS: https://aws.amazon.com/s3/
    - DigitalOcean: https://www.digitalocean.com/products/spaces/
    - Wasabi: https://wasabi.com/
    - Minio: https://min.io/
    - Ceph: https://ceph.io/
    """

    PREFIX = "s3://"

    def __init__(self, config: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """
        Initialize the S3 storage handler.

        @param config:  The configuration to use.
        @param prefix: The prefix to use for all keys in config. If None, no prefix is used.
        """

        # Convert all keys to uppercase and add the prefix if needed.
        prefix = prefix or ""
        config = {f"{prefix + k}".upper(): v for k, v in config.items()}

        default_auth_settings = {
            "S3_REGION": "us-east-1",
            "S3_ACCESS_KEY_ID": "",
            "S3_SECRET_ACCESS_KEY": "",
            "S3_SESSION_TOKEN": "",
        }

        default_s3_settings = {
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ADDRESSING_STYLE": "auto",
            "S3_ENDPOINT_URL": "",
            "S3_KEY_PREFIX": "",
            "S3_BUCKET_AUTH": True,
            "S3_MAX_AGE_SECONDS": 60 * 60,  # 1 hours.
            "S3_PUBLIC_URL": "",
            "S3_REDUCED_REDUNDANCY": False,
            "S3_CONTENT_DISPOSITION": "",
            "S3_CONTENT_LANGUAGE": "",
            "S3_METADATA": {},
            "S3_ENCRYPT_KEY": False,
            "S3_KMS_ENCRYPTION_KEY_ID": "",
            "S3_GZIP": True,
            "S3_SIGNATURE_VERSION": "s3v4",
            "S3_FILE_OVERWRITE": False,
            "S3_USE_THREADS": True,
            "S3_MAX_POOL_CONNECTIONS": 10,
            "S3_CONNECT_TIMEOUT": 60,  # 60 seconds
        }

        self.config = {**default_auth_settings, **default_s3_settings, **config}

        # if key of config starts with "$S3_" throw an error
        for key in self.config:
            if key.startswith("$S3_"):
                raise BadConfigSource(
                    f"Invalid S3 config key: {key}. "
                    "Keys must not start with $S3_. Make sure that all environment variables are populated."
                )

        self.s3 = self.create_s3_resource(self.config)
        self.suppress_errors = True

    def ensure_connection(self):
        """Check if connection to s3 is possible by checking all buckets (NOOP)"""
        try:
            for bucket in self.s3.buckets.all():
                dt = bucket.creation_date
        # catching classes that do not inherit from BaseException is not allowed
        except Exception as e:
            raise ConnectionError(
                f'Could not connect to the endpoint URL: {self.config["S3_ENDPOINT_URL"]}'
            ) from e

    def create_s3_resource(self, config) -> Any:
        """Create a boto3 S3 resource from config."""

        # print config for debugging
        logger.info("S3 config:")
        for key, value in config.items():
            logger.info(f"{key}: {value}")

        s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=config["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=config["S3_SECRET_ACCESS_KEY"],
            aws_session_token=config["S3_SESSION_TOKEN"],
            region_name=config["S3_REGION"],
            endpoint_url=config["S3_ENDPOINT_URL"],
        )
        return s3_resource

    def _get_supported_prefixes(self) -> List[str]:
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        raise Exception("Operation not supported")

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """Check if the path exists in s3 storage"""
        self._check_kwargs(kwargs)
        s = S3Url(path)

        # should check local path first and then check s3
        # resolved_path = self._get_local_path(path)
        # return os.path.exists(resolved_path)

        try:
            path = s.key
            bucket = s.bucket
            # check if bucket exists and is accessible or if key exists in bucket
            if s.key == "":
                self.s3.meta.client.head_bucket(Bucket=bucket)
            else:
                self.s3.ObjectSummary(bucket_name=bucket, key=path).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                return raise_exception(
                    e, self.suppress_errors, logger, "Missing S3 bucket/key"
                )
        except EndpointConnectionError as e:
            return raise_exception(
                e, self.suppress_errors, logger, "S3 endpoint connection error"
            )

        return True

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        raise Exception("Operation not supported")

    def _copy(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> bool:
        self._check_kwargs(kwargs)
        raise Exception("Operation not supported")

    def _write(
        self,
        src_path: StrOrBytesPath,
        dst_path: str,
        overwrite: bool = False,
        handler: Optional["PathHandler"] = None,
        **kwargs: Any,
    ) -> bool:
        if os.path.isdir(src_path):
            raise Exception(f"Cannot upload directory {src_path} to s3")

        s = S3Url(dst_path)
        bucket = s.bucket
        key = s.key

        # uploading file to s3
        try:
            extra_args = {"ServerSideEncryption": "AES256"}
            # If S3 object_name was not specified, use file_name
            if key == "":
                key = os.path.basename(src_path)
            logger.debug(f"Uploading {src_path} to {bucket}/{key}")
            # Upload the file
            with open(src_path, "rb") as data:
                self.s3.Bucket(bucket).put_object(Key=key, Body=data, **extra_args)
        except Exception as e:
            logger.error(f"Unable to upload to bucket '{bucket}' : {e}")
            if not self.suppress_errors:
                raise e
        return True

    def _read(
        self,
        path: str,
        suppress_errors: bool = False,
        **kwargs: Any,
    ) -> bytes:
        s = S3Url(path)
        try:
            bytes_buffer = io.BytesIO()
            self.s3.Bucket(s.bucket).download_fileobj(s.key, bytes_buffer)

            return bytes_buffer.getvalue()
        except Exception as e:
            logger.error(f"Unable to download from bucket '{s.bucket}' : {e}")
            if not self.suppress_errors:
                raise e

    def _read_string(
        self,
        path: str,
        **kwargs: Any,
    ) -> str:
        byte_value = self._read(path, **kwargs)
        str_value = byte_value.decode()  # python3, default decoding is utf-8

        return str_value

    # download  file from s3 to local path
    def _read_to_file(
        self,
        path: str,
        local_src: str | os.PathLike | io.BytesIO,
        overwrite=False,
        **kwargs: Any,
    ):
        s = S3Url(path)
        bucket = self.s3.Bucket(s.bucket)

        file_like = False
        if is_file_like(local_src):
            file_like = True

        try:
            if not file_like:
                if overwrite:
                    if os.path.exists(local_src) and os.path.isfile(local_src):
                        os.remove(local_src)
                else:
                    if os.path.exists(local_src):
                        raise Exception(f"File {local_src} already exists")
                # make sure the directory exists
                os.makedirs(os.path.dirname(local_src), exist_ok=True)

            if file_like:
                bucket.download_fileobj(s.key, local_src)
            else:
                with open(local_src, "wb") as data:
                    bucket.download_fileobj(s.key, data)

        except Exception as e:
            logger.error(f"Unable to write file from bucket '{s.bucket}' : {e}")
            if not self.suppress_errors:
                raise e

    def _list(self, path: str, return_full_path=False, **kwargs: Any) -> List[str]:
        """List all files in current bucket in s3 storage"""
        items = []
        try:
            s = S3Url(path)
            bucket = self.s3.Bucket(s.bucket)
            # list all objects in bucket
            for objects in bucket.objects.filter(Prefix=s.key):
                if return_full_path:
                    items.append(f"s3://{s.bucket}/{objects.key}")
                else:
                    items.append(objects.key)
        except Exception as e:
            logger.error(f"Bucket '{s.bucket}' does not exist")
            if not self.suppress_errors:
                raise e
        return items

    def _mkdir(self, path: str, **kwargs: Any) -> None:
        """
        mkdir at the given URI.
        """
        # Buckets cannot be nested in S3, so we can't create a directory
        # structure. We can create a file with a trailing slash, but that
        # doesn't work with all S3 clients.

        s = S3Url(path)
        if s.key:
            raise ValueError(f"Cannot create a directory in S3 only buckets: {s}")

        try:
            self.s3.create_bucket(Bucket=s.bucket)
        except Exception as e:
            logger.warning(f"Bucket '{s.bucket}' already exists")
            if not self.suppress_errors:
                raise e
