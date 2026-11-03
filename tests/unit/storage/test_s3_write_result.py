import io
from types import SimpleNamespace

import pytest

from marie.storage.s3_storage import S3StorageHandler


@pytest.mark.parametrize('suppress_errors', [True, False])
def test_failed_s3_write_reports_failure(suppress_errors: bool) -> None:
    class UnavailableBucket:
        def put_object(self, **_kwargs) -> None:
            raise RuntimeError('storage unavailable')

    handler = object.__new__(S3StorageHandler)
    handler.s3 = SimpleNamespace(Bucket=lambda _name: UnavailableBucket())
    handler.suppress_errors = suppress_errors
    if suppress_errors:
        assert handler._write(io.BytesIO(b'{}'), 's3://bucket/result.json') is False
    else:
        with pytest.raises(RuntimeError, match='storage unavailable'):
            handler._write(io.BytesIO(b'{}'), 's3://bucket/result.json')
