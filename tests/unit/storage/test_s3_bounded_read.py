import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marie.storage.s3_storage import S3StorageHandler


@pytest.mark.parametrize(
    'payload,advertised,limit,oversized',
    [
        (b'', 0, 0, False),
        (b'abc', 3, 3, False),
        (b'abc', 3, 2, True),
        (b'abc', 0, 2, True),
    ],
)
def test_bounded_read_closes_body_and_enforces_size(
    payload, advertised, limit, oversized
):
    handler = object.__new__(S3StorageHandler)
    body = io.BytesIO(payload)
    client = SimpleNamespace(
        get_object=Mock(return_value={'Body': body, 'ContentLength': advertised})
    )
    handler.s3 = SimpleNamespace(meta=SimpleNamespace(client=client))
    if oversized:
        with pytest.raises(ValueError, match='artifact size'):
            handler._read('s3://fixture/output.md', max_bytes=limit)
    else:
        assert handler._read('s3://fixture/output.md', max_bytes=limit) == payload
    assert body.closed
    client.get_object.assert_called_once_with(Bucket='fixture', Key='output.md')
