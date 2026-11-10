from pathlib import Path

import pytest

from marie.storage import StorageManager


@pytest.mark.parametrize('write_result', [True, False, 'raises'])
def test_directory_copy_reports_failed_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_result
) -> None:
    (tmp_path / 'one.json').write_bytes(b'{}')
    (tmp_path / 'two.json').write_bytes(b'[]')

    def write(_source, _destination):
        if write_result == 'raises':
            raise OSError('upload unavailable')
        return write_result

    monkeypatch.setattr(StorageManager, 'write', write)
    result = StorageManager.copy_dir(
        str(tmp_path), 's3://bucket/outputs', relative_to_dir=str(tmp_path)
    )

    assert result == ('s3://bucket/outputs' if write_result is True else None)
