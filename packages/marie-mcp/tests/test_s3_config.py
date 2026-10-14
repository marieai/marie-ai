import pytest
from botocore.exceptions import ClientError
from marie.mcp.server.config import Config
from marie.mcp.server.utils import s3_utils


@pytest.mark.parametrize(
    'ref_id,ref_type,expected',
    [
        ('Invoice.PDF', 'Invoice', 'invoice/invoice/Invoice.PDF'),
        ('Doc_1', 'Medical/Form', 'medical_form/doc_1/Doc_1'),
    ],
)
def test_paths_match_backend(ref_id, ref_type, expected):
    assert (
        s3_utils.s3_asset_path(ref_id, ref_type)
        == f's3://{Config.S3_BUCKET}/{expected}'
    )


@pytest.mark.parametrize(
    'code,exists', [('404', False), ('NoSuchKey', False), ('403', None), ('500', None)]
)
def test_s3_missing_distinguished_from_failure(monkeypatch, code, exists):
    class S3:
        def head_object(self, **kwargs):
            raise ClientError({'Error': {'Code': code}}, 'HeadObject')

    monkeypatch.setattr(s3_utils, '_get_s3_client', S3)
    if exists is None:
        with pytest.raises(ClientError):
            s3_utils.s3_exists('s3://bucket/key')
    else:
        assert s3_utils.s3_exists('s3://bucket/key') is exists


def test_config_validation_never_writes_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(Config, 'AWS_ACCESS_KEY_ID', None)
    monkeypatch.setattr(Config, 'AWS_SECRET_ACCESS_KEY', None)
    Config.validate()
    assert not capsys.readouterr().out


@pytest.mark.parametrize('fail', [False, True])
def test_atomic_download_preserves_previous_result_on_failure(
    monkeypatch, tmp_path, capsys, fail
):
    destination = tmp_path / 'result.json'
    destination.write_text('previous')

    class S3:
        def download_file(self, bucket, key, path):
            from pathlib import Path

            Path(path).write_text('new')
            if fail:
                raise RuntimeError('storage unavailable')

    monkeypatch.setattr(s3_utils, '_get_s3_client', S3)
    if fail:
        with pytest.raises(RuntimeError, match='storage unavailable'):
            s3_utils.download_from_s3('s3://bucket/key', str(destination))
    else:
        assert s3_utils.download_from_s3('s3://bucket/key', str(destination))
    assert destination.read_text() == ('previous' if fail else 'new')
    assert list(tmp_path.iterdir()) == [destination]
    assert not capsys.readouterr().out


def test_upload_never_writes_stdout(monkeypatch, tmp_path, capsys):
    class S3:
        def upload_file(self, *args):
            pass

    monkeypatch.setattr(s3_utils, '_get_s3_client', S3)
    assert s3_utils.upload_to_s3(str(tmp_path / 'input.pdf'), 's3://bucket/key')
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    'ref_id', ['../secret', 'folder/file.pdf', '..', '', 'file\\name.pdf']
)
def test_invalid_reference_paths_rejected(ref_id):
    from marie.mcp.server.utils.validators import ValidationError

    with pytest.raises(ValidationError):
        s3_utils.s3_asset_path(ref_id, 'invoice')
