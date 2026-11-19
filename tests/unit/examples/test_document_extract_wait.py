import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples import document_extract


class EventChannel:
    def __init__(self, events: Iterable[bytes] = ()) -> None:
        self.events = list(events)
        self.bound = False
        self.cancelled = False

    def exchange_declare(self, **kwargs: Any) -> None:
        self.exchange = kwargs

    def queue_declare(self, **kwargs: Any) -> SimpleNamespace:
        assert kwargs == {'queue': '', 'exclusive': True, 'auto_delete': True}
        return SimpleNamespace(method=SimpleNamespace(queue='private-queue'))

    def queue_bind(self, **kwargs: Any) -> None:
        assert kwargs['queue'] == 'private-queue'
        assert kwargs['routing_key'] == 'extract.#'
        self.bound = True

    def consume(
        self, queue: str, **kwargs: Any
    ) -> Iterator[tuple[object | None, None, bytes | None]]:
        assert queue == 'private-queue'
        assert kwargs['auto_ack'] is True
        for event in self.events:
            yield object(), None, event
        while True:
            yield None, None, None

    def cancel(self) -> None:
        self.cancelled = True


def event(job_id: str = 'job-1', name: str = 'extract.completed') -> bytes:
    return json.dumps(
        {
            'jobid': job_id,
            'event': name,
            'payload': json.dumps({'ref_id': 'scan.tif', 'ref_type': 'extract'}),
        }
    ).encode()


def test_wait_matches_the_submitted_job_and_ignores_other_events() -> None:
    channel = EventChannel(
        [
            b'invalid json',
            b'[]',
            event('other-job'),
            event(name='extract.started'),
            event(),
        ]
    )
    result = document_extract.wait_for_completion(channel, 'private-queue', 'job-1', 10)
    assert result['jobid'] == 'job-1'
    assert result['event'] == 'extract.completed'
    assert channel.cancelled


def test_wait_reports_failure_for_the_matching_job() -> None:
    channel = EventChannel(
        [event('other-job', 'extract.failed'), event(name='extract.failed')]
    )
    with pytest.raises(RuntimeError, match='job-1.*failed'):
        document_extract.wait_for_completion(channel, 'private-queue', 'job-1', 10)
    assert channel.cancelled


def test_wait_has_a_deadline_and_cancels_the_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(
        document_extract, 'time', SimpleNamespace(monotonic=lambda: next(ticks))
    )
    channel = EventChannel()
    with pytest.raises(TimeoutError, match='job-1'):
        document_extract.wait_for_completion(channel, 'private-queue', 'job-1', 1)
    assert channel.cancelled


@pytest.fixture
def storage_files(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, dict[str, Any], dict[str, bytes]]:
    from marie.storage import StorageManager

    monkeypatch.setenv('MARIE_S3_BUCKET', 'test-bucket')
    monkeypatch.setattr(document_extract, 'configure_storage', lambda _config: None)
    root = 's3://test-bucket/extract/scan/'
    metadata = {'pages': [{'text': 'OCR result'}], 'assets': [root + 'pdf/scan.pdf']}
    files = {
        root + 'scan.tif.meta.json': json.dumps(metadata).encode(),
        root + 'pdf/scan.pdf': b'%PDF-test',
    }

    def read_to_file(uri: str, destination: str, **_kwargs: Any) -> bool:
        if uri not in files:
            return False
        Path(destination).write_bytes(files[uri])
        return True

    monkeypatch.setattr(StorageManager, 'read_to_file', read_to_file)
    return root, metadata, files


def test_download_saves_the_actual_metadata_and_listed_assets(
    storage_files: tuple, tmp_path: Path
) -> None:
    root, metadata, files = storage_files
    metadata['assets'].insert(0, root + 'scan.tif.meta.json')
    files[root + 'scan.tif.meta.json'] = json.dumps(metadata, indent=4).encode() + b'\n'
    result = document_extract.download_results(json.loads(event()), tmp_path)
    assert Path(result['metadata_file']) == tmp_path / 'scan.tif.meta.json'
    assert (
        Path(result['metadata_file']).read_bytes() == files[root + 'scan.tif.meta.json']
    )
    assert json.loads(Path(result['metadata_file']).read_text()) == metadata
    assert (Path(result['assets_dir']) / 'pdf/scan.pdf').read_bytes() == b'%PDF-test'
    assert result['asset_count'] == 1


@pytest.mark.parametrize('missing', ['scan.tif.meta.json', 'pdf/scan.pdf'])
def test_failed_download_does_not_report_success(
    storage_files: tuple, tmp_path: Path, missing: str
) -> None:
    root, _, files = storage_files
    del files[root + missing]
    with pytest.raises(RuntimeError, match='download'):
        document_extract.download_results(json.loads(event()), tmp_path)


def test_asset_paths_cannot_overwrite_files_outside_the_asset_directory(
    storage_files: tuple, tmp_path: Path
) -> None:
    root, metadata, files = storage_files
    metadata['assets'] = [root + '../scan.tif.meta.json']
    files[root + 'scan.tif.meta.json'] = json.dumps(metadata).encode()
    files[root + '../scan.tif.meta.json'] = b'overwrite'
    with pytest.raises(ValueError, match='asset'):
        document_extract.download_results(json.loads(event()), tmp_path)
    assert json.loads((tmp_path / 'scan.tif.meta.json').read_text()) == metadata


@pytest.mark.parametrize('payload', [None, {}, '[]', '{"ref_id": "scan.tif"}'])
def test_invalid_completion_reference_is_a_clear_error(
    payload: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match='document reference'):
        document_extract.download_results({'payload': payload}, tmp_path)


def test_local_input_wait_registers_storage_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from collections import OrderedDict

    from marie import storage
    from marie.storage import PathHandler, StorageManager

    class TestHandler(PathHandler):
        def _get_supported_prefixes(self) -> list[str]:
            return ['s3://']

    monkeypatch.setattr(StorageManager, '_PATH_HANDLERS', OrderedDict())
    monkeypatch.setattr(storage, 'S3StorageHandler', lambda **_kwargs: TestHandler())
    monkeypatch.setattr(StorageManager, 'write', lambda *_args, **_kwargs: True)
    monkeypatch.setenv('MARIE_S3_BUCKET', 'before-test')
    monkeypatch.setenv('S3_STORAGE_BUCKET_NAME', 'test-bucket')
    monkeypatch.setattr(
        document_extract, 'submit_and_wait', lambda *_args: json.loads(event())
    )

    def read_to_file(_uri: str, destination: str, **_kwargs: Any) -> bool:
        Path(destination).write_text(json.dumps({'pages': [], 'assets': []}))
        return True

    monkeypatch.setattr(StorageManager, 'read_to_file', read_to_file)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'api_key': 'test-auth', 'storage': {}}))
    source = tmp_path / 'scan.tif'
    source.write_bytes(b'input')

    assert (
        document_extract.main(
            [
                '--input',
                str(source),
                '--config',
                str(config),
                '--wait',
                '--output-dir',
                str(tmp_path / 'output'),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


@pytest.mark.parametrize(
    'outcome', ['completed', 'submit-failed', 'job-failed', 'timeout', 'interrupted']
)
def test_cli_subscribes_before_submission_and_closes_the_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    storage_files: tuple,
    outcome: str,
) -> None:
    for key in ('MARIE_API_BASE_URL', 'MARIE_API_KEY', 'MARIE_PROJECT_ID'):
        monkeypatch.delenv(key, raising=False)
    channel = EventChannel()

    class Connection:
        closed = False

        def __init__(self, parameters: Any) -> None:
            assert parameters.host == 'broker.example'
            assert parameters.virtual_host == '/'

        def __enter__(self) -> 'Connection':
            return self

        def __exit__(self, *_args: Any) -> None:
            Connection.closed = True

        def channel(self) -> EventChannel:
            return channel

    def submit(**_kwargs: Any) -> dict[str, str]:
        assert channel.bound
        if outcome == 'submit-failed':
            raise RuntimeError('Submission rejected')
        if outcome == 'interrupted':
            raise KeyboardInterrupt
        # A fast worker can finish before the submission response arrives.
        if outcome == 'completed':
            channel.events.append(event())
        elif outcome == 'job-failed':
            channel.events.append(event(name='extract.failed'))
        return {'status': 'ok', 'job_id': 'job-1'}

    if outcome == 'timeout':
        ticks = iter([0, 0, 3])
        monkeypatch.setattr(
            document_extract, 'time', SimpleNamespace(monotonic=lambda: next(ticks))
        )
    monkeypatch.setattr(document_extract.pika, 'BlockingConnection', Connection)
    monkeypatch.setattr(document_extract, 'process_extract', submit)
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            {
                'api_key': 'test-auth',
                'storage': {},
                'queue': {
                    'hostname': 'broker.example',
                    'username': 'test',
                    'password': 'test',
                },
            }
        )
    )
    output = tmp_path / 'result'
    code = document_extract.main(
        [
            '--config',
            str(config),
            '--uri',
            's3://input/scan.tif',
            '--wait',
            '--wait-timeout',
            '2',
            '--output-dir',
            str(output),
        ]
    )

    assert Connection.closed
    captured = capsys.readouterr()
    assert 'test-auth' not in captured.out + captured.err
    if outcome != 'completed':
        assert code == (130 if outcome == 'interrupted' else 1)
        expected = {
            'submit-failed': 'Submission rejected',
            'job-failed': 'Job job-1 failed',
            'timeout': 'Timed out waiting for job job-1',
            'interrupted': 'Interrupted',
        }
        assert expected[outcome] in captured.err
        assert not output.exists()
    else:
        assert code == 0
        result = json.loads(captured.out)
        assert result['status'] == 'completed'
        assert result['job_id'] == 'job-1'
        assert json.loads(Path(result['metadata_file']).read_text())['pages']
        assert 'job-1' in captured.err
