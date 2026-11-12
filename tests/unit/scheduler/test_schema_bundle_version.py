from types import SimpleNamespace

import pytest

from marie.excepts import RuntimeFailToStart
from marie.scheduler.repository.async_job_repository import AsyncJobRepository


async def test_stale_schema_bundle_is_rejected_before_database_writes(tmp_path, monkeypatch):
    directory = tmp_path / 'schema'
    directory.mkdir()
    (directory / '097_old_release.sql').write_text('SELECT 1;')
    (directory / 'monitoring').mkdir()
    (directory / 'monitoring/throughput_analysis.sql').write_text('SELECT 1;')
    (tmp_path / 'cron_job_init.sql').write_text('SELECT 1;')
    monkeypatch.setenv('MARIE_PSQL_DIR', str(tmp_path))
    monkeypatch.setenv('MARIE_SCHEMA_DIR', str(directory))
    repository = object.__new__(AsyncJobRepository)
    touched_database = False

    def acquire():
        nonlocal touched_database
        touched_database = True
        raise AssertionError('A stale bundle must not write schema or version rows')

    repository._pool = SimpleNamespace(acquire=acquire)
    repository.logger = SimpleNamespace(info=lambda *args: None)
    with pytest.raises(RuntimeFailToStart, match='complete SQL bundle'):
        await repository.create_tables()
    assert not touched_database
