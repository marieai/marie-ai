from unittest.mock import AsyncMock

import pytest

import marie.messaging.publisher as publisher


@pytest.mark.asyncio
async def test_mark_as_accepted_uses_accepted_status_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_status = AsyncMock(return_value=True)
    monkeypatch.setattr(publisher, '_mark_job_status', mark_status)

    result = await publisher.mark_as_accepted(
        api_key='project-1',
        job_id='job-1',
        event_name='extract',
        job_tag='invoice',
        status='OK',
        timestamp=123,
        payload={'ref_id': 'document-1'},
    )

    assert result is True
    mark_status.assert_awaited_once_with(
        'project-1',
        'job-1',
        'extract',
        'invoice',
        'OK',
        123,
        {'ref_id': 'document-1'},
        status_suffix='accepted',
        disabled_return_value=True,
    )
