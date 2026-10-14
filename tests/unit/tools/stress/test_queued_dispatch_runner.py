import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_missing_owned_fixtures_reports_queued_dispatch_mode(tmp_path):
    from tools.stress.llm_queued_dispatch_aimock_e2e import qualify

    report = await qualify(
        SimpleNamespace(
            stores=str(tmp_path / 'missing-fixture.json'),
            output=tmp_path / 'output',
            items=16,
            seed=10,
        )
    )
    assert report['status'] == 'failed'
    assert report['mode'] == 'queued-dispatch'
    assert report['failures'][0]['phase'] == 'manifest'
    assert 'contract_version' not in report
    written = json.loads((tmp_path / 'output' / f"{report['run_id']}.json").read_text())
    assert written['mode'] == 'queued-dispatch'
