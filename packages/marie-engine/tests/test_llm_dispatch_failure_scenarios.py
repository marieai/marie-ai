"""Strict fixed-cell reporting and real HTTP failure boundaries."""

import asyncio
import copy
import importlib.util
import json

import pytest


def runner():
    name = 'tools.stress.llm_dispatch_failure_scenarios'
    assert importlib.util.find_spec(name) is not None, 'fixed failure runner is missing'
    return __import__(name, fromlist=['*'])


def valid_report():
    h = runner()
    rows = []
    for family in ('redis', 'valkey'):
        for case in h.CASES:
            row = dict(
                name=f'{family}-{case}',
                family=family,
                case=case,
                status='passed',
                fabric=f'{family}-{case}',
                provider_acceptances=9 if case == 'S03' else 2,
                writes=9 if case == 'S03' else 0,
                attempts=9 if case == 'S03' else 2,
                sibling_completions=0 if case == 'S03' else 1,
            )
            if case == 'S03':
                row.update(
                    before_outage=dict(
                        writes=1, provider_acceptances=1, first_output_sha256='a' * 64
                    ),
                    during_outage=dict(pending=8, caller_pending=True, circuit='open'),
                    after_recovery=dict(
                        writes=9, provider_acceptances=9, first_output_sha256='a' * 64
                    ),
                    same_producer=True,
                    same_deadlines=True,
                    ordered_unique_outputs=True,
                )
            else:
                expected = h.EXPECTED[case]
                row.update(
                    category=expected[0],
                    remote_settled=expected[1],
                    fully_consumed=expected[2],
                    transmitted=True,
                    reserved_before_close=int(not expected[1]),
                    reserved_after_close=int(not expected[1]),
                    execution_seq=1,
                    retries=0,
                    receipt_before_release=True,
                    application_json_rejected=case == 'S13-assistant-non-json',
                    state_before_close=(
                        'outcome_unknown'
                        if not expected[1]
                        else 'succeeded'
                        if expected[0] == 'success'
                        else 'failed'
                    ),
                    same_producer=True,
                    same_deadlines=True,
                    aimock_acceptances_by_item={'fault': 1, 'sibling': 1},
                )
            rows.append(row)
    return dict(
        status='passed',
        failures=[],
        selected=list(h.CASES),
        scenarios=rows,
        cleanup=[dict(fabric=r['fabric'], remaining_keys=0) for r in rows],
        verified_stores={f: {'verified': True} for f in ('redis', 'valkey')},
        aimock={'verified': True},
        source={'backend': {'digest': 'a' * 64}, 'studio': {'digest': 'b' * 64}},
    )


def test_complete_exact_fixed_suite_is_accepted():
    runner().accept_report(valid_report())


@pytest.mark.parametrize(
    'damage',
    [
        'missing',
        'duplicate',
        'unknown',
        'bool-counter',
        'wrong-category',
        'unsafe-settlement',
        'resend',
        'missing-cleanup',
        'remaining-key',
        'first-rewritten',
        'early-outage',
        'caller-finished',
        'missing-receipt',
        'failure',
        'wrong-state',
        'foreign-owner',
        'changed-deadline',
        'per-item-resend',
    ],
)
def test_incomplete_or_false_pass_is_rejected(damage):
    h = runner()
    report = copy.deepcopy(valid_report())
    row = report['scenarios'][1]
    if damage == 'missing':
        report['scenarios'].pop()
    elif damage == 'duplicate':
        report['scenarios'].append(row.copy())
    elif damage == 'unknown':
        row['name'] = 'redis-invented'
    elif damage == 'bool-counter':
        row['provider_acceptances'] = True
    elif damage == 'wrong-category':
        row['category'] = 'invalid_response'
    elif damage == 'unsafe-settlement':
        row['reserved_after_close'] = 0
    elif damage == 'resend':
        row['provider_acceptances'] = 3
    elif damage == 'missing-cleanup':
        report['cleanup'].pop()
    elif damage == 'remaining-key':
        report['cleanup'][0]['remaining_keys'] = 1
    elif damage == 'first-rewritten':
        report['scenarios'][0]['after_recovery']['first_output_sha256'] = 'b' * 64
    elif damage == 'early-outage':
        report['scenarios'][0]['before_outage']['writes'] = 0
    elif damage == 'caller-finished':
        report['scenarios'][0]['during_outage']['caller_pending'] = False
    elif damage == 'missing-receipt':
        row['receipt_before_release'] = False
    elif damage == 'failure':
        report['failures'].append({'phase': 'cleanup'})
    elif damage == 'wrong-state':
        row['state_before_close'] = 'succeeded'
    elif damage == 'foreign-owner':
        row['same_producer'] = False
    elif damage == 'changed-deadline':
        row['same_deadlines'] = False
    elif damage == 'per-item-resend':
        row['aimock_acceptances_by_item'] = {'fault': 2, 'sibling': 0}
    with pytest.raises(h.QualificationError):
        h.accept_report(report)


@pytest.mark.parametrize(
    'case',
    [
        'S12-5xx',
        'S12-read-timeout',
        'S12-interrupted-body',
        'S13-malformed-http-json',
        'S13-authentication',
        'S13-invalid-request',
        'S13-assistant-non-json',
    ],
)
async def test_fixed_wire_outcome_after_observed_receipt(case):
    h = runner()
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    receipts = []

    async def forward(body):
        receipts.append(body['model'])
        return {'choices': [{'message': {'content': 'not JSON'}}]}

    seam = h.OwnedFailureListener(case, forward)
    await seam.start()
    client = EndpointClient(
        RegisteredEndpoint(
            'fixed', seam.url, allow_loopback=True, call_timeout_seconds=0.15
        )
    )
    task = asyncio.create_task(
        client.execute(
            CompletionCallParams(model='fault', messages=[]), timeout_seconds=2
        )
    )
    try:
        await asyncio.wait_for(seam.received.wait(), 2)
        assert receipts == ['fault']
        assert not task.done()
        seam.release.set()
        if case == 'S12-interrupted-body':
            await asyncio.wait_for(seam.body_started.wait(), 2)
            seam.interrupt.set()
        outcome = await task
        category, settled, consumed = h.EXPECTED[case]
        assert (outcome.category or 'success', outcome.remote_settled) == (
            category,
            settled,
        )
        assert outcome.retryable is False
        assert seam.records[0]['receipt_before_release'] is True
    finally:
        await client.close()
        await seam.close()


def test_failed_report_serialization_is_safe(tmp_path):
    h = runner()
    report = valid_report() | {'run_id': 'safe', 'unserializable': object()}
    h.finalize_report(report, tmp_path)
    assert report['status'] == 'failed'
    artifact = json.loads((tmp_path / 'safe.json').read_text())
    assert artifact['status'] == 'failed'
    assert artifact['failures'][-1]['phase'] == 'report_write'


async def test_setup_error_is_failed_artifact(tmp_path):
    h = runner()
    from types import SimpleNamespace

    report = await h.qualify(
        SimpleNamespace(
            output=tmp_path, stores=tmp_path / 'absent', scenarios=list(h.CASES)
        )
    )
    assert report['status'] == 'failed'
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert report['failures'][0]['phase'] == 'manifest'


async def test_full_fixed_suite_against_owned_aimock_and_stores(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    manifest = os.environ.get('MARIE_LLM_QUEUE_TEST_STORES')
    if not manifest:
        pytest.skip('Set MARIE_LLM_QUEUE_TEST_STORES to owned qualification fixtures')
    fixture = json.loads(Path(manifest).read_text())['redis']
    ports = fixture['compose_ports']
    container = subprocess.check_output(
        [
            'docker',
            'compose',
            '-p',
            fixture['compose_project'],
            '-f',
            fixture['compose_file'],
            'ps',
            '-q',
            'aimock',
        ],
        env=os.environ | ports,
        text=True,
    ).strip()
    args = SimpleNamespace(
        stores=manifest,
        studio=os.environ.get(
            'MARIE_LLM_FAILURE_STUDIO',
            str(Path(__file__).resolve().parents[4] / 'marie-studio-llm-dispatch'),
        ),
        mock='http://127.0.0.1:' + ports['QUAL_MOCK_PORT'],
        admin='http://127.0.0.1:' + ports['QUAL_ADMIN_PORT'],
        aimock_container=container,
        scenarios=list(runner().CASES),
        output=tmp_path,
    )
    report = await runner().qualify(args)
    assert report['status'] == 'passed', report['failures']
    assert len(report['scenarios']) == 16
    assert sum(r['provider_acceptances'] for r in report['scenarios']) == 46


async def test_setup_failure_closes_started_owned_listener(tmp_path, monkeypatch):
    from types import SimpleNamespace

    h = runner()
    listeners = []
    original_start = h.OwnedFailureListener.start

    async def start(self):
        await original_start(self)
        listeners.append(self)

    monkeypatch.setattr(h.OwnedFailureListener, 'start', start)
    monkeypatch.setattr(h, 'inspect_owned_fixture', lambda *a, **k: {})

    def unavailable(*a, **k):
        raise RuntimeError('private-setup-sentinel')

    monkeypatch.setattr(h, 'RequestStore', unavailable)

    class HTTP:
        async def post(self, *a, **k):
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {'requestCount': 0}
            )

    report = {'scenarios': [], 'cleanup': [], 'failures': [], 'status': 'incomplete'}
    with pytest.raises(RuntimeError, match='private-setup-sentinel'):
        await h.run_cell(
            SimpleNamespace(output=tmp_path, admin='synthetic'),
            report,
            'redis',
            {'url': 'redis://127.0.0.1:1/0'},
            {'container_id': 'owned', 'image_id': 'owned', 'bindings': {}},
            HTTP(),
            'S03',
        )
    assert len(listeners) == 1
    assert not listeners[0].server.is_serving()


@pytest.mark.parametrize('phase', ['preflight', 'workload'])
async def test_failed_workload_keeps_safe_artifact(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace

    h = runner()
    monkeypatch.setattr(h, 'load_fixture_manifest', lambda _: {'redis': {}})

    def fail(*a, **k):
        raise RuntimeError('private-failure-sentinel')

    monkeypatch.setattr(
        h,
        'preflight_fixtures',
        fail if phase == 'preflight' else lambda *a: ({}, {}, {}),
    )
    monkeypatch.setattr(
        h,
        'source_manifest',
        lambda _: dict(base_commit='base', canonical_path_content_sha256='a' * 64),
    )

    async def workload(*a):
        fail()

    monkeypatch.setattr(h, 'run_cell', workload)
    report = await h.qualify(
        SimpleNamespace(output=tmp_path, scenarios=['S03'], studio=tmp_path)
    )
    assert report['status'] == 'failed'
    assert report['failures'][0]['phase'] == phase
    artifact = tmp_path / (report['run_id'] + '.json')
    assert 'private-failure-sentinel' not in artifact.read_text()


def test_selected_subset_requires_both_store_cells():
    h = runner()
    report = valid_report()
    report['selected'] = ['S03']
    report['scenarios'] = [r for r in report['scenarios'] if r['case'] == 'S03']
    fabrics = {r['fabric'] for r in report['scenarios']}
    report['cleanup'] = [r for r in report['cleanup'] if r['fabric'] in fabrics]
    h.accept_report(report)
    report['scenarios'].pop()
    with pytest.raises(h.QualificationError):
        h.accept_report(report)


def test_secondary_report_repair_failure_is_still_failed(tmp_path, monkeypatch):
    h = runner()
    report = valid_report() | {'run_id': 'safe-repair', 'unserializable': object()}

    def fail(_):
        raise RuntimeError('private-repair-sentinel')

    monkeypatch.setattr(h, 'repair_report_value', fail)
    h.finalize_report(report, tmp_path)
    artifact = json.loads((tmp_path / 'safe-repair.json').read_text())
    assert artifact['status'] == 'failed'
    assert [r['phase'] for r in artifact['failures']] == [
        'report_write',
        'report_repair',
    ]
    assert 'private-repair-sentinel' not in json.dumps(artifact)
