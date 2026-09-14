"""Qualification measurement definitions must not include empty-lane gaps."""

import pytest
from test_request_store import store as store


def test_longest_backlogged_interval_excludes_empty_gaps():
    from tools.stress.llm_v3_aimock_e2e import longest_joint_backlog

    samples = [
        (0.0, True),
        (1.0, True),
        (2.0, False),
        (3.0, True),
        (4.0, True),
        (5.0, True),
        (6.0, False),
    ]
    assert longest_joint_backlog(samples) == [3.0, 4.0, 5.0]
    assert longest_joint_backlog([(0.0, False), (1.0, False)]) == []


def test_early_abort_reports_no_planned_submissions():
    from collections import Counter

    from tools.stress.llm_v3_aimock_e2e import summarize_counts

    counts = summarize_counts(Counter(), {}, submitted=0, provider_attempts=0)
    assert counts['submitted_logical'] == 0
    assert counts['completed'] == counts['failed'] == counts['expired'] == 0


@pytest.mark.parametrize(
    'terminal', ['succeeded', 'failed', 'oversized', 'cancelled', 'expired']
)
def test_observed_terminal_outcomes_match_real_store_without_double_counting(
    store, terminal
):
    from collections import Counter, defaultdict

    from test_request_store import request_for, run_request

    from tools.stress.llm_v3_aimock_e2e import observe_transition, summarize_counts

    req = request_for(store)
    token, seq = run_request(store, req)
    if terminal in {'cancelled', 'expired'}:
        if terminal == 'expired':
            store.client.hset(store.keys.request(req.attempt_id), 'expires_at_ms', 1)
        reply = store.cancel_or_expire(
            req.producer_id, req.attempt_id, expire=terminal == 'expired'
        )
        op = 'cancel'
    else:
        reply = store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'answer': 'x' * (2000 if terminal == 'oversized' else 1)},
            success=terminal != 'failed',
        )
        op = 'finish'
    counters, records = Counter(), defaultdict(dict)
    observe_transition(
        counters,
        records,
        fabric_id=store.keys.fabric_id,
        attempt_id=req.attempt_id,
        op=op,
        disposition=reply.disposition,
        state=reply.state,
        observed_at=1.0,
    )
    # Re-observing the same terminal identity must not count a second logical outcome.
    observe_transition(
        counters,
        records,
        fabric_id=store.keys.fabric_id,
        attempt_id=req.attempt_id,
        op=op,
        disposition='existing',
        state=reply.state,
        observed_at=2.0,
    )
    counts = summarize_counts(counters, records, submitted=1, provider_attempts=1)
    expected = (
        'completed'
        if terminal == 'succeeded'
        else 'failed'
        if terminal == 'oversized'
        else terminal
    )
    for field in ('completed', 'failed', 'cancelled', 'expired'):
        assert counts[field] == int(field == expected)
    assert sum(counters.values()) == 2
    assert all(
        set(row) <= {'terminal_state', 'terminal_at', 'finished', 'result_bytes'}
        for row in records.values()
    )


def test_sampler_only_failure_sets_report_and_cli_failure(
    tmp_path, monkeypatch, capsys
):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    args = qualification_args(tmp_path)
    args.stores.write_text(json.dumps(valid_manifest()))
    args.mock, args.admin = 'http://127.0.0.1:4140', 'http://127.0.0.1:4141'
    quiet_external_setup(monkeypatch)
    good = valid_report()
    monkeypatch.setattr(
        harness,
        'preflight_fixtures',
        lambda *a: (good['verified_stores'], good['aimock'], {'State': {'Pid': 1}}),
    )
    original = harness._run_qualification

    async def empty_workload(args, report, fixtures, container):
        await original(args, report, {}, container)

    monkeypatch.setattr(harness, '_run_qualification', empty_workload)

    def fail_memory(self):
        raise ValueError('private-sampler-sentinel')

    monkeypatch.setattr(harness.psutil.Process, 'memory_info', fail_memory)
    monkeypatch.setattr(
        'sys.argv',
        [
            'qualification',
            '--stores',
            str(args.stores),
            '--studio',
            str(tmp_path),
            '--mock',
            args.mock,
            '--admin',
            args.admin,
            '--aimock-container',
            'mock-owned',
            '--output',
            str(tmp_path),
        ],
    )
    assert harness.main() == 1
    reports = [
        json.loads(path.read_text())
        for path in tmp_path.glob('*.json')
        if '-source' not in path.name and path != args.stores
    ]
    assert len(reports) == 1
    report = reports[0]
    assert report['status'] == 'failed'
    assert report['counts']['submitted_logical'] == 0
    assert len(report['failures']) == 1
    failure = report['failures'][0]
    assert failure['category'] == 'ValueError'
    assert failure['phase'] == 'sampler'
    assert any(frame['function'] == 'fail_memory' for frame in failure['locations'])
    assert 'private-sampler-sentinel' not in json.dumps(report)
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'


@pytest.mark.parametrize(
    'status,expected', [('passed', 0), ('failed', 1), ('incomplete', 1)]
)
def test_cli_returns_failure_for_every_nonpassing_report(
    tmp_path, monkeypatch, status, expected
):
    from tools.stress import llm_v3_aimock_e2e as harness

    async def qualify(args):
        return {'run_id': 'synthetic', 'status': status}

    monkeypatch.setattr(harness, 'qualify', qualify)
    monkeypatch.setattr(
        'sys.argv',
        [
            'qualification',
            '--stores',
            'owned',
            '--studio',
            str(tmp_path),
            '--mock',
            'http://127.0.0.1:1',
            '--admin',
            'http://127.0.0.1:2',
            '--aimock-container',
            'owned',
            '--output',
            str(tmp_path),
        ],
    )
    assert harness.main() == expected


def test_submission_accounting_stops_at_the_actual_aborted_batch():
    import threading
    from collections import Counter

    from tools.stress.llm_v3_aimock_e2e import submit_calls, summarize_counts

    class Producer:
        def execute(self, *, calls, **kwargs):
            raise RuntimeError('synthetic failure before admission')

    submissions = Counter()
    assert (
        summarize_counts(
            Counter(), {}, submitted=submissions['items'], provider_attempts=0
        )['submitted_logical']
        == 0
    )
    with pytest.raises(RuntimeError):
        submit_calls(
            Producer(), calls=[1, 2, 3], submissions=submissions, lock=threading.Lock()
        )
    counts = summarize_counts(
        Counter(), {}, submitted=submissions['items'], provider_attempts=0
    )
    assert counts['submitted_logical'] == 3
    assert counts['admitted'] == counts['completed'] == 0


@pytest.mark.parametrize('status', ['failed', 'incomplete'])
def test_module_entrypoint_exits_nonzero_for_nonpassing_report(
    tmp_path, monkeypatch, status
):
    import runpy

    from tools.stress import llm_v3_aimock_e2e as harness

    def return_report(coroutine):
        coroutine.close()
        return {'run_id': 'synthetic', 'status': status}

    monkeypatch.setattr(harness.asyncio, 'run', return_report)
    monkeypatch.setattr(
        'sys.argv',
        [
            'qualification',
            '--stores',
            'owned',
            '--studio',
            str(tmp_path),
            '--mock',
            'http://127.0.0.1:1',
            '--admin',
            'http://127.0.0.1:2',
            '--aimock-container',
            'owned',
            '--output',
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(harness.__file__, run_name='__main__')
    assert exit.value.code == 1


def test_circuit_report_requires_serialized_successful_probes():
    from tools.stress.llm_v3_aimock_e2e import summarize_circuit

    rows = [
        {
            'op': 'circuit_feedback',
            'circuit': 'open',
            'outcome': 'unavailable',
            'probe': False,
            'reserved_items': 1,
            'attempt_id': 'a',
            'producer_id': 'original',
            'claim_id': 'failed',
            'execution_seq': 1,
            'expires_at_ms': 1000,
        }
    ]
    for index in range(3):
        common = dict(
            attempt_id=chr(97 + index),
            producer_id='original',
            claim_id=str(index),
            execution_seq=2,
            expires_at_ms=1000,
            probe=True,
            reserved_items=1,
        )
        rows.append(dict(common, op='start', circuit='half_open', outcome=None))
        rows.append(
            dict(
                common,
                op='circuit_feedback',
                outcome='success',
                circuit='closed' if index == 2 else 'half_open',
            )
        )
        rows.append(
            dict(
                common,
                op='finish',
                circuit='closed' if index == 2 else 'half_open',
                outcome=None,
                reserved_items=0,
            )
        )
    report = summarize_circuit(
        rows, producer_id='original', attempt_ids=['a', 'b', 'c'], capacity=4
    )
    assert report['successful_probe_count'] == 3
    assert report['failed_probe_count'] == 0
    assert report['peak_reserved_items'] == 1
    assert report['states'] == ['open', 'half_open', 'closed']
    failed = dict(rows[1], claim_id='boot-refused', execution_seq=1)
    boot = [
        failed,
        dict(failed, op='circuit_feedback', outcome='unavailable', circuit='open'),
        dict(failed, op='defer', circuit='open', reserved_items=0),
    ]
    with_boot = summarize_circuit(
        rows[:1] + boot + rows[1:],
        producer_id='original',
        attempt_ids=['a', 'b', 'c'],
        capacity=4,
    )
    assert with_boot['successful_probe_count'] == 3
    assert with_boot['failed_probe_count'] == 1
    # A lost middle probe must not be hidden by a final closed-state sample.
    with pytest.raises(ValueError):
        summarize_circuit(
            rows[:4] + rows[7:],
            producer_id='original',
            attempt_ids=['a', 'b', 'c'],
            capacity=4,
        )
    # Replacement identity cannot inherit the original batch's successful report.
    with pytest.raises(ValueError):
        summarize_circuit(
            rows, producer_id='replacement', attempt_ids=['a', 'b', 'c'], capacity=4
        )
    overlapping = [dict(row) for row in rows]
    overlapping.insert(2, dict(rows[4]))
    with pytest.raises(ValueError):
        summarize_circuit(
            overlapping, producer_id='original', attempt_ids=['a', 'b', 'c'], capacity=4
        )


@pytest.mark.parametrize(
    'mismatch',
    [
        'identity',
        'compose_identity',
        'image',
        'owner',
        'project',
        'service',
        'port',
        'network_port',
        'running',
    ],
)
def test_owned_fixture_rejects_changed_identity_before_mutation(monkeypatch, mismatch):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    fixture = dict(
        container_id='owned',
        image_id='image',
        compose_project='project',
        compose_file='/owned/compose.yml',
        compose_ports={},
        service='redis',
        owner_label='llm-dispatch-task10',
        bindings={'6379/tcp': '12345'},
    )
    info = {
        'Id': 'owned',
        'Image': 'image',
        'Config': {
            'Labels': {
                'marie.test.owner': 'llm-dispatch-task10',
                'com.docker.compose.project': 'project',
                'com.docker.compose.service': 'redis',
            }
        },
        'State': {'Running': True, 'Pid': 5},
        'HostConfig': {
            'PortBindings': {'6379/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '12345'}]}
        },
        'NetworkSettings': {
            'Ports': {'6379/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '12345'}]}
        },
        'Mounts': [],
    }
    if mismatch == 'identity':
        info['Id'] = 'other'
    elif mismatch == 'network_port':
        info['NetworkSettings']['Ports']['6379/tcp'][0]['HostPort'] = '54321'
    elif mismatch == 'compose_identity':
        pass
    elif mismatch == 'image':
        info['Image'] = 'other'
    elif mismatch in {'owner', 'project', 'service'}:
        key = (
            'marie.test.owner'
            if mismatch == 'owner'
            else 'com.docker.compose.' + mismatch
        )
        info['Config']['Labels'][key] = 'other'
    elif mismatch == 'port':
        info['HostConfig']['PortBindings']['6379/tcp'][0]['HostIp'] = '0.0.0.0'
    else:
        info['State']['Running'] = False

    def check_output(command, **kwargs):
        return (
            ('other' if mismatch == 'compose_identity' else 'owned')
            if command[1] == 'compose'
            else json.dumps([info]).encode()
        )

    monkeypatch.setattr(harness.subprocess, 'check_output', check_output)

    def forbidden(*args, **kwargs):
        pytest.fail('Mutation reached after identity mismatch')

    monkeypatch.setattr(harness.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='Owned fixture .* mismatch'):
        harness.fixture_action(fixture, 'stop')


def qualification_args(tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        stores=tmp_path / 'stores.json',
        studio=tmp_path,
        mock='http://127.0.0.1:4010',
        admin='http://127.0.0.1:4011',
        aimock_container='mock-owned',
        output=tmp_path,
        seed=10,
        items=80,
    )


def quiet_external_setup(monkeypatch):
    import json
    from types import SimpleNamespace

    from tools.stress import llm_v3_aimock_e2e as harness

    monkeypatch.setattr(
        harness,
        'source_manifest',
        lambda root: {
            'base_commit': 'test',
            'canonical_path_content_sha256': 'a' * 64,
            'files': [],
        },
    )
    monkeypatch.setattr(
        harness.subprocess,
        'check_output',
        lambda *a, **kw: json.dumps(
            [
                {
                    'Config': {'Labels': {'marie.test.owner': 'llm-dispatch-task10'}},
                    'Image': 'synthetic',
                    'Id': 'mock-owned',
                    'State': {'Pid': 1},
                }
            ]
        ).encode(),
    )

    class Process:
        def __init__(self, *a):
            pass

        def cpu_times(self):
            return SimpleNamespace(user=0, system=0)

        def memory_info(self):
            return SimpleNamespace(rss=1)

        def children(self, **kw):
            return []

    class Http:
        def __init__(self, **kw):
            pass

        async def post(self, *a, **kw):
            await harness.asyncio.sleep(0.04)

        async def aclose(self):
            pass

    monkeypatch.setattr(harness.psutil, 'Process', Process)
    monkeypatch.setattr(harness.httpx, 'AsyncClient', Http)


@pytest.mark.parametrize('manifest', [None, '{', '{}', '[]', 'null', '"bad"'])
def test_invalid_manifest_returns_written_failure(tmp_path, monkeypatch, manifest):
    import asyncio
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    args = qualification_args(tmp_path)
    if manifest is not None:
        args.stores.write_text(manifest)
    quiet_external_setup(monkeypatch)
    report = asyncio.run(harness.qualify(args))
    assert report['status'] == 'failed'
    assert report['counts']['submitted_logical'] == 0
    assert report['scenarios'] == []
    assert (
        json.loads((tmp_path / f"{report['run_id']}.json").read_text())['status']
        == 'failed'
    )


def valid_manifest():
    ports = dict(
        QUAL_REDIS_PORT='4143',
        QUAL_VALKEY_PORT='4142',
        QUAL_MOCK_PORT='4140',
        QUAL_ADMIN_PORT='4141',
    )
    return {
        family: dict(
            url=f'redis://127.0.0.1:{port}/0',
            version=version,
            container_id=family + '-owned',
            image_id=family + '-image',
            compose_project='owned-project',
            compose_file='/owned/compose.yml',
            compose_ports=ports.copy(),
            service=family,
            owner_label='llm-dispatch-task10',
            bindings={'6379/tcp': port},
        )
        for family, port, version in [
            ('redis', '4143', '7.4.2'),
            ('valkey', '4142', '8.1.6'),
        ]
    }


@pytest.mark.parametrize(
    'mutation',
    [
        'missing_family',
        'extra_family',
        'duplicate_url',
        'duplicate_container',
        'remote',
        'credentials',
        'scheme',
        'service',
        'owner',
        'binding',
        'port',
        'missing_identity',
        'empty_identity',
        'version',
        'project',
        'compose_file',
        'ports_env',
        'entry_type',
    ],
)
def test_manifest_rejected_before_external_calls(tmp_path, monkeypatch, mutation):
    import asyncio
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    manifest = valid_manifest()
    redis = manifest['redis']
    if mutation == 'missing_family':
        del manifest['valkey']
    elif mutation == 'extra_family':
        manifest['other'] = redis.copy()
    elif mutation == 'duplicate_url':
        manifest['valkey']['url'] = redis['url']
    elif mutation == 'duplicate_container':
        manifest['valkey']['container_id'] = redis['container_id']
    elif mutation == 'remote':
        redis['url'] = 'redis://example.com:4143/0'
    elif mutation == 'credentials':
        redis['url'] = 'redis://user:private@127.0.0.1:4143/0'
    elif mutation == 'scheme':
        redis['url'] = 'http://127.0.0.1:4143/0'
    elif mutation == 'service':
        redis['service'] = 'aimock'
    elif mutation == 'owner':
        redis['owner_label'] = 'unowned'
    elif mutation == 'binding':
        redis['bindings'] = {'6379/tcp': '5555'}
    elif mutation == 'port':
        redis['url'] = 'redis://127.0.0.1/0'
    elif mutation == 'missing_identity':
        del redis['image_id']
    elif mutation == 'empty_identity':
        redis['container_id'] = ''
    elif mutation == 'version':
        redis['version'] = '7.4.1'
    elif mutation == 'project':
        manifest['valkey']['compose_project'] = 'other'
    elif mutation == 'compose_file':
        redis['compose_file'] = 'relative.yml'
    elif mutation == 'ports_env':
        redis['compose_ports']['PATH'] = '/untrusted'
    elif mutation == 'entry_type':
        manifest['redis'] = []
    args = qualification_args(tmp_path)
    args.mock, args.admin = 'http://127.0.0.1:4140', 'http://127.0.0.1:4141'
    args.stores.write_text(json.dumps(manifest))

    def forbidden(*a, **kw):
        raise AssertionError('external boundary reached')

    monkeypatch.setattr(harness.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(harness.subprocess, 'run', forbidden)
    monkeypatch.setattr(harness.httpx, 'AsyncClient', forbidden)
    report = asyncio.run(harness.qualify(args))
    assert report['status'] == 'failed'
    assert report['counts']['submitted_logical'] == 0
    assert report['failures'][0]['category'] in {'ValueError', 'QualificationError'}
    assert report['failures'][0]['phase'] == 'manifest'
    assert 'private' not in json.dumps(report)


def valid_report():
    stores = {}
    cells = []
    cleanup = []
    for family, version in [('redis', '7.4.2'), ('valkey', '8.1.6')]:
        stores[family] = dict(
            family=family,
            version=version,
            verified=True,
            container_id=family + '-owned',
            image_id=family + '-image',
            running=True,
            service=family,
            compose_project='owned-project',
            bindings={'6379/tcp': '4143' if family == 'redis' else '4142'},
            policy={
                'appendonly': 'yes',
                'appendfsync': 'always',
                'maxmemory': '134217728',
                'maxmemory-policy': 'noeviction',
            },
        )
        cells.extend(
            [
                dict(
                    name=family + '-multimodal-two-lane-load',
                    status='passed',
                    family=family,
                    fabric=family + '-fabric',
                    items=160,
                    provider_sends=160,
                    starts_during_overlap_by_lane={'one': 40, 'two': 120},
                ),
                dict(
                    name=family + '-same-producer-nine-recovery',
                    status='passed',
                    family=family,
                    fabric=family + '-fabric',
                    completed=9,
                    provider_sends=9,
                ),
                dict(
                    name=family + '-drain-and-session-cleanup',
                    status='passed',
                    family=family,
                    fabric=family + '-fabric',
                    remaining_records=0,
                    remaining_reservations=0,
                    server_version=version,
                ),
            ]
        )
        cleanup.append(
            dict(
                family=family,
                fabric=family + '-fabric',
                remaining_keys=0,
                remaining_records=0,
                remaining_reservations=0,
            )
        )
    report = dict(
        schema_version=1,
        run_id='synthetic',
        status='passed',
        failures=[],
        scenarios=cells,
        cleanup=cleanup,
        verified_stores=stores,
        source={
            'backend': {'base_commit': 'b' * 40, 'digest': 'a' * 64},
            'studio': {'base_commit': 'c' * 40, 'digest': 'd' * 64},
        },
        aimock=dict(
            version='1.29.0',
            server_sha256='e' * 64,
            verified=True,
            journal_max_entries=2,
            fixture_counts_max_test_ids=4,
            container_id='mock-owned',
            image_id='mock-image',
        ),
        config={'items_per_lane': 80},
        counts=dict(
            submitted_logical=338,
            admitted=338,
            rejected=0,
            claimed=350,
            started=350,
            retried=12,
            recovered_unsent=0,
            completed=338,
            failed=0,
            cancelled=0,
            expired=0,
            unknown=0,
            provider_attempts=350,
        ),
        measurements={
            'elapsed_seconds': 8.0,
            'transport_categories': {'success': 338, 'unavailable': 12},
            'resources': {'process_rss_bytes': {'count': 5}},
        },
    )

    from dataclasses import asdict

    from marie.engine.llm_queue.registry import (
        MAX_SNAPSHOT_BYTES,
        MAX_SNAPSHOT_READ_UNITS,
    )

    from tools.stress import llm_v3_aimock_e2e as harness

    report['config']['limits'] = asdict(harness.StoreLimits(max_execution_items=4))
    report['config']['event_snapshot_limits'] = dict(
        bytes=MAX_SNAPSHOT_BYTES,
        read_units=MAX_SNAPSHOT_READ_UNITS,
    )
    scopes, samples, records, transports = [], [], {}, []
    for cell in cells:
        scope = dict(
            family=cell['family'],
            scenario=cell['name'],
            fabric=cell['fabric'],
            cpu_sample_count=2,
            process_cpu_seconds=0.1,
            started_at=1.0,
            finished_at=2.0,
        )
        scopes.append(scope)
        identity = {key: scope[key] for key in ('family', 'scenario', 'fabric')}
        samples.append(
            dict(identity, values={key: 1 for key in harness.RESOURCE_FIELDS})
        )
        count = cell.get('items', cell.get('completed', 0))
        for index in range(count):
            records[cell['fabric'] + ':' + cell['name'] + str(index)] = dict(
                identity,
                admitted=1.0,
                first_start=2.0,
                last_start=2.0,
                finished=3.0,
                delivered=4.0,
            )
            transports.append(dict(identity, elapsed_ms=1.0, category='success'))
        if 'completed' in cell:
            transports.extend(
                dict(identity, elapsed_ms=1.0, category='unavailable') for _ in range(6)
            )
    report['measurements'].update(
        resource_samples=samples,
        by_scenario={
            scope['scenario']: harness.summarize_scope_measurements(
                [scope], samples, records, transports
            )
            for scope in scopes
        },
        by_family={
            family: harness.summarize_scope_measurements(
                [scope for scope in scopes if scope['family'] == family],
                samples,
                records,
                transports,
            )
            for family in stores
        },
    )
    return report


def test_literal_success_report_passes_actual_finalization(tmp_path):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    harness.finalize_report(report, tmp_path, items=80)
    assert report['status'] == 'passed'
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert saved['counts']['completed'] == 338
    assert len(saved['scenarios']) == 6


@pytest.mark.parametrize(
    'mutation',
    [
        'no_scenarios',
        'missing_cell',
        'duplicate_cell',
        'extra_cell',
        'failed_cell',
        'cell_items',
        'cell_sends',
        'recovery_completed',
        'cleanup_cell_records',
        'cleanup_cell_reservations',
        'cleanup_missing',
        'cleanup_duplicate',
        'cleanup_keys',
        'cleanup_records',
        'cleanup_reservations',
        'cleanup_fabric',
        'store_missing',
        'store_extra',
        'store_unverified',
        'store_version',
        'mock_version',
        'mock_source',
        'mock_retention',
        'mock_unverified',
        'completed',
        'submitted',
        'admitted',
        'failed',
        'negative',
        'float',
        'bool',
        'counter_missing',
        'provider_attempts',
        'provider_success',
        'measurements',
        'source',
        'items_config',
        'lane_missing',
    ],
)
def test_finalization_rejects_mutated_success(tmp_path, mutation):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    if mutation == 'no_scenarios':
        report['scenarios'] = []
    elif mutation == 'missing_cell':
        report['scenarios'].pop()
    elif mutation == 'duplicate_cell':
        report['scenarios'][5] = report['scenarios'][0]
    elif mutation == 'extra_cell':
        report['scenarios'].append(report['scenarios'][0].copy())
    elif mutation == 'failed_cell':
        report['scenarios'][0]['status'] = 'failed'
    elif mutation == 'cell_items':
        report['scenarios'][0]['items'] = 159
    elif mutation == 'cell_sends':
        report['scenarios'][0]['provider_sends'] = 159
    elif mutation == 'recovery_completed':
        report['scenarios'][1]['completed'] = 8
    elif mutation == 'cleanup_cell_records':
        report['scenarios'][2]['remaining_records'] = 1
    elif mutation == 'cleanup_cell_reservations':
        report['scenarios'][2]['remaining_reservations'] = 1
    elif mutation == 'cleanup_missing':
        report['cleanup'].pop()
    elif mutation == 'cleanup_duplicate':
        report['cleanup'][1] = report['cleanup'][0].copy()
    elif mutation.startswith('cleanup_'):
        key = {
            'cleanup_keys': 'remaining_keys',
            'cleanup_records': 'remaining_records',
            'cleanup_reservations': 'remaining_reservations',
            'cleanup_fabric': 'fabric',
        }[mutation]
        report['cleanup'][0][key] = 'wrong' if key == 'fabric' else 1
    elif mutation == 'store_missing':
        del report['verified_stores']['redis']
    elif mutation == 'store_extra':
        report['verified_stores']['other'] = report['verified_stores']['redis'].copy()
    elif mutation == 'store_unverified':
        report['verified_stores']['redis']['verified'] = False
    elif mutation == 'store_version':
        report['verified_stores']['redis']['version'] = '7.4.1'
    elif mutation == 'mock_version':
        report['aimock']['version'] = '1.28.0'
    elif mutation == 'mock_source':
        report['aimock']['server_sha256'] = ''
    elif mutation == 'mock_retention':
        report['aimock']['journal_max_entries'] = 100
    elif mutation == 'mock_unverified':
        report['aimock']['verified'] = False
    elif mutation in {'completed', 'submitted', 'admitted', 'failed'}:
        report['counts'][{'submitted': 'submitted_logical'}.get(mutation, mutation)] = 1
    elif mutation == 'negative':
        report['counts']['rejected'] = -1
    elif mutation == 'float':
        report['counts']['rejected'] = 0.0
    elif mutation == 'bool':
        report['counts']['rejected'] = False
    elif mutation == 'counter_missing':
        del report['counts']['rejected']
    elif mutation == 'provider_attempts':
        report['counts']['provider_attempts'] = 337
    elif mutation == 'provider_success':
        report['measurements']['transport_categories']['success'] = 337
    elif mutation == 'measurements':
        del report['measurements']
    elif mutation == 'source':
        report['source']['backend']['digest'] = ''
    elif mutation == 'items_config':
        report['config']['items_per_lane'] = 81
    elif mutation == 'lane_missing':
        report['scenarios'][0]['starts_during_overlap_by_lane'] = {'one': 40}
    harness.finalize_report(report, tmp_path, items=80)
    assert report['status'] == 'failed'
    assert report['failures']
    assert json.loads((tmp_path / 'synthetic.json').read_text())['status'] == 'failed'


def test_missing_admission_lane_is_explicit_measurement_failure():
    from collections import Counter, defaultdict

    from tools.stress import llm_v3_aimock_e2e as harness

    records = defaultdict(dict)
    harness.observe_transition(
        Counter(),
        records,
        fabric_id='f',
        attempt_id='missing',
        op='start',
        disposition='started',
        state='running',
        observed_at=1.0,
    )
    with pytest.raises(ValueError):
        harness.overlap_lane_counts(records, ['f:missing'])


@pytest.mark.parametrize('bad_value', [{None: 1, 'one': 3}, object(), float('nan')])
def test_serialization_failure_writes_safe_failed_report(tmp_path, bad_value):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    report['scenarios'][0]['broken'] = bad_value
    harness.finalize_report(report, tmp_path, items=80)
    assert report['status'] == 'failed'
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['measurements']['elapsed_seconds'] == 8.0
    assert any(row['phase'] == 'report_write' for row in saved['failures'])


@pytest.mark.parametrize(
    'mutation',
    [
        'redis_version',
        'redis_is_valkey',
        'valkey_version',
        'aof',
        'fsync',
        'eviction',
        'memory',
        'mock_version',
        'mock_source',
        'journal',
        'counts',
    ],
)
def test_live_preflight_rejects_observed_drift(tmp_path, monkeypatch, mutation):
    import hashlib
    import json
    from pathlib import Path

    from tools.stress import llm_v3_aimock_e2e as harness

    args = qualification_args(tmp_path)
    args.mock, args.admin = 'http://127.0.0.1:4140', 'http://127.0.0.1:4141'
    manifest = valid_manifest()

    def identity(fixture, **kw):
        return dict(
            container_id=fixture['container_id'],
            image_id=fixture['image_id'],
            running=True,
            pid=1,
            service=fixture['service'],
            compose_project=fixture['compose_project'],
            bindings=fixture['bindings'],
        )

    monkeypatch.setattr(harness, 'inspect_owned_fixture', identity)

    class Client:
        def __init__(self, url):
            self.family = 'redis' if ':4143/' in url else 'valkey'

        def info(self, section):
            if self.family == 'redis':
                return {
                    'redis_version': (
                        '7.4.1' if mutation == 'redis_version' else '7.4.2'
                    ),
                    **(
                        {'valkey_version': '8.1.6'}
                        if mutation == 'redis_is_valkey'
                        else {}
                    ),
                }
            return {
                'valkey_version': '8.1.5' if mutation == 'valkey_version' else '8.1.6'
            }

        def config_get(self, *keys):
            policy = {
                'appendonly': 'yes',
                'appendfsync': 'always',
                'maxmemory': '134217728',
                'maxmemory-policy': 'noeviction',
            }
            key = {
                'aof': 'appendonly',
                'fsync': 'appendfsync',
                'eviction': 'maxmemory-policy',
                'memory': 'maxmemory',
            }.get(mutation)
            if key:
                policy[key] = 'wrong'
            return policy

        def close(self):
            pass

    monkeypatch.setattr(
        'marie.engine.llm_queue.queue_io._build_sync_client',
        lambda url, **kw: Client(url),
    )

    def output(command, **kw):
        if command[1] == 'inspect':
            return json.dumps(
                [
                    {
                        'Id': 'mock-owned',
                        'Image': 'mock-image',
                        'State': {'Pid': 1},
                        'Config': {
                            'Env': [
                                'AIMOCK_JOURNAL_MAX_ENTRIES='
                                + ('3' if mutation == 'journal' else '2'),
                                'AIMOCK_FIXTURE_COUNTS_MAX_TEST_IDS='
                                + ('5' if mutation == 'counts' else '4'),
                            ]
                        },
                    }
                ]
            ).encode()
        return json.dumps(
            {
                'version': '1.28.0' if mutation == 'mock_version' else '1.29.0',
                'server_sha256': (
                    'wrong'
                    if mutation == 'mock_source'
                    else hashlib.sha256(
                        Path('Dockerfiles/aimock/programmatic/server.ts').read_bytes()
                    ).hexdigest()
                ),
            }
        ).encode()

    monkeypatch.setattr(harness.subprocess, 'check_output', output)
    with pytest.raises(ValueError):
        harness.preflight_fixtures(args, manifest)


@pytest.mark.parametrize(
    'stage',
    ['source', 'http_setup', 'workload_cleanup', 'measurement', 'sampler_context'],
)
def test_setup_and_cleanup_failures_restore_instrumentation(
    tmp_path, monkeypatch, stage
):
    import asyncio
    import json
    from types import SimpleNamespace

    from marie.engine.llm_queue.registry import SnapshotReadBudget

    from tools.stress import llm_v3_aimock_e2e as harness

    args = qualification_args(tmp_path)
    args.mock, args.admin = 'http://127.0.0.1:4140', 'http://127.0.0.1:4141'
    args.stores.write_text(json.dumps(valid_manifest()))
    quiet_external_setup(monkeypatch)
    good = valid_report()
    monkeypatch.setattr(
        harness,
        'preflight_fixtures',
        lambda *a: (good['verified_stores'], good['aimock'], {'State': {'Pid': 1}}),
    )
    original_invoke = harness.RequestStore._invoke
    original_execute = harness.EndpointClient.execute
    original_read = SnapshotReadBudget.before_read
    closed = []

    def primary(*a, **kw):
        raise RuntimeError('private-primary')

    if stage == 'source':
        monkeypatch.setattr(harness, 'source_manifest', primary)
    elif stage == 'http_setup':
        monkeypatch.setattr(harness.httpx, 'AsyncClient', primary)
    else:

        class Client:
            def scan_iter(self, **kw):
                return iter(())

            def info(self, *a):
                return {'used_memory': 0}

        class Store:
            _invoke = original_invoke

            def __init__(self, *a, **kw):
                self.client = Client()
                self.keys = SimpleNamespace(fabric_id='owned-fabric', prefix='owned:')

            def usage(self):
                return {'records': 0, 'reserved_items': 0}

            def close(self):
                closed.append('store')
                if stage == 'workload_cleanup':
                    raise LookupError('private-store-close')

        class Runtime:
            _owner_ready = False
            _runner = object()

            def __init__(self, **kw):
                pass

            async def start(self):
                raise RuntimeError('private-workload')

            async def stop(self):
                closed.append('runtime')
                if stage == 'workload_cleanup':
                    raise OSError('private-runtime-stop')

        monkeypatch.setattr(harness, 'RequestStore', Store)
        monkeypatch.setattr(harness, 'RequestDispatcher', Runtime)
        if stage == 'sampler_context':

            def fail_memory(self):
                raise ValueError('private-sampler-early')

            monkeypatch.setattr(harness.psutil.Process, 'memory_info', fail_memory)
        if stage == 'measurement':
            calls = 0

            def cpu(self):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise ArithmeticError('private-cpu')
                return SimpleNamespace(user=0, system=0)

            monkeypatch.setattr(harness.psutil.Process, 'cpu_times', cpu)
    report = asyncio.run(harness.qualify(args))
    assert report['status'] == 'failed'
    assert harness.RequestStore._invoke is original_invoke
    assert harness.EndpointClient.execute is original_execute
    assert SnapshotReadBudget.before_read is original_read
    failures = report['failures']
    assert failures[0]['category'] == 'RuntimeError'
    assert all(row['locations'] for row in failures)
    if stage == 'workload_cleanup':
        assert closed == ['runtime', 'store']
        assert [row['category'] for row in failures] == [
            'RuntimeError',
            'OSError',
            'LookupError',
        ]
        assert [row['phase'] for row in failures] == [
            'workload',
            'cleanup_runtime',
            'cleanup_store_close',
        ]
    if stage == 'sampler_context':
        sample = next(row for row in failures if row['phase'] == 'sampler')
        assert sample['scenario'] is None
    if stage == 'measurement':
        assert failures[-1]['category'] == 'ArithmeticError'
        assert failures[-1]['phase'] == 'measurement'
    saved = json.loads((tmp_path / f"{report['run_id']}.json").read_text())
    assert saved['status'] == 'failed'
    assert 'private-' not in json.dumps(saved)


def test_report_write_failure_retries_with_failed_status(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    original = Path.write_text
    calls = 0

    def fail_first(self, data, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('private-write-error')
        return original(self, data, *a, **kw)

    monkeypatch.setattr(Path, 'write_text', fail_first)
    harness.finalize_report(report, tmp_path, items=80)
    assert report['status'] == 'failed'
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert saved['failures'][0]['phase'] == 'report_write'
    assert saved['measurements']['elapsed_seconds'] == 8
    assert 'private-write-error' not in json.dumps(saved)


@pytest.mark.parametrize(
    'path',
    [
        '.env',
        'nested/.env.local',
        'secrets/token.txt',
        'nested/id.key',
        'nested/cert.pem',
    ],
)
def test_source_manifest_never_reads_restricted_files(tmp_path, monkeypatch, path):
    from pathlib import Path

    from tools.stress import llm_v3_aimock_e2e as harness

    monkeypatch.setattr(
        harness.subprocess, 'check_output', lambda *a, **kw: path.encode() + b'\0'
    )
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('private-source-sentinel')

    def forbidden(*a, **kw):
        pytest.fail('restricted content read')

    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(ValueError):
        harness.source_manifest(tmp_path)


def test_optimized_python_keeps_circuit_and_acceptance_guards(tmp_path):
    import os
    import subprocess
    import sys

    script = '''
from tools.stress.llm_v3_aimock_e2e import summarize_circuit, QualificationError
try:
    summarize_circuit([], producer_id='p', attempt_ids=[], capacity=4)
except QualificationError:
    pass
else:
    raise RuntimeError('circuit guard removed')
'''
    result = subprocess.run(
        [sys.executable, '-O', '-c', script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_preserves_existing_url_derived_binding_contract(tmp_path):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    args = qualification_args(tmp_path)
    args.mock, args.admin = 'http://127.0.0.1:4140', 'http://127.0.0.1:4141'
    manifest = valid_manifest()
    for entry in manifest.values():
        del entry['bindings']
    args.stores.write_text(json.dumps(manifest))
    loaded = harness.load_fixture_manifest(args)
    assert loaded['redis']['bindings'] == {'6379/tcp': '4143'}
    assert loaded['valkey']['bindings'] == {'6379/tcp': '4142'}


def test_fixture_restoration_failure_preserves_primary_report(tmp_path, monkeypatch):
    import asyncio
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    report['scenario'] = 'redis-same-producer-nine-recovery'
    try:
        raise RuntimeError('private-primary')
    except RuntimeError as error:
        harness.record_failure(
            report, error, phase='workload', scenario=report['scenario']
        )

    def fail_inspect(*a, **kw):
        raise OSError('private-restore')

    monkeypatch.setattr(harness, 'inspect_owned_fixture', fail_inspect)
    asyncio.run(harness.restore_mock_fixture({}, [], report))
    harness.finalize_report(report, tmp_path, items=80)
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert [row['category'] for row in saved['failures']] == ['RuntimeError', 'OSError']
    assert saved['failures'][1]['phase'] == 'fixture_restore'
    assert all(row['locations'] for row in saved['failures'])
    assert 'private-' not in json.dumps(saved)


def test_circular_serialization_data_leaves_failure_report(tmp_path):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    cycle = []
    cycle.append(cycle)
    report['scenarios'][0]['broken'] = cycle
    harness.finalize_report(report, tmp_path, items=80)
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['measurements']['elapsed_seconds'] == 8


@pytest.mark.parametrize('optimized', [False, True])
def test_empty_manifest_real_cli_writes_failure(tmp_path, optimized):
    import json
    import os
    import subprocess
    import sys

    from tools.stress import llm_v3_aimock_e2e as harness

    stores = tmp_path / 'stores.json'
    stores.write_text('{}')
    command = [
        sys.executable,
        *(['-O'] if optimized else []),
        harness.__file__,
        '--stores',
        str(stores),
        '--studio',
        str(tmp_path),
        '--mock',
        'http://127.0.0.1:4140',
        '--admin',
        'http://127.0.0.1:4141',
        '--aimock-container',
        'unused-owned-identity',
        '--output',
        str(tmp_path),
    ]
    result = subprocess.run(
        command, env=os.environ.copy(), capture_output=True, text=True
    )
    assert result.returncode == 1, result.stderr
    printed = json.loads(result.stdout)
    report = json.loads((tmp_path / (printed['run_id'] + '.json')).read_text())
    assert report['status'] == 'failed'
    assert report['counts']['submitted_logical'] == 0
    assert report['scenarios'] == []
    assert report['failures'][0]['check'] == 'manifest_families'


def test_optimized_python_rejects_corrupted_final_counts(tmp_path):
    import json
    import os
    import subprocess
    import sys

    report = valid_report()
    report['counts']['completed'] = 337
    fixture = tmp_path / 'fixture.json'
    fixture.write_text(json.dumps(report))
    script = '''
import json, sys
from pathlib import Path
from tools.stress.llm_v3_aimock_e2e import finalize_report
path = Path(sys.argv[1])
report = json.loads(path.read_text())
finalize_report(report, path.parent, items=80)
if report['status'] != 'failed' or report['failures'][0]['check'] != 'logical_counts':
    raise RuntimeError('optimized acceptance guard did not reject count mismatch')
'''
    result = subprocess.run(
        [sys.executable, '-O', '-c', script, str(fixture)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / 'synthetic.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize(
    'option,value',
    [
        ('--mock', 'http://user:synthetic-credential-canary@127.0.0.1:4140'),
        ('--admin', 'http://user:synthetic-credential-canary@127.0.0.1:4141'),
        ('--mock', 'http://example.invalid:4140'),
        ('--admin', 'http://example.invalid:4141'),
        ('--items', '0'),
        ('--items', '161'),
    ],
    ids=[
        'mock_credentials',
        'admin_credentials',
        'mock_remote',
        'admin_remote',
        'items_low',
        'items_high',
    ],
)
def test_cli_invalid_fixture_arguments_write_safe_failure(tmp_path, option, value):
    import json
    import os
    import subprocess
    import sys

    from tools.stress import llm_v3_aimock_e2e as harness

    stores = tmp_path / 'stores.json'
    stores.write_text(json.dumps(valid_manifest()))
    command = [
        sys.executable,
        harness.__file__,
        '--stores',
        str(stores),
        '--studio',
        str(tmp_path),
        '--mock',
        'http://127.0.0.1:4140',
        '--admin',
        'http://127.0.0.1:4141',
        '--items',
        '80',
        '--aimock-container',
        'unused-owned-identity',
        '--output',
        str(tmp_path),
    ]
    command[command.index(option) + 1] = value
    result = subprocess.run(
        command, env=os.environ.copy(), capture_output=True, text=True
    )
    leaked = 'synthetic-credential-canary' in result.stdout + result.stderr
    assert not leaked
    assert result.returncode != 0
    reports = [path for path in tmp_path.glob('*.json') if path != stores]
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'failed'
    assert report['counts']['submitted_logical'] == 0
    assert report['scenarios'] == []
    assert report['failures'][0]['phase'] == 'manifest'
    leaked_report = 'synthetic-credential-canary' in json.dumps(report)
    assert not leaked_report


@pytest.mark.parametrize('container_kind', ['list', 'mapping'])
def test_deep_serialization_data_leaves_bounded_failure_report(
    tmp_path, container_kind
):
    import json
    import sys

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    nested = 'leaf'
    for _ in range(sys.getrecursionlimit() + 100):
        nested = [nested] if container_kind == 'list' else {'child': nested}
    report['scenarios'][0]['broken'] = nested
    harness.finalize_report(report, tmp_path, items=80)
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['measurements']['elapsed_seconds'] == 8.0
    failure = next(row for row in saved['failures'] if row['phase'] == 'report_write')
    assert failure['category'] == 'RecursionError'
    assert failure['locations']
    assert 'maximum_nesting_depth' in json.dumps(saved['scenarios'][0]['broken'])


def test_repair_exception_still_writes_minimal_safe_failure(tmp_path, monkeypatch):
    import json

    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    report['scenarios'][0]['broken'] = {None: 1, 'one': 2}

    def fail_repair(*args, **kwargs):
        raise RuntimeError('private-repair-canary')

    monkeypatch.setattr(harness, 'repair_report_value', fail_repair)
    harness.finalize_report(report, tmp_path, items=80)
    saved = json.loads((tmp_path / 'synthetic.json').read_text())
    assert report['status'] == saved['status'] == 'failed'
    assert saved['report_unavailable'] is True
    assert [row['category'] for row in saved['failures']] == [
        'TypeError',
        'RuntimeError',
    ]
    assert [row['phase'] for row in saved['failures']] == [
        'report_write',
        'report_repair',
    ]
    assert all(row['locations'] for row in saved['failures'])
    leaked = 'private-repair-canary' in json.dumps(saved)
    assert not leaked


def test_resource_sample_keeps_scope_across_await_and_ignores_stopped_runtime():
    import asyncio
    from types import SimpleNamespace

    from tools.stress import llm_v3_aimock_e2e as harness

    async def exercise():
        current = {'family': 'redis', 'scenario': 'redis-load', 'fabric': 'r'}
        captured = current.copy()

        class Client:
            def info(self, section):
                current.update(family='valkey', scenario='valkey-load', fabric='v')
                return {'used_memory': 111}

        class Process:
            def memory_info(self):
                return SimpleNamespace(rss=22)

        class Runtime:
            _owner_ready = True
            _stop = asyncio.Event()
            store = SimpleNamespace(keys=SimpleNamespace(fabric_id='r'))

            def health(self):
                raise AssertionError('stopped runtime must not be sampled')

        runtime = Runtime()
        runtime._stop.set()
        store = SimpleNamespace(client=Client(), keys=SimpleNamespace(fabric_id='r'))
        row = await harness.sample_scope_resources(
            captured, process=Process(), mock_pid=0, store=store, runtime=runtime
        )
        assert row['family'] == 'redis'
        assert row['scenario'] == 'redis-load'
        assert row['fabric'] == 'r'
        assert row['values']['store_used_memory'] == 111
        assert row['values']['process_rss_bytes'] == 22
        assert 'snapshot_ms' not in row['values']
        assert 'mock_rss_bytes' not in row['values']
        assert current['family'] == 'valkey'

    asyncio.run(exercise())


def test_transition_latency_scope_is_fixed_at_admission():
    from collections import Counter, defaultdict

    from tools.stress import llm_v3_aimock_e2e as harness

    records = defaultdict(dict)
    counters = Counter()
    for op, disposition, timestamp, scenario in [
        ('admit', 'admitted', 1.0, 'redis-load'),
        ('start', 'started', 2.0, 'redis-recovery'),
        ('finish', 'finished', 3.0, 'valkey-load'),
        ('result', 'result', 4.0, 'valkey-drain'),
    ]:
        harness.observe_transition(
            counters,
            records,
            fabric_id='r',
            attempt_id='a',
            op=op,
            disposition=disposition,
            state='succeeded' if op == 'finish' else '',
            observed_at=timestamp,
            family='redis',
            scenario=scenario,
        )
    assert records['r:a']['family'] == 'redis'
    assert records['r:a']['scenario'] == 'redis-load'
    summary = harness.summarize_latencies(list(records.values()), [])
    assert summary['queue_wait_ms'] == harness.quantiles([1000])
    assert summary['start_to_commit_ms'] == harness.quantiles([1000])
    assert summary['commit_to_delivery_ms'] == harness.quantiles([1000])
    assert summary['end_to_end_ms'] == harness.quantiles([3000])
    assert summary['provider_ms'] == harness.quantiles([])


def test_provider_measurement_keeps_entry_scope_across_await():
    import asyncio
    from types import SimpleNamespace

    from tools.stress import llm_v3_aimock_e2e as harness

    async def exercise():
        scope = dict(family='redis', scenario='redis-load', fabric='r')
        rows = []

        async def execute(call, *, timeout_seconds):
            await asyncio.sleep(0)
            scope.update(family='valkey', scenario='valkey-load', fabric='v')
            return SimpleNamespace(category=None)

        result = await harness.observe_provider_call(
            execute, object(), timeout_seconds=1, scope=scope, rows=rows
        )
        assert result.category is None
        assert len(rows) == 1
        assert rows[0]['family'] == 'redis'
        assert rows[0]['scenario'] == 'redis-load'
        assert rows[0]['fabric'] == 'r'
        assert rows[0]['elapsed_ms'] >= 0
        assert rows[0]['category'] == 'success'

    asyncio.run(exercise())


def test_cpu_measurement_counts_real_boundary_reads():
    from types import SimpleNamespace

    from tools.stress import llm_v3_aimock_e2e as harness

    class Process:
        def __init__(self):
            self.reads = iter(
                [SimpleNamespace(user=1, system=2), SimpleNamespace(user=2, system=2.5)]
            )

        def cpu_times(self):
            return next(self.reads)

    process = Process()
    scope = {}
    harness.observe_scope_cpu(scope, process)
    assert scope['process_cpu_seconds'] is None
    assert scope['cpu_sample_count'] == 1
    harness.observe_scope_cpu(scope, process)
    assert scope['process_cpu_seconds'] == 1.5
    assert scope['cpu_sample_count'] == 2


def test_scope_summary_excludes_other_family_and_marks_empty_latency():
    from tools.stress import llm_v3_aimock_e2e as harness

    scope = dict(
        family='redis',
        scenario='redis-drain',
        fabric='r',
        started_at=1.0,
        finished_at=2.0,
        cpu_sample_count=2,
        process_cpu_seconds=0.1,
    )
    rows = [
        dict(
            family='redis',
            scenario='redis-drain',
            fabric='r',
            values={'process_rss_bytes': 10, 'store_used_memory': 11},
        ),
        dict(
            family='valkey',
            scenario='valkey-drain',
            fabric='v',
            values={'process_rss_bytes': 1000, 'store_used_memory': 1111},
        ),
        dict(
            family='redis',
            scenario='redis-load',
            fabric='r',
            values={'process_rss_bytes': 3000, 'store_used_memory': 3333},
        ),
    ]
    result = harness.summarize_scope_measurements([scope], rows, {}, [])
    assert result['resources']['process_rss_bytes'] == harness.quantiles([10])
    assert result['resources']['store_used_memory'] == harness.quantiles([11])
    assert result['provider_ms'] == harness.quantiles([])
    assert result['end_to_end_ms'] == harness.quantiles([])
    assert result['process_cpu_seconds'] == 0.1
    assert result['cpu_sample_count'] == 2


@pytest.mark.parametrize(
    'mutation',
    [
        'missing_scopes',
        'mixed_family',
        'missing_cpu',
        'unobserved_cpu',
        'unobserved_rss',
        'unobserved_latency',
        'negative_latency',
        'cap',
    ],
)
def test_scoped_measurement_guard_rejects_missing_or_false_evidence(mutation):
    from tools.stress import llm_v3_aimock_e2e as harness

    report = valid_report()
    row = report['measurements']['by_scenario']['redis-multimodal-two-lane-load']
    if mutation == 'missing_scopes':
        del report['measurements']['by_scenario']
    elif mutation == 'mixed_family':
        row['family'] = 'valkey'
    elif mutation == 'missing_cpu':
        row['cpu_sample_count'] = 0
    elif mutation == 'unobserved_cpu':
        row['process_cpu_seconds'] = None
    elif mutation == 'unobserved_rss':
        row['resources']['process_rss_bytes'] = harness.quantiles([])
    elif mutation == 'unobserved_latency':
        row['queue_wait_ms'] = harness.quantiles([])
    elif mutation == 'negative_latency':
        row['queue_wait_ms']['p50'] = -1
    elif mutation == 'cap':
        row['resources']['reserved_items']['max'] = 5
    with pytest.raises((ValueError, KeyError)):
        harness.accept_report(report, items=80)


@pytest.mark.parametrize('protected_stop', [False, True])
async def test_sampler_info_shutdown_obeys_runtime_client_ownership(
    store, monkeypatch, protected_stop
):
    import asyncio
    import socket
    import threading
    from types import SimpleNamespace

    from tools.stress import llm_v3_aimock_e2e as harness

    runtime = harness.RequestDispatcher(
        store=store,
        endpoints=[
            harness.RegisteredEndpoint(
                'endpoint',
                'http://127.0.0.1:1/v1',
                allow_loopback=True,
                execution_limit=2,
                execution_bytes=100_000,
            )
        ],
        lanes=[
            harness.DispatchLane(
                'pool',
                'endpoint',
                execution_limit=2,
                execution_bytes=100_000,
            )
        ],
        poll_seconds=0.01,
    )
    store.release_owner(store.test_owner)
    await runtime.start()
    await harness.eventually(lambda: runtime._owner_ready)
    scope = dict(family='synthetic', scenario='drain', fabric=store.keys.fabric_id)
    sample_lock = asyncio.Lock()
    entered_recv, release_recv, closed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    reader = {}
    original_info, original_close = store.client.info, store.close
    original_recv = socket.socket.recv

    def info(section):
        reader['thread'] = threading.get_ident()
        return original_info(section)

    def recv(sock, *args, **kwargs):
        if threading.get_ident() == reader.get('thread') and not entered_recv.is_set():
            entered_recv.set()
            if not release_recv.wait(5):
                raise AssertionError('sampler INFO receive barrier was not released')
        return original_recv(sock, *args, **kwargs)

    def close():
        try:
            original_close()
        finally:
            closed.set()

    monkeypatch.setattr(store.client, 'info', info)
    monkeypatch.setattr(store, 'close', close)
    monkeypatch.setattr(socket.socket, 'recv', recv)

    async def sample():
        async with sample_lock:
            return await harness.sample_scope_resources(
                scope,
                process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1)),
                mock_pid=0,
                store=store,
                runtime=runtime,
            )

    sampling = asyncio.create_task(sample())
    stopping = None
    try:
        assert await asyncio.to_thread(entered_recv.wait, 5)
        stop_entered = asyncio.Event()

        async def stop():
            stop_entered.set()
            if protected_stop:
                await harness.stop_sampled_runtime(runtime, sample_lock)
            else:
                await runtime.stop()

        stopping = asyncio.create_task(stop())
        await stop_entered.wait()
        if not protected_stop:
            assert await asyncio.to_thread(closed.wait, 5)
        else:
            assert not runtime._stop.is_set(), (
                'runtime stop crossed an active sampler read'
            )
        release_recv.set()
        observed = (await asyncio.gather(sampling, return_exceptions=True))[0]
        await stopping
        if protected_stop:
            assert isinstance(observed, dict), repr(observed)
            assert observed['values']['store_used_memory'] > 0
        else:
            assert isinstance(observed, ValueError), repr(observed)
            frames = []
            traceback = observed.__traceback__
            while traceback:
                frames.append((traceback.tb_frame.f_code.co_name, traceback.tb_lineno))
                traceback = traceback.tb_next
            assert ('_read_from_socket', 90) in frames
    finally:
        release_recv.set()
        await asyncio.gather(sampling, return_exceptions=True)
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await runtime.stop()
