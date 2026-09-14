"""Owned-loopback V3/AIMock qualification; reports contain metadata only."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import psutil
from marie.engine.completion_contract import (
    CompletionCallParams,
)
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint
from marie.engine.llm_queue.producer import V3Producer
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import RequestStore, StoreLimits
from PIL import Image


def inspect_owned_fixture(fixture: dict, *, running: bool | None = None) -> dict:
    """Resolve the recorded Compose service before any destructive fixture action."""
    required = {
        'container_id',
        'image_id',
        'compose_project',
        'compose_file',
        'compose_ports',
        'service',
        'owner_label',
        'bindings',
    }
    if (
        not required <= fixture.keys()
        or fixture['owner_label'] != 'llm-dispatch-task10'
    ):
        raise ValueError('Complete owned fixture metadata is required')
    command = [
        'docker',
        'compose',
        '--env-file',
        '/dev/null',
        '-p',
        fixture['compose_project'],
        '-f',
        fixture['compose_file'],
        'ps',
        '-a',
        '-q',
        fixture['service'],
    ]
    identity = subprocess.check_output(
        command, env=os.environ | fixture['compose_ports'], text=True
    ).strip()
    if not identity or identity != fixture['container_id']:
        raise ValueError('Owned fixture Compose identity mismatch')
    info = json.loads(subprocess.check_output(['docker', 'inspect', identity]))[0]
    labels = info.get('Config', {}).get('Labels') or {}
    if info.get('Id') != identity:
        raise ValueError('Owned fixture container identity mismatch')
    if info.get('Image') != fixture['image_id']:
        raise ValueError('Owned fixture image mismatch')
    if labels.get('marie.test.owner') != fixture['owner_label']:
        raise ValueError('Owned fixture owner label mismatch')
    if labels.get('com.docker.compose.project') != fixture['compose_project']:
        raise ValueError('Owned fixture Compose project mismatch')
    if labels.get('com.docker.compose.service') != fixture['service']:
        raise ValueError('Owned fixture Compose service mismatch')
    if running is not None and info.get('State', {}).get('Running') is not running:
        raise ValueError('Owned fixture running state mismatch')
    if not fixture['bindings']:
        raise ValueError('Owned fixture loopback bindings mismatch')
    for internal, port in fixture['bindings'].items():
        expected = [{'HostIp': '127.0.0.1', 'HostPort': str(port)}]
        if info.get('HostConfig', {}).get('PortBindings', {}).get(internal) != expected:
            raise ValueError('Owned fixture configured loopback binding mismatch')
        if (
            info['State']['Running']
            and info.get('NetworkSettings', {}).get('Ports', {}).get(internal)
            != expected
        ):
            raise ValueError('Owned fixture published loopback binding mismatch')
    return {
        'container_id': identity,
        'image_id': info['Image'],
        'running': info['State']['Running'],
        'pid': info['State']['Pid'],
        'mounts': [
            mount['Name'] for mount in info['Mounts'] if mount['Type'] == 'volume'
        ],
        'verified_at': time.time(),
        'service': fixture['service'],
        'compose_project': fixture['compose_project'],
        'bindings': fixture['bindings'],
    }


def fixture_action(fixture: dict, action: str) -> dict:
    if action not in {'stop', 'start'}:
        raise ValueError('Only retained-volume stop/start is supported')
    before = inspect_owned_fixture(fixture, running=action == 'stop')
    command = ['docker', action]
    if action == 'stop':
        command += ['--time', '1']
    subprocess.run(command + [before['container_id']], check=True, capture_output=True)
    return {'action': action, 'before': before, 'finished_at': time.time()}


def summarize_circuit(
    rows: list[dict], *, producer_id: str, attempt_ids: list[str], capacity: int
) -> dict:
    """Validate recovery using transition-time observations, including failed boot probes."""
    require(
        rows and attempt_ids and len(set(attempt_ids)) == len(attempt_ids),
        'circuit_identity',
    )
    require(
        {row['producer_id'] for row in rows} == {producer_id},
        'circuit_producer',
    )
    require(
        {row['attempt_id'] for row in rows} <= set(attempt_ids),
        'circuit_attempts',
    )
    states = list(dict.fromkeys(row['circuit'] for row in rows))
    require(
        all(state in states for state in ('open', 'half_open', 'closed')),
        'circuit_states',
    )
    peak = max(row['reserved_items'] for row in rows)
    require(
        peak <= capacity,
        'circuit_capacity',
    )
    starts = {
        (r['claim_id'], r['execution_seq']): r for r in rows if r['op'] == 'start'
    }
    successful, failed = [], []
    active_probe = None
    for row in rows:
        identity = (row['claim_id'], row['execution_seq'])
        if row['op'] == 'start':
            require(
                active_probe is None,
                'circuit_probe_serial',
            )
            if row['probe']:
                require(
                    row['circuit'] == 'half_open',
                    'circuit_probe_state',
                )
                active_probe = identity
        elif row['op'] == 'finish' or row['op'] == 'defer':
            if identity == active_probe:
                active_probe = None
        elif row['op'] == 'circuit_feedback' and row['probe']:
            require(
                identity in starts and identity == active_probe,
                'circuit_feedback_identity',
            )
            (successful if row['outcome'] == 'success' else failed).append(row)
    require(
        len(successful) == 3,
        'circuit_probe_count',
    )
    require(
        [r['circuit'] for r in successful] == ['half_open', 'half_open', 'closed'],
        'circuit_probe_order',
    )
    for attempt in attempt_ids:
        require(
            len({r['expires_at_ms'] for r in rows if r['attempt_id'] == attempt}) == 1,
            'circuit_deadline',
        )
    return {
        'producer_id': producer_id,
        'attempt_ids': attempt_ids,
        'states': states,
        'successful_probe_count': len(successful),
        'failed_probe_count': len(failed),
        'peak_reserved_items': peak,
        'capacity': capacity,
        'transitions': rows,
    }


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        'count': len(ordered),
        **{
            label: (
                ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
                if ordered
                else None
            )
            for label, fraction in [('p50', 0.5), ('p95', 0.95), ('p99', 0.99)]
        },
        'max': max(ordered) if ordered else None,
    }


def longest_joint_backlog(samples: list[tuple[float, bool]]) -> list[float]:
    """Select a continuous sampled interval, never bridging an empty-lane sample."""
    longest = []
    current = []
    for timestamp, backlogged in samples:
        if not backlogged:
            current = []
            continue
        current.append(timestamp)
        if not longest or current[-1] - current[0] > longest[-1] - longest[0]:
            longest = current.copy()
    return longest


def observe_transition(
    counters: Counter,
    records: dict,
    *,
    fabric_id: str,
    attempt_id: str | None,
    op: str,
    disposition: str,
    state: str,
    observed_at: float,
    pool_id: str | None = None,
    payload_bytes: int = 0,
    result_bytes: int = 0,
    family: str | None = None,
    scenario: str | None = None,
) -> None:
    """Record bounded scalar observations, keeping logical terminals unique by identity."""
    counters[op + ':' + disposition] += 1
    if not attempt_id:
        return
    row = records[fabric_id + ':' + attempt_id]
    if state in {'succeeded', 'failed', 'cancelled', 'expired'}:
        row.setdefault('terminal_state', state)
        row.setdefault('terminal_at', observed_at)
    if op == 'admit' and disposition == 'admitted':
        row.update(admitted=observed_at, pool=pool_id, payload_bytes=payload_bytes)
        if family is not None and scenario is not None:
            row.setdefault('family', family)
            row.setdefault('scenario', scenario)
    elif op == 'start' and disposition == 'started':
        row.setdefault('first_start', observed_at)
        row['last_start'] = observed_at
    elif op == 'finish' and disposition in {'finished', 'result_too_large'}:
        row.update(finished=observed_at, result_bytes=result_bytes)
    elif op == 'result' and disposition == 'result':
        row.setdefault('delivered', observed_at)


LATENCY_FIELDS = {
    'queue_wait_ms': ('first_start', 'admitted'),
    'start_to_commit_ms': ('finished', 'last_start'),
    'commit_to_delivery_ms': ('delivered', 'finished'),
    'end_to_end_ms': ('delivered', 'admitted'),
}
RESOURCE_LIMITS = {
    'active_items': 'max_active_items',
    'payload_bytes': 'max_payload_bytes',
    'records': 'max_records',
    'storage_bytes': 'max_storage_bytes',
    'reserved_items': 'max_execution_items',
    'reserved_bytes': 'max_execution_bytes',
}


def summarize_latencies(records: list[dict], transports: list[dict]) -> dict:
    """Summarize only observed timestamps; missing observations remain unavailable."""
    return {
        **{
            key: quantiles(
                [
                    (row[end] - row[start]) * 1000
                    for row in records
                    if end in row and start in row
                ]
            )
            for key, (end, start) in LATENCY_FIELDS.items()
        },
        'provider_ms': quantiles([row['elapsed_ms'] for row in transports]),
    }


def observe_scope_cpu(scope: dict, process) -> None:
    """Read cumulative process CPU at the two scenario boundaries."""
    cpu = process.cpu_times()
    total = cpu.user + cpu.system
    scope['cpu_sample_count'] = scope.get('cpu_sample_count', 0) + 1
    if 'cpu_started_seconds' not in scope:
        scope['cpu_started_seconds'] = total
        scope['process_cpu_seconds'] = None
    else:
        scope['process_cpu_seconds'] = total - scope['cpu_started_seconds']


async def observe_provider_call(
    execute, call, *, timeout_seconds: float, scope: dict | None, rows: list[dict]
):
    """Bind provider latency to its entry scope before yielding to the transport."""
    identity = (
        {key: scope[key] for key in ('family', 'scenario', 'fabric')} if scope else {}
    )
    began = time.monotonic()
    result = await execute(call, timeout_seconds=timeout_seconds)
    rows.append(
        dict(
            identity,
            elapsed_ms=(time.monotonic() - began) * 1000,
            category=result.category or 'success',
        )
    )
    return result


async def sample_scope_resources(
    scope: dict, *, process, mock_pid: int, store, runtime
) -> dict:
    """Sample the selected physical store and runtime using a captured identity."""
    row = {key: scope[key] for key in ('family', 'scenario', 'fabric')}
    row['observed_at'] = time.monotonic()
    values = row['values'] = {'process_rss_bytes': process.memory_info().rss}
    if mock_pid > 0:
        try:
            mock_process = psutil.Process(mock_pid)
            values['mock_rss_bytes'] = sum(
                p.memory_info().rss
                for p in [mock_process, *mock_process.children(recursive=True)]
            )
        except psutil.Error:
            pass
    require(store.keys.fabric_id == row['fabric'], 'sample_store_identity')
    values['store_used_memory'] = (
        await asyncio.to_thread(store.client.info, 'memory')
    )['used_memory']
    if runtime._owner_ready and not runtime._stop.is_set():
        require(
            runtime.store.keys.fabric_id == row['fabric'], 'sample_runtime_identity'
        )
        began = time.monotonic()
        health = await asyncio.to_thread(runtime.health)
        values['snapshot_ms'] = (time.monotonic() - began) * 1000
        require('data:image' not in json.dumps(health), 'health_projection_safe')
        depths = [lane['request_queue_depth'] for lane in health['lanes']]
        row['backlog_overlap'] = (
            time.monotonic(),
            len(depths) == 2 and min(depths) > 0,
        )
        values.update({key: health['usage'][key] for key in RESOURCE_LIMITS})
    return row


async def stop_sampled_runtime(runtime, sample_lock: asyncio.Lock) -> None:
    """Finish active sampler reads before runtime shutdown closes their shared client."""
    async with sample_lock:
        await runtime.stop()


RESOURCE_FIELDS = (
    'loop_lag_ms',
    'snapshot_ms',
    'process_rss_bytes',
    'mock_rss_bytes',
    'store_used_memory',
    'event_projection_ms',
    'event_projection_bytes',
    'event_read_units',
    *RESOURCE_LIMITS,
)


def summarize_scope_measurements(
    scopes: list[dict], samples: list[dict], records: dict, transports: list[dict]
) -> dict:
    """Aggregate only observations whose family, scenario and fabric all match."""
    identities = {
        (scope['family'], scope['scenario'], scope['fabric']) for scope in scopes
    }

    def selected(row):
        return (row.get('family'), row.get('scenario'), row.get('fabric')) in identities

    sample_rows = [row for row in samples if selected(row)]
    request_rows = [
        row
        for key, row in records.items()
        if selected(dict(row, fabric=key.split(':', 1)[0]))
    ]
    transport_rows = [row for row in transports if selected(row)]
    cpu_available = bool(scopes) and all(
        scope.get('cpu_sample_count') == 2
        and scope.get('process_cpu_seconds') is not None
        for scope in scopes
    )
    return {
        'family': scopes[0]['family'] if scopes else None,
        'scenarios': [scope['scenario'] for scope in scopes],
        'fabrics': sorted({scope['fabric'] for scope in scopes}),
        'cpu_sample_count': sum(scope.get('cpu_sample_count', 0) for scope in scopes),
        'process_cpu_seconds': (
            sum(scope['process_cpu_seconds'] for scope in scopes)
            if cpu_available
            else None
        ),
        'elapsed_seconds': sum(
            scope['finished_at'] - scope['started_at']
            for scope in scopes
            if 'finished_at' in scope
        ),
        'resource_sample_count': len(sample_rows),
        'resources': {
            key: quantiles(
                [row['values'][key] for row in sample_rows if key in row['values']]
            )
            for key in RESOURCE_FIELDS
        },
        **summarize_latencies(request_rows, transport_rows),
        'transport_categories': dict(
            Counter(row['category'] for row in transport_rows)
        ),
        'latency_scope': 'admission scope; provider calls use entry scope; empty samples unavailable',
    }


def validate_distribution(row: dict) -> None:
    count = row['count']
    require(type(count) is int and count >= 0, 'measurement_sample_count')
    values = [row[key] for key in ('p50', 'p95', 'p99', 'max')]
    require(
        (
            all(value is None for value in values)
            if count == 0
            else all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in values
            )
            and values == sorted(values)
        ),
        'measurement_values',
    )


def accept_scoped_measurements(report: dict, *, items: int) -> None:
    measurements = report['measurements']
    cells = {row['name']: row for row in report['scenarios']}
    scenarios = measurements['by_scenario']
    families = measurements['by_family']
    require(set(scenarios) == set(cells), 'measurement_scenarios')
    require(set(families) == set(STORE_VERSIONS), 'measurement_families')
    require(
        all(
            sample['scenario'] in cells
            and sample['family'] == cells[sample['scenario']]['family']
            and sample['fabric'] == cells[sample['scenario']]['fabric']
            for sample in measurements['resource_samples']
        ),
        'measurement_sample_identity',
    )
    for name, row in scenarios.items():
        cell = cells[name]
        expected = (
            2 * items
            if name.endswith(SCENARIO_SUFFIXES[0])
            else (9 if name.endswith(SCENARIO_SUFFIXES[1]) else 0)
        )
        require(
            row['family'] == cell['family']
            and row['scenarios'] == [name]
            and row['fabrics'] == [cell['fabric']],
            'measurement_identity',
        )
        require(
            type(row['cpu_sample_count']) is int
            and row['cpu_sample_count'] == 2
            and type(row['process_cpu_seconds']) in (int, float)
            and math.isfinite(row['process_cpu_seconds'])
            and row['process_cpu_seconds'] >= 0
            and row['elapsed_seconds'] > 0,
            'measurement_cpu',
        )
        require(set(row['resources']) == set(RESOURCE_FIELDS), 'measurement_resources')
        selected = [
            sample
            for sample in measurements['resource_samples']
            if sample['scenario'] == name
        ]
        require(
            len(selected) == row['resource_sample_count'] > 0
            and all(
                sample['family'] == cell['family']
                and sample['fabric'] == cell['fabric']
                for sample in selected
            ),
            'measurement_resource_identity',
        )
        for key, distribution in row['resources'].items():
            validate_distribution(distribution)
            require(
                distribution
                == quantiles(
                    [
                        sample['values'][key]
                        for sample in selected
                        if key in sample['values']
                    ]
                ),
                'measurement_resource_samples',
            )
        for key in ('process_rss_bytes', 'store_used_memory'):
            require(row['resources'][key]['count'] > 0, 'measurement_memory')
        for key in RESOURCE_LIMITS:
            distribution = row['resources'][key]
            require(
                distribution['count'] > 0
                and distribution['max']
                <= report['config']['limits'][RESOURCE_LIMITS[key]],
                'measurement_capacity',
            )
        for key in (*LATENCY_FIELDS, 'provider_ms'):
            validate_distribution(row[key])
            require(
                (
                    row[key]['count'] >= expected
                    if key == 'provider_ms'
                    else row[key]['count'] == expected
                ),
                'measurement_latency',
            )
        categories = row['transport_categories']
        require(
            all(type(n) is int and n >= 0 for n in categories.values())
            and categories.get('success', 0) == expected
            and sum(categories.values()) == row['provider_ms']['count'],
            'measurement_transports',
        )
        if expected == 0:
            require(row['provider_ms']['count'] == 0, 'measurement_drain_latency')
        if name.endswith(SCENARIO_SUFFIXES[0]):
            require(
                row['resources']['event_projection_ms']['count'] > 0,
                'measurement_metadata',
            )
        for key, limit in (
            (
                'event_read_units',
                report['config']['event_snapshot_limits']['read_units'],
            ),
            (
                'event_projection_bytes',
                report['config']['event_snapshot_limits']['bytes'],
            ),
        ):
            require(
                row['resources'][key]['max'] is None
                or row['resources'][key]['max'] <= limit,
                'measurement_metadata_budget',
            )
    for family, row in families.items():
        children = [value for value in scenarios.values() if value['family'] == family]
        require(
            row['family'] == family
            and set(row['scenarios'])
            == {name for name, cell in cells.items() if cell['family'] == family}
            and row['fabrics'] == children[0]['fabrics']
            and row['cpu_sample_count']
            == sum(child['cpu_sample_count'] for child in children)
            and row['process_cpu_seconds']
            == sum(child['process_cpu_seconds'] for child in children),
            'measurement_family_identity',
        )
        for key in (*LATENCY_FIELDS, 'provider_ms'):
            validate_distribution(row[key])
            require(
                row[key]['count'] == sum(child[key]['count'] for child in children),
                'measurement_family_samples',
            )
        for key in RESOURCE_FIELDS:
            require(
                row['resources'][key]
                == quantiles(
                    [
                        sample['values'][key]
                        for sample in measurements['resource_samples']
                        if sample['family'] == family and key in sample['values']
                    ]
                ),
                'measurement_family_resources',
            )
    require(
        sum(row['provider_ms']['count'] for row in scenarios.values())
        == report['counts']['provider_attempts'],
        'measurement_transport_total',
    )


def summarize_counts(
    counters: Counter, records: dict, *, submitted: int, provider_attempts: int
) -> dict:
    terminals = Counter(row.get('terminal_state') for row in records.values())
    return {
        'submitted_logical': submitted,
        'admitted': counters['admit:admitted'],
        'rejected': counters['admit:backpressure'],
        'claimed': counters['claim:claimed'],
        'started': counters['start:started'],
        'retried': counters['defer:deferred'],
        'recovered_unsent': counters['recover:requeued'],
        'completed': terminals['succeeded'],
        'failed': terminals['failed'],
        'cancelled': terminals['cancelled'],
        'expired': terminals['expired'],
        'unknown': counters['mark_unknown:outcome_unknown'],
        'provider_attempts': provider_attempts,
    }


def submit_calls(producer: V3Producer, *, calls, submissions: Counter, lock, **kwargs):
    """Count logical items only when the workload actually enters producer.execute."""
    with lock:
        submissions['items'] += len(calls)
    return producer.execute(calls=calls, **kwargs)


def source_manifest(root: Path) -> dict:
    files = (
        subprocess.check_output(
            ['git', 'ls-files', '--modified', '--others', '--exclude-standard', '-z'],
            cwd=root,
        )
        .decode()
        .split('\0')
    )
    digest = hashlib.sha256()
    rows = []
    for name in sorted(set(filter(None, files))):
        path = root / name
        require(
            not any(
                part == 'secrets'
                or part == '.env'
                or part.startswith('.env.')
                or part.endswith(('.pem', '.key'))
                for part in path.parts
            ),
            'restricted_source_path',
        )
        content = path.read_bytes() if path.is_file() else b''
        digest.update(name.encode() + b'\0' + content + b'\0')
        rows.append(
            {
                'path': name,
                'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest(),
                'deleted': not path.exists(),
            }
        )
    return {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root)
        .decode()
        .strip(),
        'canonical_path_content_sha256': digest.hexdigest(),
        'files': rows,
    }


def loopback(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname != '127.0.0.1' or parsed.username or parsed.password:
        raise ValueError('Only credential-free owned loopback fixtures are accepted')
    return url


async def eventually(predicate, seconds: float = 10) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError('Bounded fixture condition was not reached')


STORE_VERSIONS = {'redis': '7.4.2', 'valkey': '8.1.6'}
STORE_POLICY = {
    'appendonly': 'yes',
    'appendfsync': 'always',
    'maxmemory': '134217728',
    'maxmemory-policy': 'noeviction',
}
SCENARIO_SUFFIXES = (
    'multimodal-two-lane-load',
    'same-producer-nine-recovery',
    'drain-and-session-cleanup',
)


class QualificationError(ValueError):
    def __init__(self, check: str):
        self.check = check
        super().__init__(check)


def require(condition: bool, check: str) -> None:
    if not condition:
        raise QualificationError(check)


def record_failure(
    report: dict, error: BaseException, *, phase: str, scenario: str | None = None
) -> None:
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(
            {
                'filename': Path(tb.tb_frame.f_code.co_filename).name,
                'function': tb.tb_frame.f_code.co_name,
                'line': tb.tb_lineno,
            }
        )
        tb = tb.tb_next
    report['status'] = 'failed'
    report['failures'].append(
        {
            'category': type(error).__name__[:80],
            'phase': phase,
            'scenario': scenario,
            'check': (
                error.check if isinstance(error, QualificationError) else 'exception'
            ),
            'locations': frames[-12:],
        }
    )


def load_fixture_manifest(args) -> dict:
    fixtures = json.loads(Path(args.stores).read_text())
    require(
        isinstance(fixtures, dict) and set(fixtures) == set(STORE_VERSIONS),
        'manifest_families',
    )
    require(type(args.items) is int and 16 <= args.items <= 160, 'items_range')
    port_names = {
        'QUAL_REDIS_PORT',
        'QUAL_VALKEY_PORT',
        'QUAL_MOCK_PORT',
        'QUAL_ADMIN_PORT',
    }
    identities, urls = set(), set()
    for family, fixture in fixtures.items():
        require(isinstance(fixture, dict), 'manifest_entry')
        for field in ('container_id', 'image_id', 'compose_project', 'compose_file'):
            require(
                isinstance(fixture.get(field), str) and bool(fixture[field]),
                'manifest_identity',
            )
        require(Path(fixture['compose_file']).is_absolute(), 'manifest_compose_file')
        require(fixture.get('service') == family, 'manifest_service')
        require(fixture.get('owner_label') == 'llm-dispatch-task10', 'manifest_owner')
        require(fixture.get('version') == STORE_VERSIONS[family], 'manifest_version')
        ports = fixture.get('compose_ports')
        require(isinstance(ports, dict) and set(ports) == port_names, 'manifest_ports')
        require(
            all(
                isinstance(p, str) and p.isdecimal() and 0 < int(p) < 65536
                for p in ports.values()
            ),
            'manifest_ports',
        )
        require(len(set(ports.values())) == 4, 'manifest_duplicate_ports')
        parsed = urlsplit(loopback(fixture.get('url', '')))
        require(
            parsed.scheme == 'redis'
            and parsed.port is not None
            and parsed.path == '/0'
            and not parsed.query
            and not parsed.fragment,
            'manifest_url',
        )
        port = ports['QUAL_' + family.upper() + '_PORT']
        require(
            str(parsed.port) == port
            and fixture.get('bindings', {'6379/tcp': port}) == {'6379/tcp': port},
            'manifest_binding',
        )
        require(
            fixture['url'] not in urls and fixture['container_id'] not in identities,
            'manifest_duplicate_identity',
        )
        fixture['bindings'] = {'6379/tcp': port}
        urls.add(fixture['url'])
        identities.add(fixture['container_id'])
    for field in ('compose_project', 'compose_file', 'compose_ports'):
        require(
            fixtures['redis'][field] == fixtures['valkey'][field],
            'manifest_compose_identity',
        )
    for address, key in (
        (args.mock, 'QUAL_MOCK_PORT'),
        (args.admin, 'QUAL_ADMIN_PORT'),
    ):
        parsed = urlsplit(loopback(address))
        require(
            parsed.scheme == 'http'
            and str(parsed.port) == ports[key]
            and parsed.path in {'', '/'}
            and not parsed.query
            and not parsed.fragment,
            'manifest_mock_url',
        )
    require(
        isinstance(args.aimock_container, str)
        and bool(args.aimock_container)
        and args.aimock_container not in identities,
        'manifest_mock_identity',
    )
    return fixtures


def preflight_fixtures(args, fixtures: dict) -> tuple[dict, dict, dict]:
    from marie.engine.llm_queue.queue_io import _build_sync_client

    observed = {}
    for family, fixture in fixtures.items():
        identity = inspect_owned_fixture(fixture, running=True)
        client = _build_sync_client(
            fixture['url'], socket_connect_timeout=2, socket_timeout=2
        )
        try:
            server = client.info('server')
            actual_family = 'valkey' if server.get('valkey_version') else 'redis'
            actual_version = server.get(actual_family + '_version')
            require(
                actual_family == family and actual_version == STORE_VERSIONS[family],
                'store_version',
            )
            policy = client.config_get(*STORE_POLICY)
            require(policy == STORE_POLICY, 'store_policy')
            observed[family] = dict(
                identity,
                family=actual_family,
                version=actual_version,
                policy=policy,
                verified=True,
            )
        finally:
            client.close()
    container = json.loads(
        subprocess.check_output(['docker', 'inspect', args.aimock_container])
    )[0]
    require(container['Id'] == args.aimock_container, 'mock_container_identity')
    mock_fixture = dict(
        fixtures['redis'],
        service='aimock',
        container_id=container['Id'],
        image_id=container['Image'],
        bindings={
            '4010/tcp': str(urlsplit(args.mock).port),
            '4011/tcp': str(urlsplit(args.admin).port),
        },
    )
    mock_identity = inspect_owned_fixture(mock_fixture, running=True)
    wanted = {
        'AIMOCK_JOURNAL_MAX_ENTRIES': '2',
        'AIMOCK_FIXTURE_COUNTS_MAX_TEST_IDS': '4',
    }
    limits = {
        entry.split('=', 1)[0]: entry.split('=', 1)[1]
        for entry in container['Config'].get('Env', [])
        if '=' in entry and entry.split('=', 1)[0] in wanted
    }
    require(limits == wanted, 'mock_retention')
    script = (
        "const fs=require('fs'), c=require('crypto'); console.log(JSON.stringify({"
        "version:JSON.parse(fs.readFileSync('/app/node_modules/@copilotkit/aimock/package.json')).version,"
        "server_sha256:c.createHash('sha256').update(fs.readFileSync('/app/server.ts')).digest('hex')}));"
    )
    package = json.loads(
        subprocess.check_output(
            ['docker', 'exec', container['Id'], 'node', '-e', script]
        )
    )
    source_hash = hashlib.sha256(
        (
            Path(__file__).resolve().parents[2]
            / 'Dockerfiles/aimock/programmatic/server.ts'
        ).read_bytes()
    ).hexdigest()
    require(package.get('version') == '1.29.0', 'mock_version')
    require(package.get('server_sha256') == source_hash, 'mock_source')
    mock = dict(
        mock_identity,
        version=package['version'],
        server_sha256=package['server_sha256'],
        journal_max_entries=int(limits['AIMOCK_JOURNAL_MAX_ENTRIES']),
        fixture_counts_max_test_ids=int(limits['AIMOCK_FIXTURE_COUNTS_MAX_TEST_IDS']),
        verified=True,
    )
    return observed, mock, container


async def restore_mock_fixture(fixture: dict, lifecycle: list, report: dict) -> None:
    try:
        identity = await asyncio.to_thread(inspect_owned_fixture, fixture)
        if not identity['running']:
            lifecycle.append(await asyncio.to_thread(fixture_action, fixture, 'start'))
    except Exception as error:
        record_failure(
            report, error, phase='fixture_restore', scenario=report['scenario']
        )


def overlap_lane_counts(records: dict, attempts: list[str]) -> dict:
    lanes = [records.get(attempt, {}).get('pool') for attempt in attempts]
    require(
        bool(lanes) and all(lane in {'one', 'two'} for lane in lanes),
        'overlap_lane_identity',
    )
    counts = Counter(lanes)
    require(counts['one'] > 0 and counts['two'] > 0, 'overlap_both_lanes')
    return dict(counts)


def accept_report(report: dict, *, items: int) -> None:
    require(not report['failures'] and report['status'] == 'passed', 'run_status')
    require(
        type(items) is int
        and 16 <= items <= 160
        and report['config']['items_per_lane'] == items,
        'items_config',
    )
    for source in ('backend', 'studio'):
        digest = report['source'][source]['digest']
        require(
            isinstance(digest, str)
            and re.fullmatch('[0-9a-f]{64}', digest) is not None
            and bool(report['source'][source]['base_commit']),
            'source_identity',
        )
    stores = report['verified_stores']
    require(set(stores) == set(STORE_VERSIONS), 'verified_families')
    for family, row in stores.items():
        require(
            row['verified'] is True
            and row['family'] == family
            and row['version'] == STORE_VERSIONS[family]
            and row['policy'] == STORE_POLICY
            and row['running'] is True,
            'verified_store',
        )
    mock = report['aimock']
    require(
        mock['verified'] is True
        and mock['version'] == '1.29.0'
        and re.fullmatch('[0-9a-f]{64}', mock['server_sha256']) is not None
        and mock['journal_max_entries'] == 2
        and mock['fixture_counts_max_test_ids'] == 4,
        'verified_mock',
    )
    cells = report['scenarios']
    expected_names = {
        family + '-' + suffix
        for family in STORE_VERSIONS
        for suffix in SCENARIO_SUFFIXES
    }
    require(
        len(cells) == 6
        and {row['name'] for row in cells} == expected_names
        and all(row['status'] == 'passed' for row in cells),
        'scenario_cells',
    )
    fabrics = {}
    for family in STORE_VERSIONS:
        load, recovery, drain = [
            next(row for row in cells if row['name'] == family + '-' + suffix)
            for suffix in SCENARIO_SUFFIXES
        ]
        require(
            all(
                row['family'] == family and row['fabric'] == load['fabric']
                for row in (load, recovery, drain)
            )
            and isinstance(load['fabric'], str)
            and bool(load['fabric']),
            'scenario_identity',
        )
        fabrics[family] = load['fabric']
        for row, fields in (
            (load, ('items', 'provider_sends')),
            (recovery, ('completed', 'provider_sends')),
        ):
            require(
                all(
                    type(row[key]) is int
                    and row[key] == (2 * items if row is load else 9)
                    for key in fields
                ),
                'scenario_counts',
            )
        lanes = load['starts_during_overlap_by_lane']
        require(
            set(lanes) == {'one', 'two'}
            and all(type(n) is int and n > 0 for n in lanes.values()),
            'scenario_lanes',
        )
        require(
            all(
                type(drain[key]) is int and drain[key] == 0
                for key in ('remaining_records', 'remaining_reservations')
            )
            and drain['server_version'] == STORE_VERSIONS[family],
            'scenario_cleanup',
        )
    require(len(set(fabrics.values())) == 2, 'fabric_identity')
    cleanup = report['cleanup']
    require(
        len(cleanup) == 2 and {row['family'] for row in cleanup} == set(STORE_VERSIONS),
        'cleanup_cells',
    )
    for row in cleanup:
        require(
            row['fabric'] == fabrics[row['family']]
            and all(
                type(row[key]) is int and row[key] == 0
                for key in (
                    'remaining_keys',
                    'remaining_records',
                    'remaining_reservations',
                )
            ),
            'cleanup_remaining',
        )
    counts = report['counts']
    required_counts = set(
        summarize_counts(Counter(), {}, submitted=0, provider_attempts=0)
    )
    require(
        set(counts) == required_counts
        and all(type(n) is int and n >= 0 for n in counts.values()),
        'counter_types',
    )
    expected = 4 * items + 18
    require(
        all(
            counts[key] == expected
            for key in ('submitted_logical', 'admitted', 'completed')
        ),
        'logical_counts',
    )
    require(
        all(counts[key] == 0 for key in ('failed', 'cancelled', 'expired', 'unknown')),
        'terminal_counts',
    )
    require(
        counts['claimed']
        >= counts['started']
        == counts['provider_attempts']
        >= expected,
        'transport_counts',
    )
    measurements = report['measurements']
    require(
        measurements['elapsed_seconds'] > 0 and bool(measurements['resources']),
        'measurements',
    )
    categories = measurements['transport_categories']
    require(
        all(type(n) is int and n >= 0 for n in categories.values())
        and categories.get('success') == expected
        and sum(categories.values()) == counts['provider_attempts'],
        'provider_acceptances',
    )

    accept_scoped_measurements(report, items=items)


def repair_report_value(value, ancestors: frozenset[int] = frozenset()):
    # Used only after serialization fails; invalid evidence remains visibly unavailable.
    if isinstance(value, (dict, list)):
        if len(ancestors) >= 32:
            return {'invalid_report_value': 'maximum_nesting_depth'}
        if id(value) in ancestors:
            return {'invalid_report_value': 'circular_reference'}
        ancestors = ancestors | {id(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return {'invalid_report_value': 'non_string_mapping_key'}
        return {
            key: repair_report_value(item, ancestors) for key, item in value.items()
        }
    if isinstance(value, list):
        return [repair_report_value(item, ancestors) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return {'invalid_report_value': type(value).__name__[:80]}


def finalize_report(report: dict, output: Path, *, items: int) -> None:
    if report['status'] == 'passed':
        try:
            accept_report(report, items=items)
        except Exception as error:
            record_failure(report, error, phase='acceptance')
    elif report['status'] != 'failed':
        record_failure(report, QualificationError('incomplete_run'), phase='acceptance')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    path = output / f"{report['run_id']}.json"
    try:
        serialized = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        )
        path.write_text(serialized)
    except Exception as error:
        record_failure(report, error, phase='report_write')
        try:
            repaired = repair_report_value(report)
        except Exception as secondary:
            record_failure(report, secondary, phase='report_repair')
            repaired = {
                'schema_version': 1,
                'run_id': report['run_id'],
                'status': 'failed',
                'report_unavailable': True,
                'failures': report['failures'],
            }
        report.clear()
        report.update(repaired)
        try:
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
            )
        except Exception as secondary:
            record_failure(report, secondary, phase='report_write')


async def qualify(args) -> dict:
    report = {
        'schema_version': 1,
        'run_id': str(uuid4()),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'contract_version': 'v3',
        'seed': args.seed,
        'status': 'incomplete',
        'scenarios': [],
        'recovery_observations': [],
        'failures': [],
        'source': {},
        'cleanup': [],
        'counts': summarize_counts(Counter(), {}, submitted=0, provider_attempts=0),
        'phase': 'manifest',
        'scenario': None,
    }
    original_invoke, original_execute = RequestStore._invoke, EndpointClient.execute
    budget_class = None
    original_read = None
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        fixtures = load_fixture_manifest(args)
        report['phase'] = 'preflight'
        observed, mock, container = await asyncio.to_thread(
            preflight_fixtures, args, fixtures
        )
        report['verified_stores'], report['aimock'] = observed, mock
        report['phase'] = 'setup'
        from marie.engine.llm_queue.registry import SnapshotReadBudget

        budget_class, original_read = SnapshotReadBudget, SnapshotReadBudget.before_read
        await _run_qualification(args, report, fixtures, container)
    except Exception as error:
        record_failure(
            report, error, phase=report['phase'], scenario=report['scenario']
        )
    finally:
        RequestStore._invoke, EndpointClient.execute = original_invoke, original_execute
        if budget_class is not None:
            budget_class.before_read = original_read
        finalize_report(report, args.output, items=args.items)
    return report


async def _run_qualification(
    args, report: dict, fixtures: dict, container: dict
) -> None:
    from marie.engine.llm_queue.registry import (
        MAX_SNAPSHOT_BYTES,
        MAX_SNAPSHOT_READ_UNITS,
        SnapshotReadBudget,
        read_runtime_snapshot,
    )

    # Use the gateway package entrypoint to satisfy its existing circular imports.
    import marie.serve.runtimes.gateway.marie
    from marie.serve.runtimes.servers.marie_gateway import (
        _llm_dispatch_runtime_event_message,
    )

    run_id = report['run_id']
    report['versions'] = {
        name: version(name)
        for name in ('valkey', 'httpx', 'httpcore', 'openai', 'opentelemetry-sdk')
    }
    report['python'] = platform.python_version()
    for label, root in [('backend', Path.cwd()), ('studio', Path(args.studio))]:
        manifest = source_manifest(root)
        (args.output / f'{run_id}-{label}-source.json').write_text(
            json.dumps(manifest, indent=2)
        )
        report['source'][label] = {
            'base_commit': manifest['base_commit'],
            'digest': manifest['canonical_path_content_sha256'],
        }
    resources = {
        'loop_lag_ms': [],
        'snapshot_ms': [],
        'process_rss_bytes': [],
        'mock_rss_bytes': [],
        'store_used_memory': [],
        'backlog_overlap': [],
        'event_projection_ms': [],
        'event_projection_bytes': [],
        'event_read_units': [],
    }
    resources.update({key: [] for key in RESOURCE_LIMITS})
    resource_samples = []
    scopes = {}
    current_scope = None
    transport_rows = []
    process = psutil.Process()
    before_cpu = process.cpu_times()
    counters = Counter()
    submissions = Counter()
    records = defaultdict(dict)
    sends = []
    circuit_rows = defaultdict(list)
    recovery_fabrics = set()
    transport_ms = []
    recovery_transports = defaultdict(list)
    active_transports = 0
    transport_categories = Counter()
    original_execute = EndpointClient.execute

    async def measured_execute(self, call, *, timeout_seconds):
        nonlocal active_transports
        began = time.monotonic()
        active_transports += 1
        active_at_start = active_transports
        result = None
        try:
            result = await observe_provider_call(
                lambda call, **kw: original_execute(self, call, **kw),
                call,
                timeout_seconds=timeout_seconds,
                scope=current_scope,
                rows=transport_rows,
            )
            transport_ms.append((time.monotonic() - began) * 1000)
            transport_categories[result.category or 'success'] += 1
            return result
        finally:
            for fabric in recovery_fabrics:
                recovery_transports[fabric].append(
                    {
                        'started_at': began,
                        'finished_at': time.monotonic(),
                        'active_at_start': active_at_start,
                        'category': (
                            (result.category or 'success') if result else 'interrupted'
                        ),
                    }
                )
            active_transports -= 1

    EndpointClient.execute = measured_execute
    lock = threading.Lock()
    original = RequestStore._invoke
    owned_fabrics = set()

    def measured(self, op, **kwargs):
        with lock:
            result = original(self, op, **kwargs)
            if self.keys.fabric_id in owned_fabrics:
                now = time.monotonic()
                attempt = kwargs.get('attempt_id')
                observe_transition(
                    counters,
                    records,
                    fabric_id=self.keys.fabric_id,
                    attempt_id=attempt,
                    op=op,
                    disposition=result['disposition'],
                    state=result.get('state', ''),
                    observed_at=now,
                    family=current_scope['family'] if current_scope else None,
                    scenario=current_scope['scenario'] if current_scope else None,
                    pool_id=kwargs.get('pool_id'),
                    payload_bytes=result.get('payload_bytes', 0),
                    result_bytes=(
                        len(kwargs['result'].encode()) if op == 'finish' else 0
                    ),
                )
                if attempt and op == 'start' and result['disposition'] == 'started':
                    sends.append((now, self.keys.fabric_id + ':' + attempt))
                if (
                    self.keys.fabric_id in recovery_fabrics
                    and attempt
                    and op in {'start', 'circuit_feedback', 'finish', 'defer'}
                    and result['disposition']
                    in {'started', 'recorded', 'finished', 'deferred'}
                ):
                    request = self.client.hgetall(self.keys.request(attempt))
                    endpoint = self.client.hgetall(self.keys.endpoint('mock'))
                    circuit_rows[self.keys.fabric_id].append(
                        {
                            'op': op,
                            'observed_at': now,
                            'attempt_id': attempt,
                            'producer_id': request['producer_id'],
                            'claim_id': kwargs.get('claim_id'),
                            'execution_seq': int(request['execution_seq']),
                            'expires_at_ms': int(request['expires_at_ms']),
                            'outcome': kwargs.get('outcome'),
                            'circuit': endpoint.get('circuit', 'closed'),
                            'probe': endpoint.get('probe_claim')
                            == kwargs.get('claim_id'),
                            'reserved_items': int(endpoint.get('reserved_items', 0)),
                        }
                    )
        return result

    RequestStore._invoke = measured
    http = httpx.AsyncClient(trust_env=False, timeout=10)
    stores = []
    runtimes = []
    producers = []
    sample_stop = asyncio.Event()
    event_due = {}
    projected_backlog_events = Counter()
    read_units = 0
    original_before_read = SnapshotReadBudget.before_read

    def measured_before_read(self):
        nonlocal read_units
        original_before_read(self)
        read_units += 1

    SnapshotReadBudget.before_read = measured_before_read
    started = time.monotonic()

    sample_failure_context = {}

    sample_lock = asyncio.Lock()

    async def begin_scope(family, scenario, fabric):
        nonlocal current_scope
        async with sample_lock:
            if current_scope is not None:
                observe_scope_cpu(current_scope, process)
                current_scope['finished_at'] = time.monotonic()
            current_scope = dict(
                family=family,
                scenario=scenario,
                fabric=fabric,
                started_at=time.monotonic(),
            )
            scopes[scenario] = current_scope
            observe_scope_cpu(current_scope, process)

    async def sample_current_scope(scope, lag_ms=None):
        async with sample_lock:
            if scope is not current_scope:
                return
            if scope is None:
                resources['process_rss_bytes'].append(process.memory_info().rss)
                return
            # Select by fabric, never by the last mutable family in the workload.
            store = next(s for s in stores if s.keys.fabric_id == scope['fabric'])
            runtime = next(r for r in runtimes if r.store is store)
            row = await sample_scope_resources(
                scope,
                process=process,
                mock_pid=container['State']['Pid'],
                store=store,
                runtime=runtime,
            )
            values = row['values']
            if lag_ms is not None:
                values['loop_lag_ms'] = lag_ms
            if 'backlog_overlap' in row:
                resources['backlog_overlap'].append(row['backlog_overlap'])
            if (
                runtime._owner_ready
                and not runtime._stop.is_set()
                and time.monotonic() >= event_due.get(scope['fabric'], 0)
            ):
                began = time.monotonic()
                units_before = read_units
                snapshot = await read_runtime_snapshot(
                    fabric_group_id=scope['fabric'],
                    limit=50,
                    timeout_seconds=1,
                )
                event = _llm_dispatch_runtime_event_message(
                    snapshot=snapshot,
                    queue_config=LlmQueueConfig(
                        enabled=True,
                        fabric_group_id=scope['fabric'],
                        queue_contract_version='v3',
                    ),
                )
                projection = json.dumps(event.payload, default=str)
                require('data:image' not in projection, 'event_projection_safe')
                require(
                    event.payload['fabric_group_id'] == scope['fabric'], 'event_fabric'
                )
                values['event_projection_ms'] = (time.monotonic() - began) * 1000
                values['event_projection_bytes'] = len(projection.encode())
                values['event_read_units'] = read_units - units_before
                if row.get('backlog_overlap', (0, False))[1]:
                    projected_backlog_events[scope['fabric']] += 1
                event_due[scope['fabric']] = time.monotonic() + 1
            resource_samples.append(row)
            for key, value in values.items():
                resources[key].append(value)

    async def sampler():
        while not sample_stop.is_set():
            tick_scope = current_scope
            tick = time.monotonic()
            await asyncio.sleep(0.02)
            scope = current_scope
            try:
                await sample_current_scope(
                    scope,
                    (
                        max(0, (time.monotonic() - tick - 0.02) * 1000)
                        if scope is tick_scope
                        else None
                    ),
                )
            except Exception:
                sample_failure_context['scenario'] = (
                    scope['scenario'] if scope else None
                )
                raise

    sampling = asyncio.create_task(sampler())
    try:
        await http.post(
            args.admin + '/fault-profile',
            json={'profile': 'normal', 'resetCounters': True},
        )
        limits = StoreLimits(
            max_active_items=32,
            max_payload_bytes=32 * 1024**2,
            max_records=512,
            max_storage_bytes=64 * 1024**2,
            max_execution_items=4,
            max_execution_bytes=4 * 1024**2,
            result_allowance=256 * 1024,
            delivery_grace_ms=1000,
        )
        report['config'] = {
            'limits': asdict(limits),
            'items_per_lane': args.items,
            'producer_window_per_lane': 16,
            'producer_poll_seconds': 0.02,
            'dispatcher_poll_seconds': 0.01,
            'retry_min_ms': 100,
            'retry_max_ms': 200,
            'circuit_open_ms': 500,
            'batch_deadline_seconds': 60,
            'drain_seconds': 1,
            'persistence': 'AOF always; noeviction; standalone; physical maxmemory 128MiB',
            'event_projection_cadence_seconds': 1,
            'event_snapshot_limits': {
                'seconds': 1,
                'sample_rows': 50,
                'read_units': MAX_SNAPSHOT_READ_UNITS,
                'bytes': MAX_SNAPSHOT_BYTES,
            },
            'process_topology': 'one producer/dispatcher process; separate owned AIMock and store containers',
        }
        for family, fixture in fixtures.items():
            report['phase'] = 'workload'
            report['scenario'] = family + '-multimodal-two-lane-load'
            store = RequestStore(
                loopback(fixture['url']),
                fabric_id='qual-' + uuid4().hex,
                version='v3',
                limits=limits,
            )
            owned_fabrics.add(store.keys.fabric_id)
            stores.append(store)
            runtime = RequestDispatcher(
                store=store,
                endpoints=[
                    RegisteredEndpoint(
                        'mock',
                        args.mock + '/v1',
                        allow_loopback=True,
                        execution_limit=4,
                        execution_bytes=4 * 1024**2,
                        call_timeout_seconds=5,
                    )
                ],
                lanes=[
                    DispatchLane(
                        'one',
                        'mock',
                        execution_limit=4,
                        execution_bytes=4 * 1024**2,
                        quantum=5,
                        max_burst_per_visit=1,
                    ),
                    DispatchLane(
                        'two',
                        'mock',
                        execution_limit=4,
                        execution_bytes=4 * 1024**2,
                        quantum=15,
                        max_burst_per_visit=3,
                    ),
                ],
                poll_seconds=0.01,
                retry_min_ms=100,
                retry_max_ms=200,
                circuit_open_ms=500,
                owner_lease_ms=2000,
                drain_seconds=1,
                total_concurrent_dispatch=4,
            )
            runtimes.append(runtime)
            await runtime.start()
            await eventually(lambda: runtime._owner_ready)
            await begin_scope(family, report['scenario'], store.keys.fabric_id)
            lane_producers = [
                V3Producer(
                    config=LlmQueueConfig(
                        enabled=True,
                        queue_url=fixture['url'],
                        pool_id=lane,
                        fabric_group_id=store.keys.fabric_id,
                        queue_contract_version='v3',
                        max_buffered_requests_per_pool=16,
                    )
                )
                for lane in ('one', 'two')
            ]
            producers.extend(lane_producers)
            # Both original producer sessions run simultaneously through the actual V3 path.
            rng = random.Random(args.seed)
            images = []
            for _ in range(2):
                pixels = Image.frombytes(
                    'RGB', (256, 256), rng.randbytes(256 * 256 * 3)
                )
                data = io.BytesIO()
                pixels.save(data, format='PNG')
                images.append(
                    'data:image/png;base64,'
                    + base64.b64encode(data.getvalue()).decode()
                )
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'extract synthetic document'},
                        *[
                            {'type': 'image_url', 'image_url': {'url': value}}
                            for value in images
                        ],
                    ],
                }
            ]
            await http.post(
                args.admin + '/fault-profile',
                json={'profile': 'timeout', 'timeoutMs': 80, 'resetCounters': True},
            )
            began = time.monotonic()
            outputs = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        submit_calls,
                        producer,
                        submissions=submissions,
                        lock=lock,
                        calls=[
                            CompletionCallParams(model='synthetic', messages=messages)
                            for _ in range(args.items)
                        ],
                        batch_request_id=lane,
                        batch_timeout=60,
                    )
                    for lane, producer in zip(('one', 'two'), lane_producers)
                ]
            )
            require(
                all(
                    len(results) == args.items
                    and all(item.error is None for item in results)
                    for results in outputs
                ),
                'load_results',
            )
            fault = (await http.get(args.admin + '/fault-profile')).json()
            require(
                fault['requestCount'] == args.items * 2,
                'load_provider_count',
            )
            journal = (await http.get(args.mock + '/__aimock/journal')).json()
            require(
                len(journal) == 2,
                'load_journal_count',
            )
            journal_text = json.dumps(journal)
            require(
                all(value in journal_text for value in images),
                'load_images_present',
            )
            require(
                all(
                    [
                        part['image_url']['url']
                        for part in entry['body']['messages'][0]['content']
                        if part['type'] == 'image_url'
                    ]
                    == images
                    for entry in journal
                ),
                'load_image_order',
            )
            del journal, journal_text, outputs, messages, images
            overlap = longest_joint_backlog(
                [
                    row['backlog_overlap']
                    for row in resource_samples
                    if row['scenario'] == report['scenario']
                    and 'backlog_overlap' in row
                ]
            )
            require(
                len(overlap) >= 2 and overlap[-1] - overlap[0] > 0.5,
                'load_overlap_duration',
            )
            overlap_sends = [
                attempt
                for timestamp, attempt in sends
                if overlap[0] <= timestamp <= overlap[-1]
                and records[attempt].get('scenario') == report['scenario']
            ]
            by_lane = overlap_lane_counts(records, overlap_sends)
            require(
                by_lane['one'] > 0 and by_lane['two'] > 0,
                'load_both_lanes',
            )
            require(
                projected_backlog_events[store.keys.fabric_id] > 0,
                'load_backlog_events',
            )
            report['scenarios'].append(
                {
                    'backlogged_event_projections': projected_backlog_events[
                        store.keys.fabric_id
                    ],
                    'name': family + '-multimodal-two-lane-load',
                    'family': family,
                    'fabric': store.keys.fabric_id,
                    'status': 'passed',
                    'items': args.items * 2,
                    'provider_sends': fault['requestCount'],
                    'duration_seconds': time.monotonic() - began,
                    'backlog_overlap_seconds': overlap[-1] - overlap[0],
                    'backlog_overlap_samples': len(overlap),
                    'backlog_measurement': 'longest contiguous run of samples with both lanes ready',
                    'starts_during_overlap_by_lane': dict(by_lane),
                    'journal_entries': 2,
                    'group_images': 2,
                    'image_dimensions': [256, 256],
                    'ordered_group_verified': True,
                }
            )
            await http.post(
                args.admin + '/fault-profile',
                json={'profile': 'normal', 'resetCounters': True},
            )
            # Restart only the exact owned service, retaining its volumes.
            mock_fixture = dict(
                fixture,
                service='aimock',
                container_id=container['Id'],
                image_id=container['Image'],
                bindings={
                    '4010/tcp': str(urlsplit(args.mock).port),
                    '4011/tcp': str(urlsplit(args.admin).port),
                },
            )
            report['scenario'] = family + '-same-producer-nine-recovery'
            await begin_scope(family, report['scenario'], store.keys.fabric_id)
            recovery_fabrics.add(store.keys.fabric_id)
            retries = runtime._counts['retries']
            lifecycle = []
            report['recovery_observations'].append(
                {
                    'store': family,
                    'fixture_lifecycle': lifecycle,
                    'transitions': circuit_rows[store.keys.fabric_id],
                    'transport_calls': recovery_transports[store.keys.fabric_id],
                }
            )
            recovering = None
            try:
                lifecycle.append(
                    await asyncio.to_thread(fixture_action, mock_fixture, 'stop')
                )
                recovering = asyncio.create_task(
                    asyncio.to_thread(
                        submit_calls,
                        lane_producers[0],
                        submissions=submissions,
                        lock=lock,
                        calls=[
                            CompletionCallParams(
                                model='synthetic',
                                messages=[{'role': 'user', 'content': 'extract'}],
                            )
                            for _ in range(9)
                        ],
                        batch_request_id='same-nine',
                        batch_timeout=15,
                    )
                )
                await eventually(
                    lambda: store.endpoint_status('mock')['circuit'] == 'open'
                )
                await eventually(lambda: len(lane_producers[0]._pending) == 9)
                original_producer = lane_producers[0].producer_id
                original_attempts = sorted(lane_producers[0]._pending)
                original_deadlines = {
                    attempt: store.metadata(attempt).expires_at_ms
                    for attempt in original_attempts
                }
                # Docker publishes ports before the mock can consume HTTP.
                # Preserve the open circuit while gating starts until both listeners respond.
                await asyncio.to_thread(
                    store.configure_endpoint,
                    runtime.owner,
                    'mock',
                    execution_limit=4,
                    execution_bytes=4 * 1024**2,
                    gate_open=False,
                )
                await eventually(lambda: not runtime._calling)
                boot_gate_closed_at = time.monotonic()
                lifecycle.append(
                    await asyncio.to_thread(fixture_action, mock_fixture, 'start')
                )
                ready_until = time.monotonic() + 5
                while True:
                    try:
                        admin_ready = await http.get(
                            args.admin + '/fault-profile', timeout=0.25
                        )
                        mock_ready = await http.get(
                            args.mock + '/__aimock/journal', timeout=0.25
                        )
                        admin_ready.raise_for_status()
                        mock_ready.raise_for_status()
                        require(
                            admin_ready.json()['requestCount'] == 0,
                            'recovery_boot_count',
                        )
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= ready_until:
                            raise
                        await asyncio.sleep(0.02)
                container['State']['Pid'] = (
                    await asyncio.to_thread(
                        inspect_owned_fixture, mock_fixture, running=True
                    )
                )['pid']
                boot_ready_at = time.monotonic()
                await asyncio.to_thread(
                    store.configure_endpoint,
                    runtime.owner,
                    'mock',
                    execution_limit=4,
                    execution_bytes=4 * 1024**2,
                    gate_open=True,
                )
                container = json.loads(
                    subprocess.check_output(
                        ['docker', 'inspect', args.aimock_container]
                    )
                )[0]
                recovered = await recovering
                require(
                    len(recovered) == 9
                    and all(item.error is None for item in recovered),
                    'recovery_results',
                )
                require(
                    lane_producers[0].producer_id == original_producer,
                    'recovery_producer',
                )
                require(
                    {
                        attempt: store.metadata(attempt).expires_at_ms
                        for attempt in original_attempts
                    }
                    == original_deadlines,
                    'recovery_deadlines',
                )
                require(
                    all(
                        store.metadata(attempt).producer_id == original_producer
                        for attempt in original_attempts
                    ),
                    'recovery_attempt_ownership',
                )
                recovery_sends = (await http.get(args.admin + '/fault-profile')).json()[
                    'requestCount'
                ]
                require(
                    recovery_sends == 9,
                    'recovery_provider_count',
                )
                evidence = summarize_circuit(
                    circuit_rows[store.keys.fabric_id],
                    producer_id=original_producer,
                    attempt_ids=original_attempts,
                    capacity=4,
                )
                evidence['transport_calls'] = recovery_transports[store.keys.fabric_id]
                evidence['boot_gate_closed_at'] = boot_gate_closed_at
                evidence['boot_ready_at'] = boot_ready_at
                evidence['provider_acceptances_at_readiness'] = admin_ready.json()[
                    'requestCount'
                ]
                probe_transports = []
                for feedback in evidence['transitions']:
                    if (
                        feedback['op'] != 'circuit_feedback'
                        or not feedback['probe']
                        or feedback['outcome'] != 'success'
                    ):
                        continue
                    start = next(
                        row
                        for row in evidence['transitions']
                        if row['op'] == 'start'
                        and row['claim_id'] == feedback['claim_id']
                        and row['execution_seq'] == feedback['execution_seq']
                    )
                    matched = [
                        row
                        for row in evidence['transport_calls']
                        if start['observed_at']
                        <= row['started_at']
                        <= row['finished_at']
                        <= feedback['observed_at']
                    ]
                    require(
                        len(matched) == 1 and matched[0]['active_at_start'] == 1,
                        'recovery_serial_transport',
                    )
                    probe_transports.append(matched[0])
                evidence['successful_probe_transports'] = probe_transports
                require(
                    len(probe_transports) == 3,
                    'recovery_probe_transport_count',
                )
                evidence['peak_active_transports'] = max(
                    row['active_at_start'] for row in evidence['transport_calls']
                )
                require(
                    evidence['peak_active_transports'] <= 4,
                    'recovery_transport_capacity',
                )
                require(
                    sum(
                        row['category'] == 'success'
                        for row in evidence['transport_calls']
                    )
                    == recovery_sends,
                    'recovery_transport_acceptances',
                )
                require(
                    sum(
                        row['op'] == 'circuit_feedback' and row['outcome'] == 'success'
                        for row in evidence['transitions']
                    )
                    == recovery_sends,
                    'recovery_feedback_acceptances',
                )
                report['scenarios'].append(
                    {
                        'provider_sends': recovery_sends,
                        'name': family + '-same-producer-nine-recovery',
                        'family': family,
                        'fabric': store.keys.fabric_id,
                        'status': 'passed',
                        'completed': len(recovered),
                        'retries': runtime._counts['retries'] - retries,
                        'deadline_seconds': 15,
                        'same_batch': True,
                        'circuit_evidence': evidence,
                        'fixture_lifecycle': lifecycle,
                    }
                )
            finally:
                await restore_mock_fixture(mock_fixture, lifecycle, report)
                if recovering is not None and not recovering.done():
                    await asyncio.gather(recovering, return_exceptions=True)
                recovery_fabrics.discard(store.keys.fabric_id)
            report['scenario'] = family + '-drain-and-session-cleanup'
            await begin_scope(family, report['scenario'], store.keys.fabric_id)
            await sample_current_scope(current_scope)
            # End all live producer sessions; input/results clean up without remote reservations.
            for producer in lane_producers:
                await asyncio.to_thread(producer.close)
            await eventually(lambda: store.usage()['records'] == 0)
            require(
                store.usage()['reserved_items'] == 0,
                'drain_reservations',
            )
            before_stop = time.monotonic()
            await stop_sampled_runtime(runtime, sample_lock)
            report['scenarios'].append(
                {
                    'name': family + '-drain-and-session-cleanup',
                    'family': family,
                    'fabric': store.keys.fabric_id,
                    'status': 'passed',
                    'drain_seconds': time.monotonic() - before_stop,
                    'remaining_records': 0,
                    'remaining_reservations': 0,
                    'server_version': report['verified_stores'][family]['version'],
                    'runtime_counters': dict(runtime._counts),
                }
            )
            async with sample_lock:
                observe_scope_cpu(current_scope, process)
                current_scope['finished_at'] = time.monotonic()
                current_scope = None
        report['status'] = 'failed' if report['failures'] else 'passed'
    except Exception as error:
        report['status'] = 'failed'
        record_failure(
            report, error, phase=report['phase'], scenario=report['scenario']
        )
    finally:
        sample_stop.set()
        sample_results = await asyncio.gather(sampling, return_exceptions=True)
        if isinstance(sample_results[0], BaseException):
            report['status'] = 'failed'
            record_failure(
                report,
                sample_results[0],
                phase='sampler',
                scenario=sample_failure_context.get('scenario'),
            )
        for producer in producers:
            try:
                await asyncio.to_thread(producer.close)
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_producer', scenario=report['scenario']
                )
        for runtime in runtimes:
            try:
                if runtime._runner is not None:
                    await runtime.stop()
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_runtime', scenario=report['scenario']
                )
        # Only this run's generated namespaces, never any shared/test-fixture namespace.
        for family, store in zip(fixtures, stores):
            row = {
                'family': family,
                'fabric': store.keys.fabric_id,
                'remaining_keys': None,
                'remaining_records': None,
                'remaining_reservations': None,
            }
            report['cleanup'].append(row)
            try:
                keys = list(store.client.scan_iter(match=store.keys.prefix + '*'))
                if keys:
                    store.client.delete(*keys)
                row['remaining_keys'] = len(
                    list(store.client.scan_iter(match=store.keys.prefix + '*'))
                )
                usage = store.usage()
                row['remaining_records'] = usage['records']
                row['remaining_reservations'] = usage['reserved_items']
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_store', scenario=report['scenario']
                )
            finally:
                try:
                    store.close()
                except Exception as error:
                    record_failure(
                        report,
                        error,
                        phase='cleanup_store_close',
                        scenario=report['scenario'],
                    )
        RequestStore._invoke = original
        EndpointClient.execute = original_execute
        SnapshotReadBudget.before_read = original_before_read
        try:
            await http.aclose()
        except Exception as error:
            record_failure(
                report, error, phase='cleanup_http', scenario=report['scenario']
            )
        report['phase'] = 'measurement'
        report['transition_counts'] = dict(counters)
        report['counts'] = summarize_counts(
            counters,
            records,
            submitted=submissions['items'],
            provider_attempts=len(transport_ms),
        )
        try:
            after_cpu = process.cpu_times()
            elapsed = time.monotonic() - started
            report['measurements'] = {
                'elapsed_seconds': elapsed,
                'process_cpu_seconds': after_cpu.user
                + after_cpu.system
                - before_cpu.user
                - before_cpu.system,
                'resources': {
                    key: quantiles(values)
                    for key, values in resources.items()
                    if key != 'backlog_overlap'
                },
                'queue_wait_ms': quantiles(
                    [
                        (r['first_start'] - r['admitted']) * 1000
                        for r in records.values()
                        if 'first_start' in r and 'admitted' in r
                    ]
                ),
                'start_to_commit_ms': quantiles(
                    [
                        (r['finished'] - r['last_start']) * 1000
                        for r in records.values()
                        if 'finished' in r and 'last_start' in r
                    ]
                ),
                'end_to_end_ms': quantiles(
                    [
                        (r['delivered'] - r['admitted']) * 1000
                        for r in records.values()
                        if 'delivered' in r and 'admitted' in r
                    ]
                ),
                'commit_to_delivery_ms': quantiles(
                    [
                        (r['delivered'] - r['finished']) * 1000
                        for r in records.values()
                        if 'delivered' in r and 'finished' in r
                    ]
                ),
                'provider_ms': quantiles(transport_ms),
                'transport_categories': dict(transport_categories),
                'payload_bytes': sum(
                    r.get('payload_bytes', 0) for r in records.values()
                ),
                'result_bytes': sum(r.get('result_bytes', 0) for r in records.values()),
                'completed_per_second': sum(
                    r.get('terminal_state') == 'succeeded' for r in records.values()
                )
                / elapsed,
            }
            report['measurements'].update(
                aggregate_scope='mixed families and scenarios including setup and cleanup',
                process_memory_scope='combined local producer/dispatcher process RSS; not gateway-container memory',
                throughput_scope='whole-run completion rate including recovery and cleanup; not weighted lane throughput',
                resource_samples=resource_samples,
                by_scenario={
                    name: summarize_scope_measurements(
                        [scope], resource_samples, records, transport_rows
                    )
                    for name, scope in scopes.items()
                },
                by_family={
                    family: summarize_scope_measurements(
                        [
                            scope
                            for scope in scopes.values()
                            if scope['family'] == family
                        ],
                        resource_samples,
                        records,
                        transport_rows,
                    )
                    for family in fixtures
                },
            )
        except Exception as error:
            record_failure(
                report, error, phase='measurement', scenario=report['scenario']
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stores', required=True)
    parser.add_argument('--studio', required=True)
    parser.add_argument('--mock', required=True)
    parser.add_argument('--admin', required=True)
    parser.add_argument('--aimock-container', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--items', type=int, default=80)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(qualify(args))
    print(json.dumps({'run_id': report['run_id'], 'status': report['status']}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
