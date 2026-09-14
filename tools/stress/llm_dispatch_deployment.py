"""Fixed real gateway/scheduler/external Annotator deployment qualification."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PYTHON = '/home/gbugaj/dev/marieai/marie-ai/.venv/bin/python'


def require(condition: bool, check: str) -> None:
    if not condition:
        raise ValueError(check)


def safe_environment(root: Path, source: Path) -> dict[str, str]:
    return {
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'LANG': 'C.UTF-8',
        'PYTHONPATH': ':'.join(
            map(str, [source, *sorted(source.glob('packages/*/src'))])
        ),
        'PYTHONDONTWRITEBYTECODE': '1',
        'MARIE_DEFAULT_MOUNT': str(source),
        'MARIE_CACHE': str(root / 'cache'),
        'XDG_CACHE_HOME': str(root / 'cache'),
        'DOCKER_CONFIG': str(root / 'docker-config'),
        'HF_HOME': str(root / 'hf'),
        'JINA_LOG_LEVEL': 'WARNING',
        'MARIE_LOG_LEVEL': 'WARNING',
        'MARIE_LOG_USE_QUEUE': '0',
        'JINA_OPTOUT_TELEMETRY': '1',
        'NO_VERSION_CHECK': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'OTEL_SDK_DISABLED': 'true',
        'CUDA_VISIBLE_DEVICES': '',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MARIE_LEGACY_WORKING_DIR': 'false',
        'QUAL_DEPLOYMENT_ROOT': str(root),
        'TMPDIR': str(root / 'tmp'),
    }


def verify_output_files(writes: list[dict], root: Path) -> None:
    require(len(writes) == 9, 'output_count')
    require(len({r['task_id'] for r in writes}) == 9, 'output_tasks')
    require(len({r['path'] for r in writes}) == 9, 'output_paths')
    for row in writes:
        path = Path(row['path']).resolve()
        require(path.is_relative_to(root.resolve()), 'output_owned_path')
        require(path.is_file(), 'output_exists')
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
            'output_hash',
        )


def verify_worker(record: dict, parent_pid: int) -> Any:
    import psutil

    require(
        record['pid'] != parent_pid and record['ppid'] == parent_pid, 'worker_parent'
    )
    process = psutil.Process(record['pid'])
    require(process.ppid() == parent_pid, 'worker_parent')
    require(process.create_time() == record['created'], 'worker_start_identity')
    require(process.is_running(), 'worker_running')
    return process


class Commands:
    def __init__(self, root: Path, source: Path, env: dict[str, str]) -> None:
        self.root, self.source, self.env = root, source, env
        (root / 'tmp').mkdir(parents=True, exist_ok=True)
        self.directory = root / 'commands'
        self.directory.mkdir(exist_ok=True)
        self.background: dict[int, Any] = {}

    def run(
        self,
        label: str,
        argv: list[str],
        *,
        private: dict | None = None,
        background: bool = False,
    ) -> Any:
        number = len(list(self.directory.glob('*-command.json'))) + 1
        path = self.directory / f'{number:03d}-{label}-command.json'
        log = path.with_name(path.name.replace('-command.json', '.log'))
        private_keys = {
            key
            for key in (private or {})
            if any(
                word in key
                for word in (
                    'PASSWORD',
                    'API_KEY',
                    'SECRET',
                    'ACCESS_KEY',
                    'POSTGRES_USER',
                )
            )
        }
        record = {
            'command': argv,
            'cwd': str(self.source),
            'environment': self.env
            | {
                key: value
                for key, value in (private or {}).items()
                if key not in private_keys
            },
            'private_environment_names': sorted(private_keys),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'log': str(log),
        }
        save_json(path, record)
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            process = subprocess.Popen(
                argv,
                cwd=self.source,
                env=self.env | (private or {}),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=background,
            )
        record['pid'] = process.pid
        if background:
            self.background[process.pid] = process
            record['state'] = 'running'
            save_json(path, record)
            return process, path
        code = process.wait(timeout=300)
        record.update(
            exit_code=code, finished_at=datetime.now(timezone.utc).isoformat()
        )
        save_json(path, record)
        require(code == 0, f'command_{label}_exit_{code}')
        return log.read_text()

    def inspect(self, identity: str) -> dict:
        template = (
            '{"id":{{json .Id}},"image":{{json .Image}},'
            '"running":{{json .State.Running}},"pid":{{json .State.Pid}},'
            '"labels":{{json .Config.Labels}},'
            '"bindings":{{json .HostConfig.PortBindings}}}'
        )
        return json.loads(
            self.run('inspect', ['docker', 'inspect', '--format', template, identity])
        )


def verify_container_identity(row: dict, runtime: dict, service: str) -> None:
    planned = runtime['intended_services'][service]
    require(
        row['labels'].get('marie.test.owner') == runtime['project']
        and row['labels'].get('com.docker.compose.project') == runtime['project']
        and row['labels'].get('com.docker.compose.service') == service
        and row['image'] == planned['image']
        and row['bindings'] == planned['bindings'],
        'owned_container_identity',
    )


def stop_owned(root: Path, runtime: dict, commands: Commands, report: dict) -> None:
    report['container_cleanup'] = []
    report['container_discovery_complete'] = False
    report['running_owned_containers'] = []
    try:
        stop_sources(runtime, commands)
    except Exception as error:
        report['failures'].append(
            {'phase': 'stop_sources', 'type': type(error).__name__}
        )
    intended = runtime.get('intended_services', {})
    require(set(intended) == {'postgres', 'etcd', 'minio'}, 'persisted_service_intent')

    def discover() -> list[str]:
        result = commands.run(
            'discover-owned',
            [
                'docker',
                'ps',
                '--all',
                '--quiet',
                '--no-trunc',
                '--filter',
                'label=com.docker.compose.project=' + runtime['project'],
            ],
        )
        return sorted(set(result.split()))

    candidates = {row['id'] for row in runtime['containers'].values()}
    discovery_ok = True
    try:
        candidates.update(discover())
    except Exception as error:
        discovery_ok = False
        report['failures'].append(
            {'phase': 'discover_before_stop', 'type': type(error).__name__}
        )
    seen = {}
    for identity in sorted(candidates):
        try:
            current = commands.inspect(identity)
            require(current['id'] == identity, 'container_id')
            service = current['labels'].get('com.docker.compose.service')
            require(service in intended, 'intended_service')
            verify_container_identity(current, runtime, service)
            require(
                service not in seen or seen[service] == identity, 'single_owned_service'
            )
            seen[service] = identity
            runtime['containers'][service] = current
            save_json(root / 'private-runtime.json', runtime)
            if current['running']:
                commands.run(
                    'stop-owned-' + service,
                    ['docker', 'stop', '--time', '10', identity],
                )
            stopped = commands.inspect(identity)
            verify_container_identity(stopped, runtime, service)
            require(
                stopped['id'] == identity and not stopped['running'],
                'container_stopped',
            )
            runtime['containers'][service] = stopped
            save_json(root / 'private-runtime.json', runtime)
            report['container_cleanup'].append(
                {'name': service, 'id': identity, 'stopped': True}
            )
        except Exception as error:
            report['failures'].append(
                {'phase': 'stop_container', 'type': type(error).__name__}
            )
    try:
        after = discover()
        require(set(after) == candidates, 'complete_discovery')
        for identity in after:
            current = commands.inspect(identity)
            service = current['labels'].get('com.docker.compose.service')
            require(
                service in intended and current['id'] == identity, 'post_stop_identity'
            )
            verify_container_identity(current, runtime, service)
            if current['running']:
                report['running_owned_containers'].append(identity)
        report['container_discovery_complete'] = discovery_ok
    except Exception as error:
        report['failures'].append(
            {'phase': 'discover_after_stop', 'type': type(error).__name__}
        )
    require(
        not report['failures']
        and report['container_discovery_complete']
        and not report['running_owned_containers'],
        'complete_owned_cleanup',
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def prepare(root: Path, source: Path, fixtures: dict, commands: Commands) -> dict:
    import yaml

    token = secrets.token_hex(6)
    ports = {
        name: free_port()
        for name in ('postgres', 'etcd', 'minio', 'http', 'grpc', 'executor')
    }
    require(len(set(ports.values())) == len(ports), 'unique_ports')
    private = {
        'POSTGRES_USER': 'qual_' + token,
        'POSTGRES_PASSWORD': secrets.token_hex(24),
        'POSTGRES_DATABASE': 'postgres',
        'QUAL_GATEWAY_API_KEY': 'mas_' + secrets.token_hex(27),
        'QUAL_AIMOCK_API_KEY': secrets.token_hex(24),
        'S3_ACCESS_KEY_ID': 'qual' + token,
        'S3_SECRET_ACCESS_KEY': secrets.token_hex(24),
    }
    images = {
        'postgres': 'marie-postgres-documentdb:17-0.103.0-repack-1.5.3',
        'etcd': 'quay.io/coreos/etcd:v3.7.0',
        'minio': 'minio/minio:latest',
    }
    image_ids = {}
    for name, image in images.items():
        image_ids[name] = commands.run(
            'image-' + name,
            ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
        ).strip()
    project = 'marie-deployment-' + token
    labels = {'marie.test.owner': project}
    services = {
        'postgres': {
            'image': images['postgres'],
            'ports': [f'127.0.0.1:{ports["postgres"]}:5432'],
            'environment': {
                'POSTGRES_USER': '${POSTGRES_USER}',
                'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}',
                'POSTGRES_DB': 'postgres',
            },
            'command': [
                'postgres',
                '-c',
                'max_connections=100',
                '-c',
                'shared_preload_libraries=pg_stat_statements,pg_cron,pg_documentdb_core,pg_documentdb',
            ],
            'volumes': ['postgres-data:/var/lib/postgresql/data'],
            'shm_size': '256m',
        },
        'etcd': {
            'image': images['etcd'],
            'ports': [f'127.0.0.1:{ports["etcd"]}:2379'],
            'command': [
                '/usr/local/bin/etcd',
                '--name=single',
                '--data-dir=/etcd-data',
                '--listen-client-urls=http://0.0.0.0:2379',
                '--advertise-client-urls=http://127.0.0.1:2379',
                '--listen-peer-urls=http://0.0.0.0:2380',
                '--initial-advertise-peer-urls=http://127.0.0.1:2380',
                '--initial-cluster=single=http://127.0.0.1:2380',
            ],
            'volumes': ['etcd-data:/etcd-data'],
        },
        'minio': {
            'image': images['minio'],
            'ports': [f'127.0.0.1:{ports["minio"]}:9000'],
            'command': ['server', '/data'],
            'environment': {
                'MINIO_ROOT_USER': '${S3_ACCESS_KEY_ID}',
                'MINIO_ROOT_PASSWORD': '${S3_SECRET_ACCESS_KEY}',
            },
            'volumes': ['minio-data:/data'],
        },
    }
    for service in services.values():
        service.update(labels=labels, restart='no')
    compose = root / 'compose.yml'
    compose.write_text(
        yaml.safe_dump(
            {'services': services, 'volumes': {name + '-data': {} for name in services}}
        )
    )
    runtime = {
        'project': project,
        'ports': ports,
        'private': private,
        'fixtures': fixtures,
        'fabric': 'deployment-' + token,
        'bucket': 'deployment-' + token,
        'containers': {},
        'processes': [],
        'intended_services': {
            name: {
                'image': image_ids[name],
                'bindings': {
                    str(port) + '/tcp': [
                        {'HostIp': '127.0.0.1', 'HostPort': str(ports[name])}
                    ]
                },
            }
            for name, port in [('postgres', 5432), ('etcd', 2379), ('minio', 9000)]
        },
    }
    os.chmod(compose, 0o600)
    save_json(root / 'private-runtime.json', runtime)
    os.chmod(root / 'private-runtime.json', 0o600)
    command = [
        'docker',
        'compose',
        '--env-file',
        '/dev/null',
        '-p',
        project,
        '-f',
        str(compose),
    ]
    commands.run('compose-config', command + ['config', '--quiet'], private=private)
    commands.run('compose-up', command + ['up', '-d'], private=private)
    for name in services:
        identity = commands.run(
            'compose-id', command + ['ps', '-q', name], private=private
        ).strip()
        row = commands.inspect(identity)
        require(
            row['labels']['marie.test.owner'] == project and row['running'],
            'fresh_owned_container',
        )
        verify_container_identity(row, runtime, name)
        runtime['containers'][name] = row
        save_json(root / 'private-runtime.json', runtime)
    save_json(root / 'private-runtime.json', runtime)
    return runtime


def configurations(root: Path, runtime: dict, provider: str) -> dict[str, str]:
    import yaml

    ports, private = runtime['ports'], runtime['private']

    def env(key: str) -> str:
        return '${{ ENV.' + key + ' }}'

    postgres = dict(
        provider='postgresql',
        hostname='127.0.0.1',
        port=ports['postgres'],
        username=env('POSTGRES_USER'),
        password=env('POSTGRES_PASSWORD'),
        database='postgres',
        schema='marie_scheduler',
    )
    s3 = dict(
        enabled=True,
        metadata_only=False,
        endpoint_url=f'http://127.0.0.1:{ports["minio"]}',
        access_key_id=env('S3_ACCESS_KEY_ID'),
        secret_access_key=env('S3_SECRET_ACCESS_KEY'),
        bucket_name=runtime['bucket'],
        region='us-east-1',
        insecure=True,
        addressing_style='path',
    )
    common = dict(
        host='127.0.0.1',
        env_file='/dev/null',
        prefetch=1,
        tracing=False,
        metrics=False,
        event_tracking=True,
        discovery=True,
        discovery_host='127.0.0.1',
        discovery_port=ports['etcd'],
        discovery_service_name='gateway/marie',
        discovery_namespace='marie',
        discovery_lease_sec=6,
        discovery_heartbeat_sec=1.5,
        discovery_timeout_sec=10,
        discovery_retry_times=5,
        discovery_kwargs={'endpoints': f'127.0.0.1:{ports["etcd"]}'},
        workspace=str(root / 'workspace'),
    )
    gateway_with = common | dict(
        port=[ports['grpc'], ports['http']],
        protocol=['GRPC', 'HTTP'],
        kv_store_kwargs=postgres
        | dict(default_table='kv_store_worker', max_pool_size=5, max_connections=5),
        job_scheduler_kwargs=postgres
        | dict(
            desired_state_worker_count=1,
            desired_state_max_pending=8,
            max_pool_size=5,
            max_connections=5,
            queue_names=['gen5_extract'],
            query_planners=dict(
                watch_wheels=False,
                wheel_directories=[],
                planners=[
                    dict(
                        name='mock_planners',
                        py_module='marie.query_planner.mock_query_plans',
                    )
                ],
            ),
        ),
        expose_endpoints={
            '/annotator/llm': {'methods': ['POST']},
            '/status': {'methods': ['POST']},
        },
        llm_queue=dict(
            queue_contract_version='v3',
            fabric_group_id=runtime['fabric'],
            llm_dispatch=dict(
                policy_source='static',
                policy='fifo',
                total_concurrent_dispatch=1,
                endpoints=[
                    dict(
                        endpoint_id='aimock',
                        base_url=provider,
                        credential_env='QUAL_AIMOCK_API_KEY',
                        allow_loopback=True,
                        allow_private=False,
                        execution_limit=1,
                        execution_bytes=16777216,
                        call_timeout_seconds=30,
                        max_response_bytes=1048576,
                    )
                ],
                lanes=[
                    dict(
                        pool_id='document-small',
                        endpoint_id='aimock',
                        revision='qualification-r1',
                        execution_limit=1,
                        execution_bytes=16777216,
                        quantum=1,
                        min_concurrent=0,
                        max_burst_per_visit=1,
                    )
                ],
                limits=dict(
                    max_active_items=16,
                    max_payload_bytes=33554432,
                    max_inline_payload_bytes=16777216,
                    max_records=64,
                    max_storage_bytes=33554432,
                    max_ready_ids=64,
                    max_execution_items=1,
                    max_execution_bytes=16777216,
                    max_producers=4,
                    max_routes=4,
                    max_endpoints=4,
                ),
            ),
        ),
    )
    setup = dict(
        toast={'grpc': {'enabled': False}},
        storage={'s3': s3},
        llm_tracking={'enabled': False},
    )
    gateway = (
        {'jtype': 'Flow', 'version': '1'}
        | setup
        | {
            'prefetch': 1,
            'auth': {
                'keys': [
                    dict(
                        name='qualification',
                        api_key=env('QUAL_GATEWAY_API_KEY'),
                        enabled=True,
                        scopes=['runtime-observability'],
                        allowed_fabrics=[runtime['fabric']],
                    )
                ]
            },
            'with': gateway_with,
            'gateway': {
                'uses': {
                    'jtype': 'MarieGateway',
                    'with': {
                        'env_file': '/dev/null',
                        'llm_queue': gateway_with['llm_queue'],
                    },
                }
            },
            'executors': [
                dict(
                    name='annotator_llm',
                    external=True,
                    host='127.0.0.1',
                    port=ports['executor'],
                )
            ],
        }
    )
    executor = (
        {'jtype': 'Deployment'}
        | setup
        | {
            'with': common
            | dict(
                name='annotator_llm',
                port=ports['executor'],
                protocol='GRPC',
                kv_store_kwargs=postgres
                | dict(
                    default_table='kv_store_worker', max_pool_size=5, max_connections=5
                ),
                replicas=1,
                timeout_ready=120000,
                uses={
                    'jtype': 'DocumentAnnotatorLLMExecutor',
                    'metas': {'py_modules': ['marie.executor.extract']},
                    'with': {
                        'storage': {'s3': s3},
                        'llm_tracking': {'enabled': False},
                        'health': {'gpu_monitor_enabled': False},
                    },
                },
            )
        }
    )
    for name, config in [('gateway', gateway), ('executor', executor)]:
        path = root / (name + '.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        os.chmod(path, 0o600)
    config_root = root / 'generated-config/extract'
    base = config_root / 'base'
    layout = config_root / 'TID-mock-llm/annotator'
    base.mkdir(parents=True, exist_ok=True)
    layout.mkdir(parents=True, exist_ok=True)
    (base / 'base-config.yml').write_text('annotators: {}\ngrounding: {}\n')
    (base / 'field-config.yml').write_text('fields: {}\n')
    (layout / 'config.yml').write_text(
        yaml.safe_dump(
            dict(
                layout_id='mock-llm',
                annotators={
                    'mock-llm': dict(
                        annotator_type='llm',
                        mode='per-page',
                        parser_name='default',
                        validators=[],
                        model_config=dict(
                            model_name='gpt-5.2-mock',
                            multimodal=True,
                            prompt_path='./qualification.j2',
                            system_prompt_text='Synthetic page qualification',
                            temperature=0.0,
                            expect_output='none',
                            mini_batch_size=9,
                            min_pixels=3136,
                            max_pixels=12544,
                            max_tokens=128,
                        ),
                    )
                },
                grounding={},
            )
        )
    )
    (layout / 'qualification.j2').write_text("{{ OCR_DATA.split('|')[-1].strip() }}")
    queue = runtime['fixtures']['valkey']['url']
    return private | dict(
        LLM_QUEUE_ENABLED='true',
        LLM_QUEUE_CONTRACT_VERSION='v3',
        LLM_QUEUE_URL=queue,
        LLM_QUEUE_FABRIC_GROUP_ID=runtime['fabric'],
        LLM_QUEUE_POOL_ID='document-small',
        LLM_QUEUE_PRODUCER_TTL_SECONDS='12',
        LLM_QUEUE_PRODUCER_REFRESH_INTERVAL_SECONDS='3',
        OPENAI_API_KEY=private['QUAL_AIMOCK_API_KEY'],
        OPENAI_API_BASE=provider,
        MARIE_S3_BUCKET=runtime['bucket'],
        S3_BUCKET_NAME=runtime['bucket'],
    )


def probes(root: Path) -> list[dict]:
    path = root / 'probe.jsonl'
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.endswith('}')
    ]


def source_inventory(source: Path) -> list[dict[str, str]]:
    paths = set()
    for directory in [
        source / 'marie',
        source / 'marie_server',
        source / 'tools/stress',
        *source.glob('packages/*/src'),
    ]:
        paths.update(directory.rglob('*.py'))
        paths.update(directory.rglob('*.lua'))
    rows = []
    for path in sorted(paths):
        if path.is_file() and not any(
            p == 'secrets' or p.startswith('.env') for p in path.parts
        ):
            rows.append(
                {
                    'path': str(path.relative_to(source)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    require(bool(rows), 'nonempty_canonical_source')
    return rows


def startup_manifest(root: Path, source: Path) -> str:
    rows = source_inventory(source)
    manifest = {
        'files': rows,
        'digest': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        'config': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('gateway.yml', 'executor.yml')
        },
    }
    path = root / f'startup-source-{time.time_ns()}.json'
    save_json(path, manifest)
    return str(path)


def stop_sources(runtime: dict, commands: Commands) -> None:
    import psutil

    for row in reversed(runtime['processes']):
        try:
            process = psutil.Process(row['pid'])
            require(process.create_time() == row['created'], 'source_stop_identity')
            require(os.getpgid(process.pid) == process.pid, 'source_stop_group')
            if process.status() != 'zombie':
                commands.run(
                    'stop-source', ['kill', '-TERM', '--', '-' + str(process.pid)]
                )
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    require(
                        process.create_time() == row['created'], 'source_stop_identity'
                    )
                    commands.run(
                        'kill-source', ['kill', '-KILL', '--', '-' + str(process.pid)]
                    )
            audit = Path(row['audit'])
            record = json.loads(audit.read_text())
            record.update(
                state='stopped', stopped_at=datetime.now(timezone.utc).isoformat()
            )
            owned = commands.background.get(row['pid'])
            if owned is not None:
                record['exit_code'] = owned.wait(timeout=5)
            else:
                record['exit_code_note'] = (
                    'Detached startup process; exact exit status unavailable to later controller'
                )
            save_json(audit, record)
        except psutil.NoSuchProcess:
            continue


def gateway_ready(health: dict, runtime: dict, observations: list[dict]) -> bool:
    result = health.get('result', {})
    discovery = next(
        (
            item.get('details', {})
            for item in result.get('dependencies', [])
            if item['name'] == 'discovery'
        ),
        {},
    )
    launch = next(
        row for row in reversed(runtime['processes']) if row['role'] == 'gateway'
    )
    return (
        result.get('overall_state') == 'ok'
        and result.get('partial') is False
        and discovery.get('registered') == 1
        and discovery.get('ready') == 1
        and any(
            item['event'] == 'gateway_dispatch'
            and item.get('fabric') == runtime['fabric']
            and item['ppid'] == launch['pid']
            and item['created'] >= launch['created']
            and item['at'] >= launch['created']
            for item in observations
        )
    )


async def start_sources(
    root: Path,
    source: Path,
    runtime: dict,
    commands: Commands,
    report: dict,
    provider: str | None = None,
) -> None:
    import httpx
    import psutil

    provider = (
        provider
        or 'http://127.0.0.1:'
        + runtime['fixtures']['valkey']['compose_ports']['QUAL_MOCK_PORT']
        + '/v1'
    )
    child_env = configurations(root, runtime, provider)
    for attempt in range(60):
        try:
            commands.run(
                'postgres-ready',
                [
                    'docker',
                    'exec',
                    runtime['containers']['postgres']['id'],
                    'pg_isready',
                    '-h',
                    '127.0.0.1',
                    '-q',
                ],
            )
            break
        except ValueError:
            if attempt == 59:
                raise
            await asyncio.sleep(0.5)
    commands.run(
        'postgres-canonical-role',
        [
            PYTHON,
            '-c',
            "import os, psycopg; c=psycopg.connect(host='127.0.0.1',port=os.environ['POSTGRES_PORT'],dbname='postgres',user=os.environ['POSTGRES_USER'],password=os.environ['POSTGRES_PASSWORD'],autocommit=True); cur=c.cursor(); cur.execute(\"SELECT 1 FROM pg_roles WHERE rolname='postgres'\"); exists=cur.fetchone(); cur.execute('CREATE ROLE postgres NOLOGIN') if not exists else None; c.close()",
        ],
        private=runtime['private']
        | {'POSTGRES_PORT': str(runtime['ports']['postgres'])},
    )
    fixture = runtime['fixtures']['valkey']
    store_identity = commands.inspect(fixture['container_id'])
    require(
        store_identity['running']
        and store_identity['image'] == fixture['image_id']
        and store_identity['labels']['marie.test.owner'] == fixture['owner_label']
        and store_identity['labels']['com.docker.compose.project']
        == fixture['compose_project'],
        'reused_valkey_identity',
    )
    report['verified_valkey'] = store_identity
    manifest = startup_manifest(root, source)
    report['startup_source'] = manifest
    for row in runtime['processes']:
        try:
            prior = psutil.Process(row['pid'])
            require(
                prior.create_time() != row['created'] or prior.status() == 'zombie',
                'prior_source_must_stop',
            )
        except psutil.NoSuchProcess:
            pass
    for role in ('executor', 'gateway'):
        argv = [PYTHON, '-m', 'tools.stress.llm_dispatch_deployment_child', role]
        role_env = dict(child_env)
        if role == 'executor' and runtime.get('oom_scope'):
            argv = [
                'systemd-run',
                '--user',
                '--scope',
                '--unit=' + runtime['oom_scope'],
                '--property=MemoryMax=8G',
                '--property=MemorySwapMax=0',
                *argv,
            ]
            role_env.update(
                QUAL_OOM_SCOPE=runtime['oom_scope'],
                XDG_RUNTIME_DIR='/run/user/' + str(os.getuid()),
                DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'
                + str(os.getuid())
                + '/bus',
            )
        process, audit = commands.run(
            'source-' + role, argv, private=role_env, background=True
        )
        runtime['processes'].append(
            dict(
                role=role,
                pid=process.pid,
                created=psutil.Process(process.pid).create_time(),
                audit=str(audit),
                source_manifest=manifest,
            )
        )
        save_json(root / 'private-runtime.json', runtime)
    deadline = time.monotonic() + 100
    async with httpx.AsyncClient(timeout=3) as http:
        while time.monotonic() < deadline:
            for row in runtime['processes'][-2:]:
                process = psutil.Process(row['pid'])
                require(
                    process.is_running() and process.status() != 'zombie',
                    'source_process_alive_' + row['role'],
                )
            try:
                response = await http.get(
                    f'http://127.0.0.1:{runtime["ports"]["http"]}/api/operations/health',
                    headers={
                        'Authorization': 'Bearer '
                        + runtime['private']['QUAL_GATEWAY_API_KEY']
                    },
                )
                if response.status_code == 200:
                    health = response.json()
                    report['last_health'] = health
                    if not gateway_ready(health, runtime, probes(root)):
                        await asyncio.sleep(0.5)
                        continue
                    if runtime.get('oom_scope'):
                        launch = runtime['processes'][-2]
                        bootstrap = [
                            item
                            for item in probes(root)
                            if item['event'] == 'bootstrap'
                            and item['role'] == 'executor'
                            and item['at'] >= launch['created']
                        ][-1]
                        require(
                            bootstrap['pid'] == launch['pid']
                            or bootstrap['ppid'] == launch['pid'],
                            'scoped_bootstrap_parent',
                        )
                        launch['deployment_pid'] = bootstrap['pid']
                        save_json(root / 'private-runtime.json', runtime)
                    report['health'] = health
                    report['status'] = 'ready_not_qualified'
                    report['processes'] = runtime['processes']
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError('Gateway readiness timeout')


async def invoke_nine(root: Path, runtime: dict, report: dict) -> None:
    import boto3
    import httpx
    from botocore.config import Config
    from PIL import Image, ImageDraw

    ref = 'synthetic-' + secrets.token_hex(6)
    filename = ref + '.tif'
    images = []
    ocr = []
    for index in range(9):
        frame = Image.new('RGB', (112, 112), (220, 220 - index * 10, 200))
        ImageDraw.Draw(frame).text((10, 10), str(index), fill=(0, 0, 0))
        images.append(frame)
        ocr.append(
            dict(
                meta=dict(
                    lines=[1],
                    page=index,
                    imageSize=dict(width=112, height=112),
                    lines_bboxes=[[10, 10, 20, 10]],
                ),
                lines=[
                    dict(
                        line=1,
                        wordids=[1],
                        text=str(index),
                        bbox=[10, 10, 20, 10],
                        confidence=1.0,
                    )
                ],
                words=[
                    dict(
                        id=1,
                        line=1,
                        word_index=0,
                        text=str(index),
                        box=[10, 10, 20, 10],
                        confidence=1.0,
                    )
                ],
            )
        )
    asset = root / filename
    images[0].save(asset, save_all=True, append_images=images[1:])
    for frame in images:
        frame.close()
    prefix = f'llm-qualification/{ref}'
    uri = f's3://{runtime["bucket"]}/{prefix}/{filename}'
    metadata = dict(ref_id=filename, ref_type='llm-qualification', pages=9, ocr=ocr)
    s3 = boto3.client(
        's3',
        endpoint_url=f'http://127.0.0.1:{runtime["ports"]["minio"]}',
        aws_access_key_id=runtime['private']['S3_ACCESS_KEY_ID'],
        aws_secret_access_key=runtime['private']['S3_SECRET_ACCESS_KEY'],
        region_name='us-east-1',
        config=Config(s3={'addressing_style': 'path'}),
    )
    existing = [item['Name'] for item in s3.list_buckets()['Buckets']]
    if runtime['bucket'] not in existing:
        s3.create_bucket(Bucket=runtime['bucket'])
    s3.upload_file(str(asset), runtime['bucket'], prefix + '/' + filename)
    s3.put_object(
        Bucket=runtime['bucket'],
        Key=prefix + '/' + filename + '.meta.json',
        Body=json.dumps(metadata).encode(),
    )
    s3.close()
    start = time.time()
    request_id = secrets.token_hex(16)
    envelope = {
        'data': [{'id': request_id + '-0', 'text': 'synthetic qualification'}],
        'parameters': {
            'invoke_action': {
                'action_type': 'command',
                'command': 'job',
                'action': 'submit',
                'name': 'gen5_extract',
                'api_key': runtime['private']['QUAL_GATEWAY_API_KEY'],
                'metadata': dict(
                    planner='mock_annotator_llm',
                    project_id=runtime['project'],
                    ref_type='llm-qualification',
                    ref_id=filename,
                    policy='allow_all',
                    pool_id='document-small',
                    uri=uri,
                ),
            }
        },
        'header': {'requestId': request_id, 'targetExecutor': ''},
    }
    base = f'http://127.0.0.1:{runtime["ports"]["http"]}'
    async with httpx.AsyncClient(
        timeout=10,
        headers={
            'Authorization': 'Bearer ' + runtime['private']['QUAL_GATEWAY_API_KEY']
        },
    ) as http:
        response = await http.post(base + '/api/v1/invoke', json=envelope)
        require(response.status_code == 200, 'invoke_http_status')
        reply = response.json()
        parameters = reply.get('parameters', {})
        require(
            parameters.get('status') == 'ok' and parameters.get('job_id'),
            'invoke_acknowledgement',
        )
        dag = parameters['job_id']
        report['submission'] = dict(
            request_id=request_id,
            dag_id=dag,
            http_status=response.status_code,
            acknowledgement_received=True,
            asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        )
        save_json(root / 'submission.json', report['submission'])
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            events = [
                row
                for row in probes(root)
                if row['at'] >= start and row.get('dag_id') == dag
            ]
            ended = [
                row
                for row in events
                if row['event'] in ('executor_end', 'executor_error')
            ]
            if ended:
                report['ordinary_executor_result'] = ended
                report['ordinary_writes'] = [
                    row for row in events if row['event'] == 'write'
                ]
                report['ordinary_admissions'] = [
                    row for row in events if row['event'] == 'admitted'
                ]
                history = await http.get(
                    base + '/api/operations/execution-history', params={'dag_id': dag}
                )
                report['ordinary_history'] = history.json()
                require(
                    ended[-1]['event'] == 'executor_end'
                    and ended[-1]['status'] == 'success',
                    'ordinary_executor_success',
                )
                verify_output_files(report['ordinary_writes'], root)
                report['status'] = 'ordinary_annotation_passed_not_qualified'
                return
            await asyncio.sleep(0.5)
        report['ordinary_probes'] = [
            row
            for row in probes(root)
            if row['at'] >= start and row.get('dag_id') == dag
        ]
        raise RuntimeError('Ordinary annotation timed out')


async def wait_until(predicate: Any, seconds: float = 45) -> Any:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(0.05)
    raise TimeoutError('Observed deployment boundary not reached')


async def qualify_s03(
    root: Path, source: Path, runtime: dict, commands: Commands, report: dict
) -> None:
    import httpx
    from marie.engine.llm_queue.store import OwnerToken, RequestStore

    from tools.stress.llm_dispatch_failure_scenarios import OwnedFailureListener

    ports = runtime['fixtures']['valkey']['compose_ports']
    started = time.time()
    store = None
    call = None
    async with httpx.AsyncClient(timeout=10) as http:

        async def forward(body: dict) -> dict:
            response = await http.post(
                'http://127.0.0.1:' + ports['QUAL_MOCK_PORT'] + '/v1/chat/completions',
                json=body,
                headers={
                    'Authorization': 'Bearer '
                    + runtime['private']['QUAL_AIMOCK_API_KEY']
                },
            )
            response.raise_for_status()
            return response.json()

        async def count() -> int:
            response = await http.get(
                'http://127.0.0.1:' + ports['QUAL_ADMIN_PORT'] + '/fault-profile'
            )
            response.raise_for_status()
            return response.json()['requestCount']

        seam = OwnedFailureListener('S03', forward)
        row = {'name': 'S03', 'status': 'failed'}
        report['scenarios'].append(row)
        try:
            await seam.start()
            stop_sources(runtime, commands)
            runtime['fabric'] = runtime['project'] + '-s03-' + secrets.token_hex(4)
            row['fabric'] = runtime['fabric']
            await start_sources(root, source, runtime, commands, report, seam.url)
            report['status'] = 'failed'
            store = RequestStore.for_producer(
                runtime['fixtures']['valkey']['url'], fabric_id=runtime['fabric']
            )
            await wait_until(lambda: store.resolve_route('document-small'))
            gateway = [
                item
                for item in probes(root)
                if item['event'] == 'gateway_dispatch' and item['at'] >= started
            ][-1]
            owner_value = await wait_until(lambda: store.client.get(store.keys.owner))
            identity, generation = owner_value.rsplit(':', 1)
            owner = OwnerToken(identity, int(generation))

            async def gate(opened: bool) -> None:
                await asyncio.to_thread(
                    store.configure_endpoint,
                    owner,
                    'aimock',
                    execution_limit=1,
                    execution_bytes=16777216,
                    gate_open=opened,
                )

            seam.before_first_response = lambda: gate(False)
            before_count = await count()
            case_started = time.time()
            events = lambda kind: [
                item
                for item in probes(root)
                if item['at'] >= case_started and item['event'] == kind
            ]
            call = asyncio.create_task(invoke_nine(root, runtime, report))
            await wait_until(
                lambda: (
                    len(events('write')) == 1
                    and len(events('admitted')) == 9
                    and events('pending')
                    and len(events('pending')[-1]['attempts']) == 8
                )
            )
            first = events('write')[0]
            producer = events('producer')[-1]
            child = verify_worker(producer, runtime['processes'][-2]['pid'])
            row['before_outage'] = {
                'first_output': first,
                'producer': producer,
                'admissions': events('admitted'),
                'provider_delta': await count() - before_count,
            }
            require(
                row['before_outage']['provider_delta'] == 1 and len(seam.records) == 1,
                'first_provider_receipt',
            )
            row['before_history'] = await inspect_runtime(root, runtime)
            await seam.stop_listener()
            await gate(True)
            await wait_until(
                lambda: store.endpoint_status('aimock')['circuit'] == 'open'
            )
            row['during_outage'] = {
                'pending': events('pending')[-1],
                'endpoint': store.endpoint_status('aimock'),
                'caller_pending': not call.done(),
                'child_alive': child.is_running(),
                'history': await inspect_runtime(root, runtime),
            }
            require(
                not call.done()
                and len(row['during_outage']['pending']['attempts']) == 8,
                'same_active_call',
            )
            require(
                hashlib.sha256(Path(first['path']).read_bytes()).hexdigest()
                == first['sha256'],
                'first_output_preserved',
            )
            await gate(False)
            await wait_until(
                lambda: int(store.endpoint_status('aimock')['reserved_items'] or 0) == 0
            )
            await seam.start()
            await gate(True)
            await call
            require(
                not seam.errors and len(seam.records) == 9, 'provider_unique_receipts'
            )
            require(
                sorted(item['index'] for item in seam.records) == list(range(9)),
                'provider_unique_indices',
            )
            require(await count() - before_count == 9, 'provider_count_delta')
            require(events('producer') == [producer], 'same_producer_process')
            verify_output_files(events('write'), root)
            require(
                hashlib.sha256(Path(first['path']).read_bytes()).hexdigest()
                == first['sha256'],
                'first_output_unchanged',
            )
            row.update(
                status='passed',
                receipts=seam.records,
                writes=events('write'),
                after_history=await inspect_runtime(root, runtime),
            )
            assert_scheduler_recovery(
                row,
                report['ordinary_executor_result'][-1]['job_id'],
                report['submission']['dag_id'],
            )
            report['status'] = 's03_passed_not_death_qualified'
        finally:
            if call and not call.done():
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            await seam.close()
            if store:
                store.close()


def assert_scheduler_recovery(row: dict, job: str, dag: str) -> None:
    phases = [
        row['before_history'],
        row['during_outage']['history'],
        row['after_history'],
    ]
    attempts = [
        [
            item
            for item in phase['attempts']['body']['result']['items']
            if item['job_id'] == job
        ]
        for phase in phases
    ]
    require(all(len(items) == 1 for items in attempts), 'one_scheduler_attempt')
    require(
        len({items[0]['run_attempt_id'] for items in attempts}) == 1,
        'same_scheduler_attempt',
    )
    require(
        all(
            items[0]['state'] == 'activated' and items[0]['terminal_at'] is None
            for items in attempts[:2]
        ),
        'scheduler_attempt_active_during_outage',
    )
    final = attempts[-1][0]
    require(
        final['state'] == 'completed'
        and final['terminal_status'] == 'SUCCEEDED'
        and final['terminal_accepted'] is True,
        'scheduler_attempt_succeeded',
    )
    dags = [
        item
        for item in phases[-1]['dags']['body']['result']['items']
        if item['id'] == dag
    ]
    require(
        len(dags) == 1 and dags[0]['state'] == 'completed', 'scheduler_dag_completed'
    )


async def qualify_death(
    root: Path,
    source: Path,
    runtime: dict,
    commands: Commands,
    report: dict,
    *,
    oom: bool = False,
) -> None:
    import httpx
    import psutil
    from marie.engine.llm_queue.store import OwnerToken, RequestStore

    from tools.stress.llm_dispatch_failure_scenarios import OwnedFailureListener

    ports = runtime['fixtures']['valkey']['compose_ports']
    row = {'name': 'OOM' if oom else 'SIGKILL', 'status': 'failed'}
    report['scenarios'].append(row)
    store = None
    call = None
    async with httpx.AsyncClient(timeout=10) as http:

        async def forward(body: dict) -> dict:
            response = await http.post(
                'http://127.0.0.1:' + ports['QUAL_MOCK_PORT'] + '/v1/chat/completions',
                json=body,
                headers={
                    'Authorization': 'Bearer '
                    + runtime['private']['QUAL_AIMOCK_API_KEY']
                },
            )
            response.raise_for_status()
            return response.json()

        seam = OwnedFailureListener('S03', forward)
        try:
            await seam.start()
            stop_sources(runtime, commands)
            runtime['fabric'] = runtime['project'] + '-death-' + secrets.token_hex(4)
            if oom:
                runtime['oom_scope'] = runtime['project'] + '-executor-oom.scope'
            row['fabric'] = runtime['fabric']
            await start_sources(root, source, runtime, commands, report, seam.url)
            report['status'] = 'failed'
            store = RequestStore.for_producer(
                runtime['fixtures']['valkey']['url'], fabric_id=runtime['fabric']
            )
            await wait_until(lambda: store.resolve_route('document-small'))
            identity, generation = store.client.get(store.keys.owner).rsplit(':', 1)
            owner = OwnerToken(identity, int(generation))

            async def gate(opened: bool) -> None:
                await asyncio.to_thread(
                    store.configure_endpoint,
                    owner,
                    'aimock',
                    execution_limit=1,
                    execution_bytes=16777216,
                    gate_open=opened,
                )

            seam.before_first_response = lambda: gate(False)
            started = time.time()
            events = lambda kind: [
                item
                for item in probes(root)
                if item['at'] >= started and item['event'] == kind
            ]
            original_report = {}
            call = asyncio.create_task(invoke_nine(root, runtime, original_report))
            await wait_until(
                lambda: (
                    len(events('write')) == 1
                    and len(events('admitted')) == 9
                    and events('pending')
                    and len(events('pending')[-1]['attempts']) == 8
                )
            )
            producer = events('producer')[-1]
            executor_parent = runtime['processes'][-2]
            gateway_parent = runtime['processes'][-1]
            worker = verify_worker(
                producer, executor_parent.get('deployment_pid', executor_parent['pid'])
            )
            gateway = psutil.Process(gateway_parent['pid'])
            gateway_children = [
                (child.pid, child.create_time()) for child in gateway.children()
            ]
            attempts = [item['attempt_id'] for item in events('admitted')]
            row.update(
                producer=producer,
                gateway_parent=gateway_parent,
                gateway_children=gateway_children,
                before_history=await inspect_runtime(root, runtime),
                admissions=events('admitted'),
            )
            # Exact PID, creation time, and direct Deployment parent were checked above.
            if oom:
                pressure = [
                    item for item in events('oom_ready') if item['pid'] == worker.pid
                ]
                require(len(pressure) == 1, 'scoped_pressure_observer')
                cgroup = Path(pressure[0]['cgroup'])
                require(cgroup.name == runtime['oom_scope'], 'owned_oom_scope')
                require(
                    (cgroup / 'memory.max').read_text().strip() == '8589934592',
                    'oom_limit',
                )
                require(
                    (cgroup / 'memory.swap.max').read_text().strip() == '0',
                    'oom_swap_limit',
                )
                require(
                    (cgroup / 'memory.oom.group').read_text().strip() == '0',
                    'oom_not_group_kill',
                )
                require(
                    str(worker.pid) in (cgroup / 'cgroup.procs').read_text().split(),
                    'actual_worker_in_scope',
                )
                require(
                    all(
                        str(pid) not in (cgroup / 'cgroup.procs').read_text().split()
                        for pid in [gateway.pid, *[pid for pid, _ in gateway_children]]
                    ),
                    'gateway_outside_scope',
                )
                read_events = lambda: {
                    k: int(v)
                    for k, v in (
                        line.split()
                        for line in (cgroup / 'memory.events').read_text().splitlines()
                    )
                }
                row['oom_before'] = {'probe': pressure[0], 'events': read_events()}
                commands.run(
                    'trigger-bounded-owned-worker-oom',
                    [
                        PYTHON,
                        '-c',
                        "import sys;from pathlib import Path;Path(sys.argv[1]).write_text('bounded executor-only pressure')",
                        str(root / ('oom-trigger-' + str(worker.pid))),
                    ],
                )
            else:
                commands.run(
                    'kill-owned-executor-child', ['kill', '-KILL', str(worker.pid)]
                )
            await wait_until(
                lambda: not worker.is_running() or worker.status() == 'zombie'
            )
            row['worker_death_observed_at'] = time.time()
            stat_path = Path('/proc') / str(worker.pid) / 'stat'
            if stat_path.exists():
                row['worker_wait_status'] = int(
                    stat_path.read_text().rsplit(')', 1)[1].split()[49]
                )
                require(
                    os.WIFSIGNALED(row['worker_wait_status'])
                    and os.WTERMSIG(row['worker_wait_status']) == signal.SIGKILL,
                    'actual_worker_sigkill_wait_status',
                )
            if oom:
                row['oom_after'] = {
                    'events': read_events(),
                    'pressure': events('oom_pressure'),
                }
                require(
                    row['oom_after']['events']['oom_kill']
                    > row['oom_before']['events']['oom_kill'],
                    'kernel_oom_kill_observed',
                )
                require(
                    row['oom_after']['events']['oom_group_kill']
                    == row['oom_before']['events']['oom_group_kill'],
                    'no_group_oom_kill',
                )
            await wait_until(
                lambda: (
                    not store.client.exists(store.keys.alive(producer['producer_id']))
                    and not list(
                        store.client.scan_iter(
                            match=store.keys.prefix
                            + 'producer:'
                            + producer['producer_id']
                            + ':*'
                        )
                    )
                    and all(store.metadata(attempt) is None for attempt in attempts)
                )
            )
            row['cleanup'] = {
                'observed_at': time.time(),
                'producer_keys': 0,
                'request_records': 0,
                'old_attempts': attempts,
            }
            require(
                gateway.is_running()
                and gateway.create_time() == gateway_parent['created'],
                'gateway_parent_survived',
            )
            require(
                all(
                    psutil.Process(pid).create_time() == created
                    for pid, created in gateway_children
                ),
                'gateway_children_survived',
            )
            row['after_death_history'] = await inspect_runtime(root, runtime)
            stop_sources({'processes': [executor_parent]}, commands)
            child_env = configurations(root, runtime, seam.url)
            manifest = startup_manifest(root, source)
            replacement_started = time.time()
            process, audit = commands.run(
                'replacement-executor',
                [
                    PYTHON,
                    '-m',
                    'tools.stress.llm_dispatch_deployment_child',
                    'executor',
                ],
                private=child_env,
                background=True,
            )
            replacement = dict(
                role='executor',
                pid=process.pid,
                created=psutil.Process(process.pid).create_time(),
                audit=str(audit),
                source_manifest=manifest,
            )
            runtime['processes'].append(replacement)
            save_json(root / 'private-runtime.json', runtime)
            await wait_until(
                lambda: any(
                    item['event'] == 'deployment_ready' and item['pid'] == process.pid
                    for item in probes(root)
                )
            )
            seam.before_first_response = None
            await gate(True)
            new_started = time.time()
            new_report = {}
            await invoke_nine(root, runtime, new_report)
            new_producers = [
                item
                for item in probes(root)
                if item['event'] == 'producer' and item['at'] >= replacement_started
            ]
            require(
                len(new_producers) == 1
                and new_producers[0]['producer_id'] != producer['producer_id'],
                'fresh_producer_identity',
            )
            verify_worker(new_producers[0], replacement['pid'])
            require(
                all(
                    item['producer_id'] == new_producers[0]['producer_id']
                    and item['attempt_id'] not in attempts
                    for item in new_report['ordinary_admissions']
                ),
                'replacement_never_adopts_old_work',
            )
            require(
                all(store.metadata(attempt) is None for attempt in attempts),
                'dead_work_not_recreated',
            )
            require(
                gateway.is_running()
                and gateway.create_time() == gateway_parent['created'],
                'gateway_survived_replacement',
            )
            row.update(
                status='passed',
                replacement=replacement,
                new_producer=new_producers[0],
                replacement_annotation=new_report,
                after_replacement_history=await inspect_runtime(root, runtime),
                replacement_mechanism='explicit real Deployment startup; no automatic supervision',
            )
            report['status'] = (
                'oom_passed' if oom else 'sigkill_passed_not_oom_qualified'
            )
        finally:
            if call:
                if not call.done():
                    call.cancel()
                result = await asyncio.gather(call, return_exceptions=True)
                row['original_controller_result'] = type(result[0]).__name__
            await seam.close()
            if store:
                store.close()


async def inspect_runtime(root: Path, runtime: dict) -> dict:
    import httpx

    result = {}
    async with httpx.AsyncClient(
        timeout=5,
        headers={
            'Authorization': 'Bearer ' + runtime['private']['QUAL_GATEWAY_API_KEY']
        },
    ) as http:
        for endpoint in ('health', 'jobs', 'attempts', 'dags'):
            response = await http.get(
                f'http://127.0.0.1:{runtime["ports"]["http"]}/api/operations/'
                + endpoint
            )
            result[endpoint] = {
                'http_status': response.status_code,
                'body': response.json(),
            }
    path = root / f'operations-{time.time_ns()}.json'
    save_json(path, result)
    return result


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture-manifest', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start-only', action='store_true')
    parser.add_argument('--stop-sources', action='store_true')
    parser.add_argument('--stop-owned', action='store_true')
    parser.add_argument('--invoke-only', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--s03', action='store_true')
    parser.add_argument('--death', action='store_true')
    parser.add_argument('--oom', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    if os.environ.get('QUAL_DEPLOYMENT_BOOTSTRAPPED') != '1':
        env = safe_environment(args.output, source) | {
            'QUAL_DEPLOYMENT_BOOTSTRAPPED': '1'
        }
        os.execve(
            PYTHON,
            [PYTHON, '-m', 'tools.stress.llm_dispatch_deployment', *sys.argv[1:]],
            env,
        )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    history = args.output / 'reports'
    history.mkdir(exist_ok=True)
    existing_report = args.output / 'report.json'
    if existing_report.exists():
        (history / f'{time.time_ns()}.json').write_bytes(existing_report.read_bytes())
    report: dict[str, Any] = {'status': 'failed', 'failures': [], 'scenarios': []}
    command_path = args.output / f'controller-{time.time_ns()}-command.json'
    controller_record = {
        'command': [
            sys.executable,
            '-m',
            'tools.stress.llm_dispatch_deployment',
            *sys.argv[1:],
        ],
        'cwd': str(Path.cwd()),
        'environment': safe_environment(args.output, source),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(),
    }
    save_json(command_path, controller_record)
    exit_code = 1
    phase = 'fixture_manifest'
    try:
        fixture = json.loads(args.fixture_manifest.read_text())
        source = Path(__file__).resolve().parents[2]
        commands = Commands(args.output, source, safe_environment(args.output, source))
        phase = 'infrastructure_prepare'
        runtime = (
            json.loads((args.output / 'private-runtime.json').read_text())
            if args.resume
            else prepare(args.output, source, fixture, commands)
        )
        report.update(
            project=runtime['project'],
            ports=runtime['ports'],
            containers=runtime['containers'],
        )
        if args.oom:
            phase = 'worker_oom'
            asyncio.run(
                qualify_death(args.output, source, runtime, commands, report, oom=True)
            )
            exit_code = 0
            return 0
        if args.death:
            phase = 'worker_death'
            asyncio.run(qualify_death(args.output, source, runtime, commands, report))
            exit_code = 0
            return 0
        if args.s03:
            phase = 's03'
            asyncio.run(qualify_s03(args.output, source, runtime, commands, report))
            exit_code = 0
            return 0
        if args.inspect:
            phase = 'inspect_operations'
            report['operations'] = asyncio.run(inspect_runtime(args.output, runtime))
            report['status'] = 'inspected_not_qualified'
            exit_code = 0
            return 0
        if args.stop_owned:
            phase = 'stop_owned'
            stop_owned(args.output, runtime, commands, report)
            report['status'] = 'owned_resources_stopped'
            exit_code = 0
            return 0
        if args.stop_sources:
            phase = 'stop_sources'
            stop_sources(runtime, commands)
            report['status'] = 'sources_stopped_not_qualified'
            exit_code = 0
            return 0
        if args.invoke_only:
            phase = 'ordinary_annotation'
            asyncio.run(invoke_nine(args.output, runtime, report))
            exit_code = 0
            return 0
        if args.restart:
            stop_sources(runtime, commands)
        if args.prepare:
            report['status'] = 'prepared_not_qualified'
            exit_code = 0
            return 0
        phase = 'source_startup'
        asyncio.run(start_sources(args.output, source, runtime, commands, report))
        if args.start_only:
            exit_code = 0
            return 0
        phase = 'deployment_not_implemented'
        raise RuntimeError('Deployment coverage has not executed')
    except Exception as error:
        import traceback

        report['failures'].append(
            {
                'phase': phase,
                'type': type(error).__name__,
                'locations': [
                    {'file': f.filename, 'line': f.lineno, 'function': f.name}
                    for f in traceback.extract_tb(error.__traceback__)
                ],
            }
        )
    finally:
        save_json(args.output / 'report.json', report)
        for case in report['scenarios']:
            if case.get('status') == 'passed' and not report['failures']:
                save_json(
                    args.output / ('case-report-' + case['name'] + '.json'), report
                )
        log = command_path.with_name(command_path.name.replace('-command.json', '.log'))
        log.write_text(
            json.dumps({'status': report['status'], 'failures': report['failures']})
            + '\n'
        )
        controller_record['log'] = str(log)
        controller_record.update(
            exit_code=exit_code,
            finished_at=datetime.now(timezone.utc).isoformat(),
            report=str(args.output / 'report.json'),
        )
        save_json(command_path, controller_record)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
