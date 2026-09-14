"""Content-free process probes and owned output paths for the fixed deployment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

request_identity: ContextVar[dict] = ContextVar(
    'qualification_request_identity', default={}
)


def observe_oom_trigger(root: Path) -> None:
    scope = os.environ.get('QUAL_OOM_SCOPE')
    if not scope:
        return
    membership = Path('/proc/self/cgroup').read_text().strip().split('::')[1]
    cgroup = Path('/sys/fs/cgroup') / membership.lstrip('/')
    limit = 8 * 1024**3
    if (
        cgroup.name != scope
        or (cgroup / 'memory.max').read_text().strip() != str(limit)
        or (cgroup / 'memory.swap.max').read_text().strip() != '0'
        or (cgroup / 'memory.oom.group').read_text().strip() != '0'
    ):
        emit(root, 'oom_refused', category='scope_limit_mismatch')
        return
    emit(root, 'oom_ready', cgroup=str(cgroup), memory_max=limit, swap_max=0)
    trigger = root / ('oom-trigger-' + str(os.getpid()))
    while not trigger.exists():
        time.sleep(0.1)
    Path('/proc/self/oom_score_adj').write_text('1000')
    emit(
        root,
        'oom_pressure',
        cgroup=str(cgroup),
        limit=limit,
        oom_score_adj=Path('/proc/self/oom_score_adj').read_text().strip(),
    )
    chunks = []
    for _ in range(limit // (64 * 1024**2) + 2):
        chunk = bytearray(64 * 1024**2)
        chunk[::4096] = b'x' * (len(chunk) // 4096)
        chunks.append(chunk)
    emit(root, 'oom_not_killed', allocated=sum(map(len, chunks)))


def emit(root: Path, event: str, **values: Any) -> None:
    import psutil

    process = psutil.Process()
    row = dict(
        event=event,
        pid=process.pid,
        ppid=process.ppid(),
        created=process.create_time(),
        at=time.time(),
        **(request_identity.get() | values),
    )
    fd = os.open(root / 'probe.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(row) + '\n').encode())
    finally:
        os.close(fd)


def install(root: Path) -> None:
    import marie.constants as constants

    constants.__cache_path__ = str(root / 'native')
    Path(constants.__cache_path__).mkdir(exist_ok=True)
    import marie.utils.server_runtime as server_runtime

    server_runtime.__cache_path__ = constants.__cache_path__
    from marie.serve.runtimes.gateway.marie import MarieGateway

    # isort: split
    # Import the package first to avoid its direct module's circular import.
    import marie.serve.runtimes.servers.marie_gateway as gateway_module

    original_load_env = gateway_module.load_env_file

    def load_env(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get('dotenv_path') != '/dev/null':
            raise ValueError(
                'Qualification gateway requires explicit /dev/null env_file'
            )
        return original_load_env(*args, **kwargs)

    gateway_module.load_env_file = load_env
    import marie.serve.discovery as discovery_module
    import marie.serve.runtimes.worker.request_handling as worker_module

    # These modules have no advertised-address override. The actual listener is local.
    discovery_module.get_internal_ip = lambda: '127.0.0.1'
    worker_module.get_internal_ip = lambda: '127.0.0.1'
    import marie.scheduler.planner_util as planner_util

    original_plan_to_yaml = planner_util.plan_to_yaml

    def plan_to_yaml(plan: Any, output_path: str = 'query_plan_pretty.yaml') -> str:
        if output_path == 'query_plan_pretty.yaml':
            output_path = str(root / f'query-plan-{time.time_ns()}.yaml')
        return original_plan_to_yaml(plan, output_path)

    planner_util.plan_to_yaml = plan_to_yaml
    import marie.utils.asset_util as asset_util

    class AssetPath(type(Path())):
        @classmethod
        def home(cls) -> Path:
            return Path(root / 'asset-locks')

    asset_util.Path = AssetPath
    import marie.executor.extract.document_annotator_executor as executor_module
    import marie.extract.annotators.llm_annotator as annotator_module

    executor_module.__config_dir__ = str(root / 'generated-config')
    annotator_module.__config_dir__ = str(root / 'generated-config')

    from marie.engine.llm_queue.producer import V3Producer
    from marie.engine.llm_queue.store import RequestStore

    from marie.extract.annotators import util
    from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
        GatewayLlmDispatchRuntime,
    )

    original_start = V3Producer._start
    original_admit = RequestStore.admit
    original_write = util._write_single_result
    original_request = (
        executor_module.DocumentAnnotatorExecutor._process_annotation_request
    )
    original_gateway = GatewayLlmDispatchRuntime._start_v3

    def start(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_start(self, *args, **kwargs)
        if not getattr(self, '_qualification_observer', False):
            self._qualification_observer = True
            emit(root, 'producer', producer_id=self.producer_id, fabric=self.fabric_id)

            def observe() -> None:
                previous = None
                while not self._stop.wait(0.1):
                    with self._condition:
                        pending = sorted(self._pending)
                    if pending != previous:
                        emit(
                            root,
                            'pending',
                            producer_id=self.producer_id,
                            attempts=pending,
                        )
                        previous = pending

            threading.Thread(target=observe, daemon=True).start()
            if os.environ.get('QUAL_OOM_SCOPE'):
                threading.Thread(
                    target=observe_oom_trigger, args=(root,), daemon=True
                ).start()
        return result

    def admit(self: Any, request: Any) -> Any:
        result = original_admit(self, request)
        if result.disposition == 'admitted':
            emit(
                root,
                'admitted',
                producer_id=request.producer_id,
                attempt_id=request.attempt_id,
                task_id=request.logical_task_id,
                expires_at_ms=request.expires_at_ms,
            )
        return result

    def write(**kwargs: Any) -> Any:
        result = original_write(**kwargs)
        suffix = '.json' if kwargs['expect_output'] == 'json' else '.md'
        path = Path(kwargs['output_path']) / (
            Path(kwargs['b_image_path']).stem + kwargs.get('output_suffix', '') + suffix
        )
        emit(
            root,
            'write',
            task_id=kwargs['task_id'],
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return result

    async def request(
        self: Any, docs: Any, parameters: dict, *args: Any, **kwargs: Any
    ) -> Any:
        identity = {
            key: parameters.get(key) for key in ('job_id', 'dag_id', 'node_task_id')
        }
        token = request_identity.set(identity)
        emit(root, 'executor_start', **identity)
        try:
            result = await original_request(self, docs, parameters, *args, **kwargs)
        except BaseException as error:
            emit(root, 'executor_error', category=type(error).__name__, **identity)
            request_identity.reset(token)
            raise
        emit(root, 'executor_end', status=result.get('status'), **identity)
        request_identity.reset(token)
        return result

    async def gateway(self: Any) -> None:
        await original_gateway(self)
        emit(root, 'gateway_dispatch', fabric=self.config.fabric_group_id)

    V3Producer._start = start
    RequestStore.admit = admit
    util._write_single_result = write
    executor_module.DocumentAnnotatorExecutor._process_annotation_request = request
    GatewayLlmDispatchRuntime._start_v3 = gateway
    emit(
        root,
        'imports',
        modules={
            module.__name__: module.__file__
            for module in (executor_module, annotator_module, asset_util)
        },
        python=sys.executable,
    )


def main() -> None:
    root = Path(os.environ['QUAL_DEPLOYMENT_ROOT'])
    install(root)
    role = sys.argv[1]
    emit(root, 'bootstrap', role=role)
    if role == 'gateway':
        from marie.cli import main as cli

        from marie.serve.runtimes.gateway.marie import MarieGateway

        sys.argv = [
            'marie',
            'server',
            '--start',
            '--uses',
            str(root / 'gateway.yml'),
            '--env-file',
            '/dev/null',
        ]
        cli()
    elif role == 'executor':
        from marie.conf.helper import load_yaml
        from marie.orchestrate.deployments import Deployment

        config = load_yaml(str(root / 'executor.yml'), substitute=True)
        deployment = Deployment.load_config(config, include_gateway=False)
        with deployment:
            emit(
                root,
                'deployment_ready',
                role=role,
                children=[pod.worker.pid for pod in deployment._iter_pods()],
            )
            deployment.block()
    else:
        raise ValueError('Unknown fixed bootstrap role')


if __name__ == '__main__':
    main()
