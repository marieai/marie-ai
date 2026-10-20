from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_executor_compute as compute
import yaml


class ExecutorSmokeTest(unittest.TestCase):
    def test_smoke_rollouts_match_actual_chart_resources(self) -> None:
        # Render real resources; every cluster/container command in the script is a stub.
        rendered = subprocess.run(
            [
                'helm',
                'template',
                'marie',
                str(compute.CHART),
                '--namespace',
                'marie',
                '--values',
                str(compute.CHART / 'values-local.yaml'),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        resources = [item for item in yaml.safe_load_all(rendered.stdout) if item]
        with tempfile.TemporaryDirectory(prefix='executor-smoke-test-') as scratch:
            directory = Path(scratch)
            (directory / 'resources.json').write_text(json.dumps(resources))
            stub = directory / 'stub'
            stub.write_text(
                '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ['SMOKE_TEST_DIR'])
if command == 'docker' and args[:2] == ['image', 'inspect']:
    sys.exit(1)
if command == 'k3d' and args == ['cluster', 'list']:
    print('marie-helm-smoke 1 2 running')
    sys.exit(0)
if command == 'helm' and (args[:2] == ['dependency', 'build'] or args[:2] == ['upgrade', '--install']):
    sys.exit(0)
if command == 'kubectl':
    if args == ['cluster-info'] or args == ['-n', 'marie', 'get', 'pods', '-o', 'wide']:
        sys.exit(0)
    if args[:4] == ['-n', 'marie', 'rollout', 'status']:
        target = args[4]
        kind, _, name = target.partition('/')
        kind = {'deploy': 'Deployment', 'deployment': 'Deployment', 'statefulset': 'StatefulSet'}.get(kind)
        selected = [r for r in json.loads((root / 'resources.json').read_text()) if r['kind'] == kind]
        if name:
            selected = [r for r in selected if r['metadata']['name'] == name]
        elif '--selector' in args:
            selector = dict(pair.split('=', 1) for pair in args[args.index('--selector') + 1].split(','))
            selected = [r for r in selected if all(r['metadata']['labels'].get(k) == v for k, v in selector.items())]
        else:
            selected = []
        if len(selected) != 1:
            sys.exit('rollout target did not match exactly one rendered workload: ' + target)
        with (root / 'rollouts.jsonl').open('a') as log:
            log.write(json.dumps(selected[0]['metadata']) + '\\n')
        sys.exit(0)
sys.exit('unexpected stub command: ' + command + ' ' + repr(args))
'''
            )
            stub.chmod(0o755)
            for command in ['docker', 'k3d', 'helm', 'kubectl']:
                (directory / command).symlink_to(stub)
            env = {
                **os.environ,
                'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                'SMOKE_TEST_DIR': scratch,
                'CLUSTER_NAME': 'marie-helm-smoke',
                'NAMESPACE': 'marie',
                'RELEASE': 'marie',
                'LOAD_LOCAL_IMAGES': 'false',
                'INSTALL_ARGOCD': 'false',
            }
            result = subprocess.run(
                ['bash', str(compute.CHART.parents[2] / 'smoke-marie-helm.sh'), 'k3d'],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rollouts = [
                json.loads(line)
                for line in (directory / 'rollouts.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(rollouts), 3)
            self.assertEqual(
                sum(
                    item['labels'].get('marie.ai/pool') == 'cpu-local'
                    for item in rollouts
                ),
                1,
            )


if __name__ == '__main__':
    unittest.main()
