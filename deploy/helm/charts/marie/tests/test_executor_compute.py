from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[1]
IMAGE = 'example.invalid/executor@sha256:' + 'a' * 64


def pool(name: str, gpu: bool = False, stateful: bool = False) -> dict:
    resources = {
        'requests': {'cpu': '500m', 'memory': '2Gi'},
        'limits': {'cpu': '2', 'memory': '4Gi'},
    }
    if gpu:
        resources['requests']['nvidia.com/gpu'] = 2
        resources['limits']['nvidia.com/gpu'] = 2
    return {
        'name': name,
        'enabled': True,
        'replicas': 2,
        'gpu': gpu,
        'stateful': stateful,
        'image': {'repository': IMAGE.split('@')[0], 'digest': IMAGE.split('@')[1]},
        'resources': resources,
        'nodeSelector': {'test.example/pool': 'gpu' if gpu else 'cpu'},
        'runtimeClassName': 'selected-runtime',
        'env': [{'name': 'PRESERVE', 'value': 'configured'}],
        'tolerations': [{'key': 'test.example/placement', 'operator': 'Exists'}],
    }


class ExecutorComputeTest(unittest.TestCase):
    def render(
        self,
        pools: list[dict],
        release: str = 'compute-test',
        executor_values: dict | None = None,
        namespace: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix='executor-compute-') as directory:
            path = Path(directory) / 'values.yaml'
            values = {'enabled': True, 'pools': pools, **(executor_values or {})}
            path.write_text(yaml.safe_dump({'executor': values}))
            return subprocess.run(
                [
                    'helm',
                    'template',
                    release,
                    str(CHART),
                    '--namespace',
                    namespace or release,
                    '--values',
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env={
                    **os.environ,
                    'HELM_CACHE_HOME': str(Path(directory) / 'cache'),
                    'HELM_CONFIG_HOME': str(Path(directory) / 'config'),
                    'HELM_DATA_HOME': str(Path(directory) / 'data'),
                },
            )

    def executors(self, result: subprocess.CompletedProcess[str]) -> list[dict]:
        self.assertEqual(result.returncode, 0, result.stderr)
        return [
            document
            for document in yaml.safe_load_all(result.stdout)
            if document
            and document.get('kind') in ('Deployment', 'StatefulSet')
            and document['metadata']
            .get('labels', {})
            .get('app.kubernetes.io/component')
            == 'executor'
        ]

    def test_mixed_cpu_gpu_and_stateful_workloads_preserve_selected_configuration(
        self,
    ) -> None:
        pools = [
            pool('cpu'),
            pool('gpu', gpu=True),
            pool('gpu-stateful', gpu=True, stateful=True),
        ]
        documents = self.executors(self.render(pools))
        self.assertEqual(len(documents), 3)
        for selected in pools:
            with self.subTest(pool=selected['name']):
                document = next(
                    item
                    for item in documents
                    if item['metadata']['labels'].get('marie.ai/pool')
                    == selected['name']
                )
                self.assertEqual(
                    document['kind'],
                    'StatefulSet' if selected['stateful'] else 'Deployment',
                )
                self.assertEqual(document['spec']['replicas'], 2)
                pod = document['spec']['template']['spec']
                self.assertEqual(pod['nodeSelector'], selected['nodeSelector'])
                self.assertEqual(pod['tolerations'], selected['tolerations'])
                self.assertEqual(pod['runtimeClassName'], selected['runtimeClassName'])
                container = pod['containers'][0]
                self.assertEqual(container['image'], IMAGE)
                self.assertEqual(container['resources'], selected['resources'])
                self.assertIn(
                    {'name': 'PRESERVE', 'value': 'configured'}, container['env']
                )
                generated_visibility = [
                    item
                    for item in container['env']
                    if item['name']
                    in ('NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')
                ]
                self.assertEqual(
                    generated_visibility,
                    [],
                    'chart must leave allocation visibility to the device plugin/runtime',
                )

    def test_cpu_only_pool_requests_no_gpu(self) -> None:
        documents = self.executors(self.render([pool('cpu')]))
        self.assertEqual(len(documents), 1)
        resources = documents[0]['spec']['template']['spec']['containers'][0][
            'resources'
        ]
        self.assertNotIn('nvidia.com/gpu', resources['requests'])
        self.assertNotIn('nvidia.com/gpu', resources['limits'])

    def test_rolling_update_preserves_counts_zero_and_percentages(self) -> None:
        for strategy, expected in (
            ({}, {'maxSurge': 1, 'maxUnavailable': 0}),
            (
                {'maxSurge': 0, 'maxUnavailable': 1},
                {'maxSurge': 0, 'maxUnavailable': 1},
            ),
            (
                {'maxSurge': '25%', 'maxUnavailable': '10%'},
                {'maxSurge': '25%', 'maxUnavailable': '10%'},
            ),
        ):
            with self.subTest(strategy=strategy):
                selected = pool('cpu')
                selected['strategy'] = strategy
                document = self.executors(self.render([selected]))[0]
                self.assertEqual(
                    document['spec']['strategy']['rollingUpdate'], expected
                )

    def test_limit_only_gpu_reservation_is_valid(self) -> None:
        selected = pool('gpu', gpu=True)
        del selected['resources']['requests']['nvidia.com/gpu']
        self.assertEqual(len(self.executors(self.render([selected]))), 1)

    def test_invalid_gpu_reservations_fail_for_both_workload_kinds(self) -> None:
        cases = {
            'missing-limit': {'gpu': True, 'resources': {}},
            'request-only': {
                'gpu': True,
                'resources': {'requests': {'nvidia.com/gpu': 1}},
            },
            'mismatch': {
                'gpu': True,
                'resources': {
                    'requests': {'nvidia.com/gpu': 1},
                    'limits': {'nvidia.com/gpu': 2},
                },
            },
            'zero': {'gpu': True, 'resources': {'limits': {'nvidia.com/gpu': 0}}},
            'negative': {'gpu': True, 'resources': {'limits': {'nvidia.com/gpu': -1}}},
            'fractional': {
                'gpu': True,
                'resources': {'limits': {'nvidia.com/gpu': '0.5'}},
            },
            'excessive': {
                'gpu': True,
                'resources': {'limits': {'nvidia.com/gpu': 1000000001}},
            },
            'cpu-limit': {'gpu': False, 'resources': {'limits': {'nvidia.com/gpu': 1}}},
            'cpu-request': {
                'gpu': False,
                'resources': {'requests': {'nvidia.com/gpu': 1}},
            },
            'string-mode': {'gpu': 'false'},
            'string-stateful': {'stateful': 'false'},
            'bad-resources': {'resources': 'invalid'},
            'bad-requests': {'resources': {'requests': []}},
        }
        for stateful in (False, True):
            for name, overrides in cases.items():
                with self.subTest(name=name, stateful=stateful):
                    selected = {**pool('invalid', stateful=stateful), **overrides}
                    result = self.render([selected])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('executor compute:', result.stderr)

    def test_duplicate_pool_names_fail(self) -> None:
        result = self.render([pool('same'), pool('same', gpu=True)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unique workload names', result.stderr)

    def test_long_pool_names_do_not_collide_after_truncation(self) -> None:
        selected = [pool('same-executor-cpu'), pool('same-executor-gpu', gpu=True)]
        documents = self.executors(self.render(selected, 's' * 47))
        self.assertEqual(len({item['metadata']['name'] for item in documents}), 2)

    def test_disabled_pool_does_not_need_complete_compute_configuration(self) -> None:
        documents = self.executors(
            self.render(
                [pool('cpu'), {'name': 'disabled', 'enabled': False, 'gpu': True}]
            )
        )
        self.assertEqual(len(documents), 1)

    def test_headless_service_names_do_not_collide_with_another_pool(self) -> None:
        selected = [pool('a', stateful=True), pool('a-headless', gpu=True)]
        for pools in (selected, list(reversed(selected))):
            result = self.render(pools)
            self.assertEqual(result.returncode, 0, result.stderr)
            services = [
                item
                for item in yaml.safe_load_all(result.stdout)
                if item
                and item['kind'] == 'Service'
                and item['metadata'].get('labels', {}).get('marie.ai/pool')
            ]
            self.assertEqual(len(services), 3)
            self.assertEqual(len({item['metadata']['name'] for item in services}), 3)
            self.assertTrue(
                all(len(item['metadata']['name']) <= 63 for item in services)
            )

    def test_numeric_pool_ids_remain_strings_in_labels_and_selectors(self) -> None:
        result = self.render([pool('123', stateful=True)])
        self.assertEqual(result.returncode, 0, result.stderr)
        checked = 0
        for document in yaml.safe_load_all(result.stdout):
            if (
                not document
                or document.get('metadata', {}).get('labels', {}).get('marie.ai/pool')
                != '123'
            ):
                continue
            self.assertEqual(document['metadata']['labels']['marie.ai/pool'], '123')
            if document['kind'] == 'Service':
                self.assertEqual(document['spec']['selector']['marie.ai/pool'], '123')
            elif document['kind'] == 'StatefulSet':
                self.assertEqual(
                    document['spec']['template']['metadata']['labels']['marie.ai/pool'],
                    '123',
                )
                self.assertEqual(
                    document['spec']['selector']['matchLabels']['marie.ai/pool'], '123'
                )
            checked += 1
        self.assertEqual(checked, 3)

    def test_explicit_operator_environment_is_preserved(self) -> None:
        selected = pool('gpu', gpu=True)
        selected['env'] = [{'name': 'CUDA_VISIBLE_DEVICES', 'value': '0,1'}]
        document = self.executors(self.render([selected]))[0]
        env = document['spec']['template']['spec']['containers'][0]['env']
        self.assertEqual(
            [item for item in env if item['name'] == 'CUDA_VISIBLE_DEVICES'],
            selected['env'],
        )


if __name__ == '__main__':
    unittest.main()
