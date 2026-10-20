from __future__ import annotations

import subprocess
import unittest
from pathlib import PurePosixPath

import test_executor_compute as compute
import yaml

CONFIG = '/etc/marie/config/service/mock/marie-mock-sandbox.yml'
PROFILE = compute.CHART / 'values-sandbox-mock.yaml'


class ExecutorMockTest(unittest.TestCase):
    def test_mock_profile_resolves_mounted_flow_for_both_controllers(self) -> None:
        self.assertTrue(PROFILE.is_file(), 'mock setup profile must be packaged')
        profile = yaml.safe_load(PROFILE.read_text())
        chart = compute.ExecutorComputeTest()
        for stateful in (False, True):
            with self.subTest(stateful=stateful):
                selected = profile['executor']['pools'][0]
                selected['stateful'] = stateful
                result = chart.render([selected], namespace='mock-child')
                worker = chart.executors(result)[0]
                pod = worker['spec']['template']['spec']
                container = pod['containers'][0]
                self.assertEqual(container['args'][-1], CONFIG)
                mount = next(
                    m
                    for m in container['volumeMounts']
                    if CONFIG.startswith(m['mountPath'] + '/')
                )
                self.assertTrue(mount['readOnly'])
                volume = next(v for v in pod['volumes'] if v['name'] == mount['name'])
                configmap = next(
                    d
                    for d in yaml.safe_load_all(result.stdout)
                    if d
                    and d['kind'] == 'ConfigMap'
                    and d['metadata']['name'] == volume['configMap']['name']
                )
                filename = PurePosixPath(CONFIG).name
                flow = yaml.safe_load(configmap['data'][filename])
                executor = flow['executors'][0]
                self.assertEqual(len(flow['executors']), 1)
                self.assertEqual(executor['name'], 'mock_executor_a')
                self.assertEqual(executor['uses']['jtype'], 'IntegrationExecutorMock')
                self.assertEqual(executor['uses']['with']['device'], 'cpu')
                self.assertIn('/document/process', flow['with']['expose_endpoints'])
                self.assertTrue(flow['with']['discovery'])
                self.assertEqual(
                    flow['with']['discovery_host'], '${{ ENV.ETCD_HOSTNAME }}'
                )
                self.assertEqual(flow['with']['port'], '${{ ENV.MARIE_PORT }}')
                self.assertEqual(
                    flow['with']['kv_store_kwargs']['password'],
                    '${{ ENV.DATABASE_PASSWORD }}',
                )
                self.assertNotIn('scheduler', flow)
                self.assertFalse(executor['uses']['with']['storage'])
                for section in ('requests', 'limits'):
                    self.assertNotIn('nvidia.com/gpu', container['resources'][section])
                self.assertFalse(
                    any(
                        e['name'].startswith('MARIE_PLUGIN_DAEMON_')
                        for e in container['env']
                    )
                )

    def test_mock_profile_registers_scheduler_planner(self) -> None:
        for selected in (False, True):
            args = ['helm', 'template', 'mock-check', str(compute.CHART)]
            if selected:
                args += ['-f', str(PROFILE)]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            configs = [
                d
                for d in yaml.safe_load_all(result.stdout)
                if d and d['kind'] == 'ConfigMap'
            ]
            config = next(
                d['data']['marie-gateway-4.0.0.yml']
                for d in configs
                if 'marie-gateway-4.0.0.yml' in d.get('data', {})
            )
            planners = yaml.safe_load(config.removeprefix('!MarieGateway\n'))['with'][
                'job_scheduler_kwargs'
            ]['query_planners']['planners']
            self.assertEqual(
                planners,
                (
                    [
                        {
                            'name': 'mock_planners',
                            'py_module': 'marie.query_planner.mock_query_plans',
                        }
                    ]
                    if selected
                    else []
                ),
            )

    def test_regular_cpu_gpu_pools_keep_their_selected_flow(self) -> None:
        chart = compute.ExecutorComputeTest()
        pools = [compute.pool('cpu'), compute.pool('gpu', gpu=True)]
        for selected in pools:
            selected['configFile'] = '/etc/marie/custom-flow.yml'
        for worker in chart.executors(chart.render(pools)):
            self.assertEqual(
                worker['spec']['template']['spec']['containers'][0]['args'][-1],
                '/etc/marie/custom-flow.yml',
            )


if __name__ == '__main__':
    unittest.main()
