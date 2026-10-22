from __future__ import annotations

import copy
import unittest

import test_executor_compute as compute

pool = compute.pool

CONFIG = '/etc/marie/plugin-executor.yml'
RUNTIME = {
    'serviceName': 'scheduled-plugins',
    'signingSecret': 'scheduled-plugin-signing',
}
RESERVED = (
    'MARIE_PLUGIN_DAEMON_URL',
    'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID',
    'MARIE_PLUGIN_DAEMON_SIGNING_SECRET',
)


class ExecutorPluginRuntimeTest(unittest.TestCase):
    def test_only_selected_pool_receives_namespaced_signed_runtime_binding(
        self,
    ) -> None:
        chart = compute.ExecutorComputeTest()
        for stateful in (False, True):
            with self.subTest(stateful=stateful):
                selected = pool('plugins', stateful=stateful)
                selected.update(configFile=CONFIG, pluginRuntime=RUNTIME)
                regular = pool('ocr', gpu=True)
                documents = chart.executors(
                    chart.render([selected, regular], namespace='child-proof')
                )
                worker = next(
                    doc
                    for doc in documents
                    if doc['metadata']['labels']['marie.ai/pool'] == 'plugins'
                )
                container = worker['spec']['template']['spec']['containers'][0]
                entries = [
                    entry for entry in container['env'] if entry['name'] in RESERVED
                ]
                self.assertEqual(
                    entries,
                    [
                        {
                            'name': RESERVED[0],
                            'value': 'http://scheduled-plugins.child-proof.svc:8099',
                        },
                        {
                            'name': RESERVED[1],
                            'valueFrom': {
                                'secretKeyRef': {
                                    'name': 'scheduled-plugin-signing',
                                    'key': 'key-id',
                                }
                            },
                        },
                        {
                            'name': RESERVED[2],
                            'valueFrom': {
                                'secretKeyRef': {
                                    'name': 'scheduled-plugin-signing',
                                    'key': 'secret',
                                }
                            },
                        },
                    ],
                )
                self.assertEqual(
                    container['args'], ['server', '--start', '--uses', CONFIG]
                )
                other = next(
                    doc
                    for doc in documents
                    if doc['metadata']['labels']['marie.ai/pool'] == 'ocr'
                )
                self.assertFalse(
                    any(
                        entry['name'] in RESERVED
                        for entry in other['spec']['template']['spec']['containers'][0][
                            'env'
                        ]
                    )
                )
                self.assertEqual(
                    other['spec']['template']['spec']['containers'][0]['resources'],
                    regular['resources'],
                )

    def test_invalid_runtime_bindings_fail_before_deployment(self) -> None:
        chart = compute.ExecutorComputeTest()
        cases = [
            None,
            False,
            'runtime',
            {},
            {**RUNTIME, 'url': 'http://parent'},
            {**RUNTIME, 'serviceName': ''},
            {**RUNTIME, 'serviceName': 'other.parent.svc'},
            {**RUNTIME, 'serviceName': 'http://localhost:8099'},
            {**RUNTIME, 'serviceName': '1bad'},
            {**RUNTIME, 'signingSecret': ''},
            {**RUNTIME, 'signingSecret': 'bad/name'},
            {**RUNTIME, 'signingSecret': 3},
        ]
        for runtime in cases:
            with self.subTest(runtime=runtime):
                selected = pool('plugins')
                selected.update(configFile=CONFIG, pluginRuntime=runtime)
                result = chart.render([selected])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('plugin runtime', result.stderr)

    def test_binding_requires_explicit_flow_and_rejects_env_overrides(self) -> None:
        chart = compute.ExecutorComputeTest()
        cases = [
            {'configFile': path}
            for path in ('', 'relative.yml', 3, '/etc/marie/$(FLOW).yml')
        ]
        cases += [{'env': [{'name': name, 'value': 'override'}]} for name in RESERVED]
        for stateful in (False, True):
            for change in cases:
                with self.subTest(stateful=stateful, change=change):
                    selected = pool('plugins', stateful=stateful)
                    selected.update(configFile=CONFIG, pluginRuntime=RUNTIME)
                    selected.update(change)
                    result = chart.render([selected])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('plugin runtime', result.stderr)
        selected = pool('plugins')
        selected['pluginRuntime'] = RUNTIME
        self.assertNotEqual(chart.render([selected]).returncode, 0)

    def test_disabled_incomplete_pool_is_ignored(self) -> None:
        selected = copy.deepcopy(pool('plugins'))
        selected.update(enabled=False, pluginRuntime={})
        chart = compute.ExecutorComputeTest()
        self.assertEqual(len(chart.executors(chart.render([selected, pool('cpu')]))), 1)


if __name__ == '__main__':
    unittest.main()
