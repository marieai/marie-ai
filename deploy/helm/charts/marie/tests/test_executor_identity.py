from __future__ import annotations

import hashlib
import json
import unittest

import test_executor_compute as compute
import yaml

HOSTED_NAMESPACE = 'sbx-550e8400-e29b-41d4-a716-446655440000-a1b2c3'


def identity(namespace: str, release: str, name: str) -> str:
    encoded = json.dumps(
        ['marie.executor-workload', namespace, release, name], separators=(',', ':')
    )
    return 'ex-' + hashlib.sha256(encoded.encode()).hexdigest()[:24]


class ExecutorIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chart = compute.ExecutorComputeTest()

    def documents(
        self,
        pools: list[dict],
        release: str = 'compute-test',
        namespace: str | None = None,
    ) -> list[dict]:
        result = self.chart.render(pools, release=release, namespace=namespace)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [item for item in yaml.safe_load_all(result.stdout) if item]

    def test_hosted_mixed_workloads_services_and_self_references(self) -> None:
        pools = [
            compute.pool('cpu'),
            compute.pool('gpu', gpu=True),
            compute.pool('cpu-state', stateful=True),
            compute.pool('gpu-state', gpu=True, stateful=True),
        ]
        documents = self.documents(pools, HOSTED_NAMESPACE)
        workers = [
            item
            for item in documents
            if item['kind'] in ('Deployment', 'StatefulSet')
            and item['metadata'].get('labels', {}).get('app.kubernetes.io/component')
            == 'executor'
        ]
        self.assertEqual(len(workers), 4)
        services = {
            item['metadata']['name']: item
            for item in documents
            if item['kind'] == 'Service'
        }
        for worker in workers:
            pool_name = worker['metadata']['labels']['marie.ai/pool']
            selected = next(pool for pool in pools if pool['name'] == pool_name)
            name = identity(HOSTED_NAMESPACE, HOSTED_NAMESPACE, pool_name)
            self.assertEqual(worker['metadata']['name'], name)
            self.assertEqual(worker['spec']['replicas'], 2)
            selector = worker['spec']['selector']['matchLabels']
            self.assertEqual(selector['app.kubernetes.io/instance'], HOSTED_NAMESPACE)
            self.assertEqual(selector['marie.ai/pool'], pool_name)
            self.assertEqual(services[name]['spec']['selector'], selector)
            pod = worker['spec']['template']
            for key, value in selector.items():
                self.assertEqual(pod['metadata']['labels'][key], value)
            container = pod['spec']['containers'][0]
            self.assertEqual(container['image'], compute.IMAGE)
            self.assertEqual(container['resources'], selected['resources'])
            self.assertEqual(pod['spec']['nodeSelector'], selected['nodeSelector'])
            self.assertEqual(pod['spec']['tolerations'], selected['tolerations'])
            self.assertEqual(
                pod['spec']['runtimeClassName'], selected['runtimeClassName']
            )
            self.assertIn({'name': 'PRESERVE', 'value': 'configured'}, container['env'])
            key = (
                'K8S_STATEFULSET_NAME'
                if selected['stateful']
                else 'K8S_DEPLOYMENT_NAME'
            )
            self.assertIn({'name': key, 'value': name}, container['env'])
            if selected['stateful']:
                service = name + '-headless'
                self.assertEqual(worker['spec']['serviceName'], service)
                self.assertEqual(services[service]['spec']['clusterIP'], 'None')
                self.assertEqual(services[service]['spec']['selector'], selector)
                self.assertLessEqual(len(name + '-4294967295'), 63)
                self.assertLessEqual(len(name + '-9999'), 63)
                self.assertLessEqual(len(service), 63)
                self.assertEqual(
                    worker['spec']['volumeClaimTemplates'][0]['metadata']['name'],
                    'data',
                )
                self.assertLessEqual(len('data-' + name + '-9999'), 253)

    def test_identity_uses_namespace_release_and_complete_pool_name(self) -> None:
        names = set()
        for namespace, release, pool_name in [
            ('one', 'one', 'a'),
            ('two', 'one', 'a'),
            ('one', 'two', 'a'),
            ('one', 'one', 'a-headless'),
            ('one', 'one', '123'),
            ('one', 'one', 'x' * 62 + 'a'),
            ('one', 'one', 'x' * 62 + 'b'),
        ]:
            documents = self.documents([compute.pool(pool_name)], release, namespace)
            worker = next(
                item
                for item in documents
                if item['kind'] == 'Deployment'
                and item['metadata'].get('labels', {}).get('marie.ai/pool') == pool_name
            )
            name = worker['metadata']['name']
            self.assertEqual(name, identity(namespace, release, pool_name))
            self.assertNotIn(name, names)
            names.add(name)
            self.assertEqual(
                documents, self.documents([compute.pool(pool_name)], release, namespace)
            )

    def test_stateful_replicas_require_a_canonical_bounded_count(self) -> None:
        for replicas in [
            -1,
            True,
            1.5,
            '00010',
            '01',
            '00',
            10001,
            'private-invalid-value',
        ]:
            with self.subTest(replicas=replicas):
                selected = {**compute.pool('cpu', stateful=True), 'replicas': replicas}
                result = self.chart.render([selected])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('executor identity:', result.stderr)
                self.assertNotIn('private-invalid-value', result.stderr)

    def test_zero_default_and_maximum_stateful_replicas(self) -> None:
        for replicas in [None, 0, 10000, '10000']:
            with self.subTest(replicas=replicas):
                selected = compute.pool('cpu', stateful=True)
                if replicas is None:
                    del selected['replicas']
                else:
                    selected['replicas'] = replicas
                documents = self.documents([selected], HOSTED_NAMESPACE)
                worker = next(
                    item
                    for item in documents
                    if item['kind'] == 'StatefulSet'
                    and item['metadata'].get('labels', {}).get('marie.ai/pool') == 'cpu'
                )
                self.assertEqual(
                    worker['spec']['replicas'],
                    None if replicas is None else int(replicas),
                )

    def test_render_order_does_not_change_pool_identity(self) -> None:
        pools = [compute.pool('a', stateful=True), compute.pool('a-headless')]

        def services(selected: list[dict]) -> dict:
            return {
                item['metadata']['name']: item
                for item in self.documents(selected)
                if item['kind'] == 'Service'
            }

        self.assertEqual(services(pools), services(list(reversed(pools))))


if __name__ == '__main__':
    unittest.main()
