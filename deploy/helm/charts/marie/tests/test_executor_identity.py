from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import test_executor_compute as compute
import yaml

FIXTURE = Path(__file__).with_name('executor-headless-identities.json')
HOSTED_NAMESPACE = 'sbx-550e8400-e29b-41d4-a716-446655440000-a1b2c3'


class ExecutorIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chart = compute.ExecutorComputeTest()

    def documents(self, result: subprocess.CompletedProcess[str]) -> list[dict]:
        self.assertEqual(result.returncode, 0, result.stderr)
        return [item for item in yaml.safe_load_all(result.stdout) if item]

    def test_canonical_headless_identity_fixture(self) -> None:
        for case in json.loads(FIXTURE.read_text()):
            with self.subTest(case=case['id']):
                selected = compute.pool(case['pool'], stateful=True)
                documents = self.documents(
                    self.chart.render(
                        [selected],
                        case['release'],
                        {'headlessServiceNamingVersion': case['version']},
                        case['namespace'],
                    )
                )
                stateful = next(
                    item
                    for item in documents
                    if item['kind'] == 'StatefulSet'
                    and item['metadata']['name'].endswith('-executor-' + case['pool'])
                )
                self.assertEqual(
                    stateful['spec']['serviceName'], case['headlessService']
                )
                headless = next(
                    item
                    for item in documents
                    if item['kind'] == 'Service'
                    and item['metadata']['name'] == case['headlessService']
                )
                self.assertEqual(headless['spec']['clusterIP'], 'None')
                self.assertEqual(
                    headless['spec']['selector'],
                    stateful['spec']['selector']['matchLabels'],
                )

    def test_omitted_version_renders_the_same_manifests_as_explicit_v1(self) -> None:
        pools = [compute.pool('cpu'), compute.pool('stateful', stateful=True)]
        default = self.documents(self.chart.render(pools))
        explicit = self.documents(
            self.chart.render(
                pools, executor_values={'headlessServiceNamingVersion': 'v1'}
            )
        )
        self.assertEqual(default, explicit)

    def test_v2_changes_only_the_governing_service_identity(self) -> None:
        selected = [compute.pool('cpu', stateful=True)]
        old = self.documents(self.chart.render(selected))
        new = self.documents(
            self.chart.render(
                selected, executor_values={'headlessServiceNamingVersion': 'v2'}
            )
        )
        new_stateful = next(
            item
            for item in new
            if item['kind'] == 'StatefulSet' and 'executor' in item['metadata']['name']
        )
        old_stateful = next(
            item
            for item in old
            if item['kind'] == 'StatefulSet' and 'executor' in item['metadata']['name']
        )
        new_service_name = new_stateful['spec']['serviceName']
        old_service_name = old_stateful['spec']['serviceName']
        self.assertNotEqual(new_service_name, old_service_name)
        for item in new:
            if (
                item['kind'] == 'StatefulSet'
                and item['metadata']['name'] == new_stateful['metadata']['name']
            ):
                item['spec']['serviceName'] = old_service_name
            if (
                item['kind'] == 'Service'
                and item['metadata']['name'] == new_service_name
            ):
                item['metadata']['name'] = old_service_name
        self.assertEqual(
            old,
            new,
            'workload names, selectors, claim templates and all other manifests must remain unchanged',
        )

    def test_long_namespace_with_short_release_renders_stateful_pools(self) -> None:
        selected = [
            compute.pool('cpu', stateful=True),
            compute.pool('gpu', gpu=True, stateful=True),
        ]
        documents = self.documents(
            self.chart.render(
                selected,
                'compute-test',
                {'headlessServiceNamingVersion': 'v2'},
                HOSTED_NAMESPACE,
            )
        )
        workers = [
            item
            for item in documents
            if item['kind'] == 'StatefulSet'
            and '-executor-' in item['metadata']['name']
        ]
        self.assertEqual(len(workers), 2)
        for worker in workers:
            self.assertLessEqual(len(worker['spec']['serviceName']), 63)
            self.assertLessEqual(len(worker['metadata']['name'] + '-1'), 63)
            self.assertLessEqual(len(worker['metadata']['name'] + '-4294967295'), 63)
            self.assertEqual(
                worker['spec']['volumeClaimTemplates'][0]['metadata']['name'], 'data'
            )

    def test_v1_keeps_rejecting_unsupported_hosted_stateful_identities(self) -> None:
        result = self.chart.render(
            [compute.pool('cpu', stateful=True)], HOSTED_NAMESPACE
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('headless Service name limit', result.stderr)

    def test_v2_services_remain_unique_with_former_legacy_suffix_collision(
        self,
    ) -> None:
        pools = [compute.pool('a', stateful=True), compute.pool('a-headless')]
        for selected in (pools, list(reversed(pools))):
            with self.subTest(order=[item['name'] for item in selected]):
                documents = self.documents(
                    self.chart.render(
                        selected, executor_values={'headlessServiceNamingVersion': 'v2'}
                    )
                )
                names = [
                    item['metadata']['name']
                    for item in documents
                    if item['kind'] == 'Service'
                ]
                self.assertEqual(len(names), len(set(names)))

    def test_v2_zero_default_and_maximum_replica_counts(self) -> None:
        for replicas in [None, 0, 10000]:
            with self.subTest(replicas=replicas):
                selected = compute.pool('cpu', stateful=True)
                if replicas is None:
                    del selected['replicas']
                else:
                    selected['replicas'] = replicas
                self.documents(
                    self.chart.render(
                        [selected],
                        executor_values={'headlessServiceNamingVersion': 'v2'},
                    )
                )

    def test_v2_rejects_hosted_release_names_despite_short_services(self) -> None:
        for replicas in [1, 100, 101]:
            with self.subTest(replicas=replicas):
                selected = {**compute.pool('cpu', stateful=True), 'replicas': replicas}
                result = self.chart.render(
                    [selected], HOSTED_NAMESPACE, {'headlessServiceNamingVersion': 'v2'}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('controller revision label', result.stderr)
        too_long = self.chart.render(
            [compute.pool('cpu-long', stateful=True)],
            HOSTED_NAMESPACE,
            {'headlessServiceNamingVersion': 'v2'},
        )
        self.assertNotEqual(too_long.returncode, 0)
        self.assertIn('controller revision label', too_long.stderr)

    def test_unknown_versions_and_invalid_replica_shapes_reject(self) -> None:
        for version in ['v3', '', 2, False]:
            with self.subTest(version=version):
                result = self.chart.render(
                    [compute.pool('cpu')],
                    executor_values={'headlessServiceNamingVersion': version},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('executor identity:', result.stderr)
        for replicas in [-1, 1.5, True, 'private-invalid-value', 10001]:
            with self.subTest(replicas=replicas):
                selected = {**compute.pool('cpu', stateful=True), 'replicas': replicas}
                result = self.chart.render(
                    [selected], executor_values={'headlessServiceNamingVersion': 'v2'}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('executor identity:', result.stderr)
                self.assertNotIn('private-invalid-value', result.stderr)

    def test_stateful_names_reserve_the_controller_revision_label_budget(self) -> None:
        for version in ['v1', 'v2']:
            for release_length, valid in [(39, True), (40, False)]:
                with self.subTest(version=version, release_length=release_length):
                    result = self.chart.render(
                        [compute.pool('cpu', stateful=True)],
                        'r' * release_length,
                        {'headlessServiceNamingVersion': version},
                    )
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    if valid:
                        documents = self.documents(result)
                        worker = next(
                            item
                            for item in documents
                            if item['kind'] == 'StatefulSet'
                            and item['metadata']['name'].endswith('-executor-cpu')
                        )
                        self.assertLessEqual(
                            len(worker['metadata']['name'] + '-4294967295'), 63
                        )
                    else:
                        self.assertIn('controller revision label', result.stderr)

    def test_v2_rejects_noncanonical_replica_strings(self) -> None:
        for replicas in ['00010', '01000', '00', '01']:
            with self.subTest(replicas=replicas):
                selected = {**compute.pool('cpu', stateful=True), 'replicas': replicas}
                result = self.chart.render(
                    [selected], executor_values={'headlessServiceNamingVersion': 'v2'}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('executor identity:', result.stderr)


if __name__ == '__main__':
    unittest.main()
