from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_one_compose_file_uses_existing_credentials() -> None:
    result = subprocess.run(
        ['docker', 'compose', '--env-file', '/dev/null', '--project-directory', str(ROOT),
         '-f', str(ROOT / 'Dockerfiles/docker-compose.clickhouse.yml'), 'config',
         '--no-interpolate', '--no-env-resolution'], text=True, capture_output=True, check=True,
    )
    config = yaml.safe_load(result.stdout)
    assert 'clickhouse-query-schema' in config['services']
    job = config['services']['clickhouse-query-schema']
    assert job['environment']['CLICKHOUSE_PASSWORD'] == '${CLICKHOUSE_PASSWORD:-}'
    assert job['environment']['CLICKHOUSE_USER'] == '${CLICKHOUSE_USER:-default}'
    assert 'secrets' not in job
    assert not (ROOT / 'Dockerfiles/docker-compose.clickhouse-query-explorer.yml').exists()


def test_helm_reuses_the_existing_schema_job() -> None:
    assert not (ROOT / 'deploy/helm/charts/marie/charts/clickhouse/templates/query-explorer-job.yaml').exists()
    result = subprocess.run(
        ['helm', 'template', 'query-fixture', str(ROOT / 'deploy/helm/charts/marie'),
         '--set', 'clickhouse.enabled=true', '--set', 'clickhouse.queryExplorer.enabled=true',
         '--set', 'clickhouse.queryExplorer.readerSecret.name=synthetic-reader'],
        text=True, capture_output=True, check=True,
    )
    docs = list(yaml.safe_load_all(result.stdout))
    jobs = [doc for doc in docs if doc and doc.get('kind') == 'Job' and 'clickhouse' in doc['metadata']['name']]
    assert len(jobs) == 1
    assert 'bootstrap.sh' in str(jobs[0]['spec']['template']['spec']['containers'][0]['command'])
    assert 'query-bootstrap' not in result.stdout
