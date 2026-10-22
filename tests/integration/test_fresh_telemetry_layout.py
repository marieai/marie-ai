"""Run directly with unittest; only disposable containers are used."""

from __future__ import annotations

import http.client
import json
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

import yaml


class FreshTelemetryLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.name = f'fresh-telemetry-{uuid4().hex[:10]}'
        cls.password = uuid4().hex
        compose = (cls.root / 'Dockerfiles/docker-compose.clickhouse.yml').read_text()
        cls.system_user = re.search(r'CLICKHOUSE_USER=\$\{CLICKHOUSE_USER:-([^}]+)\}', compose).group(1)
        cls.docker(['network', 'create', '--internal', cls.name])
        cls.addClassCleanup(cls.docker, ['network', 'rm', cls.name])
        cls.docker([
            'run', '--detach', '--network', cls.name, '--name', cls.name,
            '--label', 'codex.telemetry=disposable', '--memory', '2g',
            '--tmpfs', '/var/lib/clickhouse:rw,size=1073741824',
            '-e', f'CLICKHOUSE_USER={cls.system_user}',
            '-e', f'CLICKHOUSE_PASSWORD={cls.password}',
            '-e', 'CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1',
            '--mount', f'type=bind,src={cls.root}/config/clickhouse/telemetry-dictionary.xml,dst=/etc/clickhouse-server/config.d/telemetry-dictionary.xml,readonly',
            'clickhouse/clickhouse-server@sha256:7234d193fbeb3a375f21f0bb99040396f68bb2e82cada4f4947d7fcf9638bbf3',
        ])
        cls.addClassCleanup(cls.docker, ['rm', '--force', '--volumes', cls.name])
        for _ in range(150):
            try:
                cls.sql('SELECT 1')
                break
            except subprocess.CalledProcessError:
                time.sleep(0.1)
        else:
            raise AssertionError('Disposable ClickHouse did not start')
        cls.schemas = sorted(re.findall(
            r'- \./(config/clickhouse/[^:\s]+\.sql):/docker-entrypoint-initdb\.d/([^:\s]+)', compose
        ), key=lambda item: item[1])
        assert cls.schemas, 'No fresh initialization SQL is mounted'
        for source, _ in cls.schemas:
            cls.sql((cls.root / source).read_text())

    @staticmethod
    def docker(args: list[str], data: str | None = None) -> str:
        result = subprocess.run(['docker', *args], input=data, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, ['docker', args[0]], output=result.stdout, stderr=result.stderr)
        return result.stdout

    @classmethod
    def sql(cls, statement: str) -> str:
        return cls.docker([
            'exec', '-i', '-e', f'CLICKHOUSE_PASSWORD={cls.password}', cls.name,
            'clickhouse-client', '--user', cls.system_user, '--multiquery',
        ], statement)

    def test_fresh_bootstrap_has_one_telemetry_store(self) -> None:
        self.assertEqual(self.sql("SELECT count() FROM system.databases WHERE name='otel' FORMAT TSV").strip(), '0')
        self.assertEqual(self.sql("SELECT count() FROM system.tables WHERE database='telemetry' AND name IN ('spans','logs','metrics_gauge','metrics_sum','metrics_histogram','metrics_summary','metrics_exponential_histogram') FORMAT TSV").strip(), '7')

    def test_only_two_application_databases_include_pricing_and_helpers(self) -> None:
        self.assertEqual(self.sql("SELECT name FROM system.databases WHERE name NOT IN ('default','system','information_schema','INFORMATION_SCHEMA') ORDER BY name FORMAT TSV").strip(), 'telemetry\nworkspace_query')
        self.assertEqual(self.sql("SELECT count() FROM system.tables WHERE database='telemetry' AND name IN ('model_prices','model_price_sync_runs','model_price_dict','model_names','spans_trace_id_ts','capacity_ts') FORMAT TSV").strip(), '6')

    def test_trace_lookup_and_model_catalog_populate_from_new_spans(self) -> None:
        self.sql("""INSERT INTO telemetry.spans(Timestamp,TraceId,SpanId,SpanAttributes)
VALUES(now64(9),'lookup-contract','lookup-span',map('llm.model_name','lookup-model','llm.provider','lookup-provider','llm.token_count.prompt','10'));""")
        self.assertEqual(self.sql("SELECT count() FROM telemetry.spans_trace_id_ts WHERE TraceId='lookup-contract' AND Start<=End FORMAT TSV").strip(), '1')
        self.assertEqual(self.sql("SELECT count() FROM telemetry.model_names WHERE model='lookup-model' FORMAT TSV").strip(), '1')

    def test_fresh_system_identity_is_marie_without_default_login(self) -> None:
        self.assertEqual(self.system_user, 'marie')
        self.assertEqual(self.sql('SELECT currentUser() FORMAT TSV').strip(), 'marie')
        self.assertEqual(self.sql("SELECT count() FROM system.users WHERE name='default' FORMAT TSV").strip(), '0')

    def test_legacy_initialization_is_not_reintroduced_on_reapply(self) -> None:
        for source, _ in self.schemas:
            self.sql((self.root / source).read_text())
        self.assertEqual(self.sql("SELECT count() FROM system.databases WHERE name='otel' FORMAT TSV").strip(), '0')

    def test_event_projection_keeps_insert_time_materialization(self) -> None:
        self.assertEqual(self.sql("SELECT engine FROM system.tables WHERE database='telemetry' AND name='events_mv' FORMAT TSV").strip(), 'MaterializedView')

    def test_event_projection_preserves_alert_and_event_fields(self) -> None:
        self.sql("""INSERT INTO telemetry.logs (Timestamp,ServiceName,Body,SeverityText,LogAttributes)
VALUES(now64(9),'fresh-test','failure','ERROR',map('event.id','cutover-event','event.type','job.failed','event.source','scheduler','event.source_type','job','job.id','cutover-job','job.tag','FAILED','status','failed','payload.message','visible-message','payload.metadata','{}'));""")
        value = self.sql("SELECT event_type, job_id, message, severity FROM telemetry.events_mv WHERE event_id='cutover-event' FORMAT TSV").strip()
        self.assertEqual(value, 'job.failed\tcutover-job\tvisible-message\tERROR')
        self.assertEqual(self.sql("SELECT count() FROM telemetry.error_logs WHERE ServiceName='fresh-test' FORMAT TSV").strip(), '1')

    def test_annotations_keep_workspace_alias_without_another_store(self) -> None:
        self.sql("INSERT INTO telemetry.annotations(trace_id,span_id,workspace_id,name) VALUES('fresh-trace','fresh-span','fresh-workspace','quality')")
        self.assertEqual(self.sql("SELECT project_id FROM telemetry.span_annotations WHERE trace_id='fresh-trace' FORMAT TSV").strip(), 'fresh-workspace')

    def test_capacity_projection_keeps_executor_payload_semantics(self) -> None:
        self.sql("""INSERT INTO telemetry.logs(Timestamp,ServiceName,LogAttributes)
VALUES(now64(9),'capacity-check',map('job.tag','RESOURCE_EXECUTOR_UPDATED','payload.metadata','{"capacity":[["executor",{"capacity":10,"used":3,"available":7,"holder_count":2}]]}'));""")
        self.assertEqual(self.sql('SELECT total_capacity,used,available,holders FROM telemetry.capacity_ts FORMAT TSV').strip(), '10\t3\t7\t2')

    def test_actual_collector_writes_every_signal_without_recreating_legacy_storage(self) -> None:
        collector = self.name + '-collector'
        with tempfile.TemporaryDirectory(prefix='fresh-collector-') as folder:
            config = yaml.safe_load((self.root / 'config/clickstack/otel-collector-config.yml').read_text())
            self.assertFalse(config['exporters']['clickhouse']['create_schema'])
            config['exporters']['clickhouse']['endpoint'] = f'tcp://{self.name}:9000'
            config['processors']['batch']['timeout'] = '100ms'
            config_path = Path(folder) / 'collector.yaml'
            config_path.write_text(yaml.safe_dump(config))
            self.docker([
                'run', '--detach', '--network', self.name, '--name', collector,
                '--label', 'codex.telemetry=disposable', '--memory', '512m',
                '-e', f'CLICKHOUSE_USERNAME={self.system_user}', '-e', f'CLICKHOUSE_PASSWORD={self.password}',
                '-e', 'DEPLOYMENT_ENV=synthetic-test',
                '--mount', f'type=bind,src={config_path},dst=/etc/collector.yaml,readonly',
                'otel/opentelemetry-collector-contrib:0.149.0', '--config=/etc/collector.yaml',
            ])
            try:
                ip = self.docker(['inspect', '--format', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}', collector]).strip()
                for _ in range(100):
                    try:
                        connection = http.client.HTTPConnection(ip, 13133, timeout=1)
                        connection.request('GET', '/')
                        response = connection.getresponse()
                        response.read()
                        connection.close()
                        if response.status == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.1)
                else:
                    self.fail('Disposable collector did not become healthy')
                now = str(time.time_ns())
                attrs = lambda values: [{'key': key, 'value': {'stringValue': value}} for key, value in values.items()]
                resource = {'attributes': attrs({'service.name': 'collector-cutover',
                    'marie.organization_id': '11111111-1111-4111-8111-111111111111',
                    'marie.workspace_id': 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1',
                    'marie.identity_state': 'verified'})}
                trace_id, span_id = uuid4().hex, uuid4().hex[:16]

                def send(path: str, body: dict) -> None:
                    connection = http.client.HTTPConnection(ip, 4318, timeout=5)
                    connection.request('POST', path, body=json.dumps(body), headers={'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    payload = response.read().decode()
                    connection.close()
                    self.assertEqual(response.status, 200, payload)

                send('/v1/traces', {'resourceSpans': [{'resource': resource, 'scopeSpans': [{'spans': [{
                    'traceId': trace_id, 'spanId': span_id, 'name': 'collector-cutover', 'kind': 1,
                    'startTimeUnixNano': now, 'endTimeUnixNano': str(int(now) + 1000000), 'status': {'code': 2},
                }]}]}]})
                send('/v1/logs', {'resourceLogs': [{'resource': resource, 'scopeLogs': [{'logRecords': [{
                    'timeUnixNano': now, 'severityNumber': 17, 'severityText': 'ERROR',
                    'body': {'stringValue': 'collector-cutover'}, 'traceId': trace_id, 'spanId': span_id,
                    'attributes': attrs({'event.id': 'collector-cutover', 'event.type': 'job.failed', 'event.source_type': 'job'}),
                }]}]}]})
                point = {'timeUnixNano': now, 'startTimeUnixNano': now}
                metrics = [
                    {'name': 'collector-gauge', 'gauge': {'dataPoints': [{**point, 'asDouble': 12.5}]}},
                    {'name': 'collector-sum', 'sum': {'aggregationTemporality': 2, 'isMonotonic': True,
                        'dataPoints': [{**point, 'asDouble': 12.5}]}},
                    {'name': 'collector-histogram', 'histogram': {'aggregationTemporality': 2,
                        'dataPoints': [{**point, 'count': '2', 'sum': 3, 'bucketCounts': ['1', '1'], 'explicitBounds': [1.5]}]}},
                    {'name': 'collector-summary', 'summary': {'dataPoints': [{**point, 'count': '2', 'sum': 3,
                        'quantileValues': [{'quantile': 0.5, 'value': 1.5}]}]}},
                    {'name': 'collector-exponential_histogram', 'exponentialHistogram': {'aggregationTemporality': 2,
                        'dataPoints': [{**point, 'count': '2', 'sum': 3, 'scale': 0, 'zeroCount': '0',
                            'positive': {'offset': 0, 'bucketCounts': ['2']}}]}},
                ]
                send('/v1/metrics', {'resourceMetrics': [{'resource': resource, 'scopeMetrics': [{'metrics': metrics}]}]})
                for _ in range(200):
                    counts = [int(self.sql(f"SELECT count() FROM telemetry.{table} WHERE {condition} FORMAT TSV").strip())
                        for table, condition in [('spans', "SpanName='collector-cutover'"), ('logs', "Body='collector-cutover'")]
                        + [(f'metrics_{kind}', f"MetricName='collector-{kind}'") for kind in ('gauge','sum','histogram','summary','exponential_histogram')]]
                    if counts == [1] * 7:
                        break
                    time.sleep(0.1)
                else:
                    logs = self.docker(['logs', '--tail', '25', collector]).replace(self.password, '[redacted]')
                    self.fail(f'Collector did not persist all signals: {counts}; {logs}')
                self.assertEqual(self.sql("SELECT identity_state,organization_id FROM telemetry.spans WHERE SpanName='collector-cutover' FORMAT TSV").strip(), 'unattributed')
                self.assertEqual(self.sql("SELECT count() FROM system.databases WHERE name='otel' FORMAT TSV").strip(), '0')
            finally:
                self.docker(['rm', '--force', '--volumes', collector])


if __name__ == '__main__':
    unittest.main()
