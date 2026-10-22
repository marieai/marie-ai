"""Run against an explicitly supplied, disposable PostgreSQL database."""

import json
import os
from contextlib import asynccontextmanager

import psycopg
import pytest
from psycopg.rows import tuple_row

from marie.scheduler.repository.failure_report import read_failure_report
from marie.storage.database.postgres_pool import AsyncPostgresConnection


async def test_postgres_export_correlates_retained_failure_after_successful_retry():
    port = os.getenv('M3TOP_FAILURE_REPORT_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip('Disposable failure-report PostgreSQL fixture required')
    async with await psycopg.AsyncConnection.connect(
        host='127.0.0.1',
        port=int(port),
        user='postgres',
        dbname='postgres',
        autocommit=True,
        row_factory=tuple_row,
    ) as connection:
        async with connection.transaction(force_rollback=True):
            await connection.execute('CREATE SCHEMA marie_scheduler')
            await connection.execute('''
                CREATE TABLE marie_scheduler.job (
                    id uuid PRIMARY KEY, dag_id uuid, name text, data jsonb,
                    state text, retry_count int, retry_limit int, run_attempt_id uuid,
                    created_on timestamptz DEFAULT now(), started_on timestamptz,
                    completed_on timestamptz, output jsonb
                );
                CREATE TABLE marie_scheduler.dag (
                    id uuid PRIMARY KEY, submission_name text, planner text,
                    project_id text, ref_type text, ref_id text, priority int, task_count int
                );
                CREATE TABLE marie_scheduler.kv_store_worker (
                    namespace text, key text, value jsonb, is_deleted bool
                );
                CREATE TABLE marie_scheduler.kv_store_worker_history (
                    history_id bigint PRIMARY KEY, namespace text, key text, value jsonb,
                    change_time timestamptz, operation text
                );
                CREATE TABLE marie_scheduler.job_history (
                    history_id bigint, id uuid, history_created_on timestamptz,
                    state text, run_attempt_id uuid, retry_count int, retry_limit int,
                    started_on timestamptz, completed_on timestamptz, output jsonb
                );
                CREATE TABLE marie_scheduler.job_attempt (
                    run_attempt_id uuid PRIMARY KEY, job_id uuid, run_owner text,
                    scheduler_lease_owner text, gateway_instance_id text, executor text,
                    attempt_state text, activated_at timestamptz,
                    dispatch_started_at timestamptz, dispatch_confirmed_at timestamptz,
                    dispatch_error text, terminal_at timestamptz, terminal_status text,
                    terminal_work_state text, terminal_source text, terminal_accepted bool,
                    terminal_reject_reason text, terminal_gateway_instance_id text,
                    terminal_scheduler_lease_owner text, recovery_at timestamptz,
                    recovery_state text, recovery_reason text, updated_on timestamptz
                )
            ''')
            job_id = '06a68ba1-5890-7896-8000-30e4db21aeef'
            attempt_id = '84f69f16-ceb6-7c75-8000-37d2d2acb56e'
            worker_key = 'marie_internal/job_info_' + job_id
            await connection.execute(
                '''
                INSERT INTO marie_scheduler.job (id, name, state, retry_count, retry_limit, data, output)
                VALUES (%s::uuid, 'gen5_extract', 'completed', 1, 3, '{"payload":"private-document"}',
                        '{"document":"private-output"}')
            ''',
                (job_id,),
            )
            value = {
                'status': 'FAILED',
                'message': 'Job failed.',
                'run_attempt_id': attempt_id,
                'runtime_env_json': json.dumps(
                    {
                        'attributes': {
                            'host': '192.168.106.75',
                            'executor': 'DocumentAnnotatorLLMExecutor',
                            'runtime_name': 'annotator_llm/rep-0',
                            'executor_endpoint': '/annotator/llm',
                        },
                        'credentials': 'private-environment',
                        'error': {
                            'type': 'QueueTaskError',
                            'message': 'call_timeout',
                            'traceback': 'Traceback\nworker.py:412\ncall_timeout',
                            'traceback_truncated': True,
                            'payload': 'private-error-payload',
                            'request_id': 'request-a',
                            'primary_task_id': 'task-a',
                            'failed_count': 1,
                            'failed_tasks': [
                                {
                                    'task_id': 'task-a',
                                    'error': {
                                        'type': 'QueueTaskError',
                                        'category': 'call_timeout',
                                        'state': 'outcome_unknown',
                                    },
                                }
                            ],
                        },
                    }
                ),
            }
            await connection.execute(
                '''
                INSERT INTO marie_scheduler.kv_store_worker_history
                VALUES (912, 'job', %s, %s::jsonb, '2026-09-30T14:41:28Z', 'UPDATE')
            ''',
                (worker_key, json.dumps(value)),
            )
            await connection.execute(
                '''
                INSERT INTO marie_scheduler.kv_store_worker_history
                SELECT id, 'job', %s, '{"status":"SUCCEEDED"}'::jsonb,
                       '2026-09-30T15:00:00Z'::timestamptz + id * interval '1 second', 'UPDATE'
                FROM generate_series(1000, 1250) id
            ''',
                (worker_key,),
            )
            await connection.execute(
                '''
                INSERT INTO marie_scheduler.job_attempt
                    (job_id, run_attempt_id, attempt_state, terminal_status, terminal_accepted, activated_at)
                VALUES (%s::uuid, %s::uuid, 'terminal', 'failed', true, '2026-09-30T14:40:00Z')
            ''',
                (job_id, attempt_id),
            )
            await connection.execute(
                '''
                INSERT INTO marie_scheduler.job_attempt (job_id, run_attempt_id, attempt_state, activated_at)
                SELECT %s::uuid, gen_random_uuid(), 'terminal', '2026-09-30T15:00:00Z'
                FROM generate_series(1, 251)
            ''',
                (job_id,),
            )

            class Pool:
                @asynccontextmanager
                async def acquire(self):
                    yield AsyncPostgresConnection(connection)

            report = await read_failure_report(Pool(), job_id, 912)
            assert report['job']['scheduler_state'] == 'completed'
            assert report['selected_event']['status'] == 'FAILED'
            assert report['selected_event']['error']['request_id'] == 'request-a'
            assert (
                report['selected_event']['error']['failed_tasks'][0]['error']['state']
                == 'outcome_unknown'
            )
            assert report['selected_event']['executor_host'] == '192.168.106.75'
            assert report['selected_event']['error']['traceback'].endswith(
                'call_timeout'
            )
            assert report['selected_attempt']['run_attempt_id'] == attempt_id
            assert report['selected_attempt']['terminal_accepted'] is True
            assert len(report['worker_history']) == 250
            assert report['truncated']['worker_history'] is True
            assert report['truncated']['attempts'] is True
            assert report['selected_event']['error']['traceback_truncated'] is True
            assert report['truncated']['text'] is True
            assert 'private' not in repr(report)
            with pytest.raises(LookupError):
                await read_failure_report(Pool(), job_id, 999)
            assert (
                await read_failure_report(
                    Pool(), '00000000-0000-0000-0000-000000000000', 912
                )
                is None
            )
