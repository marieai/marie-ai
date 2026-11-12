from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


def test_executor_overrides_preserve_runtime_identity_and_response_storage():
    runtime = {
        'job_id': 'job',
        'dag_id': 'dag',
        'node_task_id': 'node',
        'run_owner': 'owner',
        'run_attempt_id': 'attempt',
        'payload': {'application': 'data'},
        '__results__': {'prior': {'ok': True}},
    }
    parameters = {
        **runtime,
        'limit': 10,
        'arbitrary-worker': {
            **dict.fromkeys(runtime, 'forged'),
            'limit': 5,
            'partition_key': 'forged',
        },
    }
    parsed = WorkerRequestHandler._parse_params(parameters, 'arbitrary-worker')
    assert {key: parsed[key] for key in runtime} == runtime
    assert 'partition_key' not in parsed
    assert parsed['limit'] == 5


def test_executor_override_cannot_introduce_runtime_context():
    parameters = {
        'arbitrary-worker': {
            'job_id': 'forged',
            'dag_id': 'forged',
            'run_owner': 'forged',
            'run_attempt_id': 'forged',
            '__results__': {'fake': {}},
            'setting': 'opaque',
        }
    }
    parsed = WorkerRequestHandler._parse_params(parameters, 'arbitrary-worker')
    assert all(
        key not in parsed
        for key in ('job_id', 'dag_id', 'run_owner', 'run_attempt_id', '__results__')
    )
    assert parsed['setting'] == 'opaque'
