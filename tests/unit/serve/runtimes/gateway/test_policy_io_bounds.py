import os
import socket
import time
from types import SimpleNamespace

import psycopg
import pytest


@pytest.mark.parametrize('ready', [False, True])
def test_policy_connection_deadline_closes_silent_or_endless_io(ready):
    from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
        _PolicyConnection,
    )

    reader, writer = socket.socketpair()
    if ready:
        writer.send(b'x')
    finished = []
    connection = object.__new__(_PolicyConnection)
    connection.pgconn = SimpleNamespace(
        socket=reader.fileno(),
        status=psycopg.pq.ConnStatus.OK,
        transaction_status=psycopg.pq.TransactionStatus.ACTIVE,
        finish=lambda: (
            finished.append(True),
            setattr(connection.pgconn, 'status', psycopg.pq.ConnStatus.BAD),
        ),
    )
    connection.policy_read_deadline = time.monotonic() + 0.02

    def pending():
        while True:
            yield psycopg.waiting.Wait.R

    began = time.monotonic()
    try:
        with pytest.raises(
            psycopg.OperationalError, match='Policy database I/O timed out'
        ):
            connection.wait(pending())
        assert time.monotonic() - began < 0.3
        assert finished == [True]
    finally:
        reader.close()
        writer.close()


def test_policy_connection_returns_completed_io():
    from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
        _PolicyConnection,
    )

    reader, writer = socket.socketpair()
    connection = object.__new__(_PolicyConnection)
    connection.pgconn = SimpleNamespace(
        socket=reader.fileno(), status=psycopg.pq.ConnStatus.BAD
    )

    def completed():
        if False:
            yield
        return 'complete'

    try:
        assert connection.wait(completed()) == 'complete'
    finally:
        reader.close()
        writer.close()


def test_policy_pool_replaces_timed_out_connection():
    from psycopg_pool import ConnectionPool

    from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
        _PolicyConnection,
    )

    database_url = os.environ.get('MARIE_LLM_POLICY_TEST_DATABASE_URL')
    if not database_url:
        pytest.skip('Set MARIE_LLM_POLICY_TEST_DATABASE_URL to an owned test database')
    with ConnectionPool(
        database_url,
        connection_class=_PolicyConnection,
        min_size=1,
        max_size=1,
        timeout=2,
    ) as pool:
        pool.wait(timeout=5)
        began = time.monotonic()
        with pytest.raises(
            psycopg.OperationalError, match='Policy database I/O timed out'
        ):
            with pool.connection() as conn:
                conn.policy_read_deadline = time.monotonic() + 0.1
                conn.execute('SELECT pg_sleep(30)')
        assert time.monotonic() - began < 1
        assert conn.closed
        with pool.connection() as replacement:
            assert replacement is not conn
            assert replacement.execute('SELECT 1').fetchone() == (1,)
