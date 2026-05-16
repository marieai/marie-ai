import threading
import time

import pytest
from psycopg import pq
from psycopg_pool import PoolTimeout

from marie.storage.database.postgres import PostgresqlMixin


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeConnection:
    closed = False
    autocommit = True

    class PGConn:
        transaction_status = pq.TransactionStatus.IDLE

    pgconn = PGConn()

    def rollback(self):
        pass


class FakePool:
    min_size = 1
    max_size = 1
    closed = False

    def __init__(self, conn):
        self.conn = conn
        self.available = threading.Event()
        self.available.set()
        self.getconn_calls = 0
        self.putconn_calls = 0

    def getconn(self, timeout=None):
        self.getconn_calls += 1
        if not self.available.wait(timeout):
            raise PoolTimeout("couldn't get a connection after timeout")
        self.available.clear()
        return self.conn

    def putconn(self, conn):
        self.putconn_calls += 1
        self.available.set()

    def get_stats(self):
        available = 1 if self.available.is_set() else 0
        return {
            "pool_min": self.min_size,
            "pool_max": self.max_size,
            "pool_size": 1,
            "pool_available": available,
            "requests_waiting": 0,
        }


class FakeSqlCursor:
    description = None
    rowcount = 1
    closed = False

    def __init__(self):
        self.executed = []

    def execute(self, statement, data=None):
        self.executed.append((statement, data))

    def close(self):
        self.closed = True


class FakeSqlConnection(FakeConnection):
    def __init__(self):
        self.cursor_instance = FakeSqlCursor()
        self.commits = 0

    def cursor(self, *args, **kwargs):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


def make_mixin(timeout: float = 0.25) -> tuple[PostgresqlMixin, FakePool]:
    mixin = object.__new__(PostgresqlMixin)
    mixin.logger = FakeLogger()
    pool = FakePool(FakeConnection())
    mixin.postgreSQL_pool = pool
    mixin._pg_pool_acquire_timeout_seconds = timeout
    mixin._pg_pool_acquire_warn_after_seconds = 60.0
    mixin._pg_pool_acquire_trace_after_seconds = 60.0
    return mixin, pool


def test_get_connection_waits_for_pool_capacity():
    mixin, pool = make_mixin(timeout=1.0)
    pool.available.clear()

    def release_later():
        time.sleep(0.05)
        pool.available.set()

    thread = threading.Thread(target=release_later)
    thread.start()
    started = time.perf_counter()
    conn = mixin._get_connection()
    elapsed = time.perf_counter() - started
    thread.join()

    assert elapsed >= 0.04
    assert conn is pool.conn
    assert pool.getconn_calls == 1

    mixin._close_connection(conn)
    assert pool.putconn_calls == 1
    assert pool.available.is_set()


def test_get_connection_times_out_when_pool_capacity_does_not_recover():
    mixin, _pool = make_mixin(timeout=0.01)

    _pool.available.clear()
    with pytest.raises(PoolTimeout, match="Timed out waiting"):
        mixin._get_connection()


def test_execute_sql_gracefully_can_defer_commit_for_caller_owned_transaction():
    mixin, _pool = make_mixin()
    conn = FakeSqlConnection()

    result = mixin._execute_sql_gracefully(
        "INSERT INTO test VALUES (%s)",
        (1,),
        connection=conn,
        commit=False,
    )

    assert result == 1
    assert conn.commits == 0
    assert conn.cursor_instance.executed == [("INSERT INTO test VALUES (%s)", (1,))]


def test_execute_sql_gracefully_commits_by_default():
    mixin, _pool = make_mixin()
    conn = FakeSqlConnection()

    result = mixin._execute_sql_gracefully(
        "INSERT INTO test VALUES (%s)",
        (1,),
        connection=conn,
    )

    assert result == 1
    assert conn.commits == 1
