import asyncio

from psycopg.types.json import Jsonb

from marie.assets.tracker import AssetTracker


class FakeCursor:
    def __init__(self):
        self.execute_calls = []
        self._next_id = 0
        self.closed = False

    def execute(self, query, params=None):
        self.execute_calls.append((query, params))

    def fetchone(self):
        self._next_id += 1
        return (self._next_id,)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FailingCursor(FakeCursor):
    def execute(self, query, params=None):
        super().execute(query, params)
        raise RuntimeError("db failed")


class FailingConnection(FakeConnection):
    def __init__(self):
        super().__init__()
        self.cursor_instance = FailingCursor()


class FakeStorageHandler:
    def __init__(self, connection):
        self.connection = connection
        self.closed_connections = []

    def _get_connection(self):
        return self.connection

    def _close_connection(self, connection):
        self.closed_connections.append(connection)


def test_record_materializations_uses_psycopg3_jsonb_adapters():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    conn = FakeConnection()
    tracker = AssetTracker(FakeStorageHandler(conn), {})

    try:
        future = tracker.record_materializations(
            storage_event_id=42,
            assets=[
                {
                    "asset_key": "ocr/text",
                    "version": "v:sha256:abc",
                    "size_bytes": 1024,
                    "checksum": "sha256:abc",
                    "kind": "text",
                    "uri": "file:///tmp/output.txt",
                    "metadata": {"language": "en"},
                }
            ],
            job_id="job-1",
            dag_id="dag-1",
        )

        assert loop.run_until_complete(future) == [(1, "ocr/text")]
    finally:
        tracker._db_executor.shutdown(wait=True)
        loop.close()
        asyncio.set_event_loop(None)

    registry_params = conn.cursor_instance.execute_calls[0][1]
    materialization_params = conn.cursor_instance.execute_calls[1][1]

    assert isinstance(registry_params[2], Jsonb)
    assert isinstance(materialization_params[10], Jsonb)
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_record_materializations_retrieves_fire_and_forget_exceptions():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    conn = FailingConnection()
    tracker = AssetTracker(FakeStorageHandler(conn), {})
    future = None

    try:
        future = tracker.record_materializations(
            storage_event_id=None,
            assets=[{"asset_key": "ocr/text", "kind": "text", "metadata": {}}],
            job_id="job-1",
        )

        while not future.done():
            loop.run_until_complete(asyncio.sleep(0.01))

        assert conn.rollbacks == 1
        assert getattr(future, "_log_traceback", True) is False
    finally:
        if future is not None and future.done():
            try:
                future.exception()
            except asyncio.CancelledError:
                pass
        tracker._db_executor.shutdown(wait=True)
        loop.close()
        asyncio.set_event_loop(None)
