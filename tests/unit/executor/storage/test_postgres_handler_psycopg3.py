from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from marie.executor.storage.PostgreSQLStorage.postgreshandler import PostgreSQLHandler


class FakeLogger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeCursor:
    def __init__(self):
        self.executemany_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def executemany(self, query, params):
        self.executemany_calls.append((query, params))


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


def make_handler(conn):
    handler = object.__new__(PostgreSQLHandler)
    handler.logger = FakeLogger()
    handler.schema = None
    handler.table = "documents"
    handler.virtual_shards = 16
    handler._get_connection = lambda: conn
    handler._close_connection = lambda _conn: None
    handler._safe_rollback = lambda _conn: None
    return handler


def test_add_uses_psycopg3_executemany_for_content_rows():
    conn = FakeConnection()
    handler = make_handler(conn)
    doc = SimpleNamespace(
        id="doc-1",
        tags={"type": "unit"},
        embedding=None,
        blob=None,
        content={"value": 1},
    )

    handler.add([doc], store_mode="content", ref_id="ref-1", ref_type="test")

    query, params = conn.cursor_instance.executemany_calls[0]
    assert "%s::jsonb" in query
    assert "%s::vector" in query
    assert params[0][0] == "doc-1"
    assert params[0][5] is None
    assert conn.commits == 1


def test_add_converts_embedding_to_pgvector_literal():
    conn = FakeConnection()
    handler = make_handler(conn)
    doc = SimpleNamespace(
        id="doc-2",
        tags=None,
        embedding=np.array([1, 2], dtype=np.float32),
        blob=None,
        content=None,
    )

    handler.add([doc], store_mode="embedding", ref_id="ref-1", ref_type="test")

    _query, params = conn.cursor_instance.executemany_calls[0]
    assert params[0][5] == "[1.0,2.0]"


@pytest.mark.parametrize('schema', ['marie_scheduler', 'document_storage'])
def test_metadata_tables_use_the_document_storage_schema(schema):
    handler = make_handler(FakeConnection())
    handler.schema = schema
    handler._execute_sql_gracefully = Mock(return_value=[(2,)])

    handler._create_meta_table()
    handler._create_model_table()
    handler._assert_table_schema_version()

    statements = handler._execute_sql_gracefully.call_args_list
    assert f'CREATE TABLE IF NOT EXISTS "{schema}"."metas"' in statements[0].args[0]
    assert f'CREATE TABLE IF NOT EXISTS "{schema}"."models"' in statements[1].args[0]
    assert f'FROM "{schema}"."metas"' in statements[2].args[0]
    assert statements[2].args[1] == (f'{schema}.documents',)


def test_missing_storage_schema_defaults_to_marie_scheduler(monkeypatch):
    setup = Mock()
    monkeypatch.setattr(PostgreSQLHandler, '_setup_storage', setup)

    PostgreSQLHandler(schema=None)

    assert setup.call_args.args[0]['schema'] == 'marie_scheduler'
