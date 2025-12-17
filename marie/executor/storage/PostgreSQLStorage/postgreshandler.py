import json
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import pool  # noqa: F401
from psycopg2.extensions import AsIs, register_adapter

from marie import Document, DocumentArray
from marie.logging_core.logger import MarieLogger
from marie.numpyencoder import NumpyEncoder
from marie.storage.database.postgres import PostgresqlMixin
from marie.storage.pgvector.psycopg2 import register_vector
from marie.utils.json import to_json


def _adapt_np_array(np_array):
    return AsIs(tuple(np_array))


register_adapter(np.ndarray, _adapt_np_array)


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    if True:
        serialized = json.dumps(
            new_doc.content,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=2,
            cls=NumpyEncoder,
        )
        return serialized


SCHEMA_VERSION = 2
META_TABLE_NAME = "metas"
MODEL_TABLE_NAME = "models"


class PostgreSQLHandler(PostgresqlMixin):
    """
    Postgres Handler to connect to the database and
     can apply add, update, delete and query. It inherits robust connection
     management from PostgresqlMixin.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param table: the table name
    :param min_connections: the minimum number of connections
    :param max_connections: the maximum number of connections
    :param dump_dtype: the numpy dtype for embeddings
    :param dry_run: If True, no database connection will be built
    :param virtual_shards: the number of shards
    """

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = 5432,
        username: str = "default_name",
        password: str = "default_pwd",
        database: str = "postgres",
        table: Optional[str] = "default_table",
        min_connections: int = 1,
        max_connections: int = 5,
        dump_dtype: type = np.float64,
        dry_run: bool = False,
        virtual_shards: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = MarieLogger("psq_handler")
        self.dump_dtype = dump_dtype
        self.virtual_shards = virtual_shards
        self.snapshot_table = f"snapshot_{table}"

        # Track vector type registration state
        self._vector_registered = False
        self._vector_registration_attempted = False

        if not dry_run:
            config = {
                "hostname": hostname,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
                "default_table": table,
                "max_connections": max_connections,
                "min_connections": min_connections,
            }
            # Use the robust setup method from the mixin
            self._setup_storage(
                config, create_table_callback=self._create_versioned_tables
            )

    def _create_versioned_tables(self, table_name: str):
        """Callback to create all necessary tables with versioning."""
        self._create_meta_table()
        self._create_model_table()

        if self._table_exists():
            self._assert_table_schema_version()
            self.logger.info("Using existing table")
        else:
            self._create_table()

    def _create_meta_table(self):
        self._execute_sql_gracefully(
            f"""CREATE TABLE IF NOT EXISTS {META_TABLE_NAME} (
                table_name varchar,
                schema_version integer
            );"""
        )

    def _create_model_table(self):
        self._execute_sql_gracefully(
            f"""CREATE TABLE IF NOT EXISTS {MODEL_TABLE_NAME} (
                table_name varchar,
                model_blob BYTEA,
                model_checksum varchar,
                updated_at timestamp with time zone default current_timestamp
            );"""
        )

    def _create_table(self):
        self._execute_sql_gracefully(
            f"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS {self.table} (
                event_id SERIAL PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                ref_id VARCHAR(64) not null,
                ref_type VARCHAR(64) not null,
                store_mode VARCHAR(32) not null,
                tags JSONB,
                embedding vector,
                blob BYTEA,
                content JSONB,
                doc BYTEA,
                shard int,
                created_at timestamp with time zone default current_timestamp,
                updated_at timestamp with time zone default current_timestamp,
                is_deleted BOOL DEFAULT FALSE
            );
            INSERT INTO {META_TABLE_NAME} (table_name, schema_version) VALUES ('{self.table}', {SCHEMA_VERSION});
            INSERT INTO {MODEL_TABLE_NAME} (table_name) VALUES ('{self.table}');"""
        )

        # Now that the extension is created, re-attempt vector type registration
        if not self._vector_registered:
            conn = super()._get_connection()
            try:
                self._vector_registered = register_vector(conn)
                if self._vector_registered:
                    self.logger.info(
                        "pgvector type registered after extension creation"
                    )
            finally:
                self._close_connection(conn)

    def _assert_table_schema_version(self):
        result = self._execute_sql_gracefully(
            f"SELECT schema_version FROM {META_TABLE_NAME} WHERE table_name=%s;",
            (self.table,),
        )
        if result:
            if result[0][0] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"DB schema version {result[0][0]} does not match Executor version {SCHEMA_VERSION}."
                )
        else:
            raise RuntimeError(f"No schema version found for table {self.table}.")

    def add(self, docs: DocumentArray, store_mode="content", *args, **kwargs):
        """Insert the documents into the database."""
        if "ref_id" not in kwargs or "ref_type" not in kwargs:
            raise ValueError("ref_id and ref_type must be provided in kwargs.")

        ref_id = kwargs.pop("ref_id")
        ref_type = kwargs.pop("ref_type")

        conn = None
        try:
            conn = self._get_connection()
            query_obj = [
                (
                    doc.id,
                    ref_id,
                    ref_type,
                    store_mode,
                    to_json(doc.tags) if doc.tags is not None else None,
                    (
                        doc.embedding
                        if store_mode == "embedding" and doc.embedding is not None
                        else None
                    ),
                    doc.blob if store_mode == "blob" and doc.blob is not None else None,
                    (
                        to_json(doc.content)
                        if store_mode == "content" and doc.content is not None
                        else None
                    ),
                    None,
                    self._get_next_shard(doc.id),
                )
                for doc in docs
            ]

            with conn.cursor() as cursor:
                psycopg2.extras.execute_batch(
                    cursor,
                    f"INSERT INTO {self.table} (doc_id, ref_id, ref_type, store_mode,tags, embedding, blob, "
                    " content, doc, shard, created_at, updated_at) VALUES (%s, %s, %s, %s, %s,"
                    " %s, %s, %s, %s,%s, current_timestamp, current_timestamp)",
                    query_obj,
                )
            conn.commit()
        except psycopg2.errors.UniqueViolation as e:
            self.logger.warning(f"Document already exists. Skipping. Error: {e}")
            self._safe_rollback(conn)
        except Exception as e:
            self.logger.error(f"Error while inserting documents: {e}")
            self._safe_rollback(conn)
            raise
        finally:
            self._close_connection(conn)

    def update(self, docs: DocumentArray, *args, **kwargs):
        """Update documents in the database."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                psycopg2.extras.execute_batch(
                    cursor,
                    f"UPDATE {self.table} SET embedding = %s, doc = %s,"
                    " is_deleted = false, updated_at = current_timestamp WHERE doc_id = %s",
                    [
                        (
                            doc.embedding.astype(self.dump_dtype).tobytes(),
                            doc_without_embedding(doc),
                            doc.id,
                        )
                        for doc in docs
                    ],
                )
            conn.commit()
        finally:
            self._close_connection(conn)

    def prune(self):
        """Hard-delete entries marked for soft-deletion."""
        self._execute_sql_gracefully(
            f"DELETE FROM {self.table} WHERE is_deleted = true"
        )

    def clear(self):
        """Full hard-deletion of all entries."""
        self._execute_sql_gracefully(f"DELETE FROM {self.table}")

    def delete(self, docs: DocumentArray, soft_delete=False, *args, **kwargs):
        """Delete documents from the database."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if soft_delete:
                    psycopg2.extras.execute_batch(
                        cursor,
                        f"UPDATE {self.table} SET is_deleted = true, updated_at = current_timestamp WHERE doc_id = %s;",
                        [(doc.id,) for doc in docs],
                    )
                else:
                    psycopg2.extras.execute_batch(
                        cursor,
                        f"DELETE FROM {self.table} WHERE doc_id = %s;",
                        [(doc.id,) for doc in docs],
                    )
            conn.commit()
        finally:
            self._close_connection(conn)

    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, "postgreSQL_pool") and self.postgreSQL_pool:
            self.postgreSQL_pool.closeall()

    def search(self, docs: DocumentArray, return_embeddings: bool = True, **kwargs):
        """Use the Postgres db as a key-value engine to retrieve document metadata."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                for doc in docs:
                    embeddings_field = ", embedding" if return_embeddings else ""
                    cursor.execute(
                        f"SELECT doc {embeddings_field} FROM {self.table} WHERE doc_id = %s and is_deleted = false;",
                        (doc.id,),
                    )
                    result = cursor.fetchone()

                    if result is None:
                        continue

                    data = bytes(result[0])
                    retrieved_doc = Document(data)
                    if return_embeddings and result[1] is not None:
                        retrieved_doc.embedding = np.frombuffer(
                            result[1], dtype=self.dump_dtype
                        )
                    doc.MergeFrom(retrieved_doc)
        finally:
            self._close_connection(conn)

    def get_size(self):
        """Get the number of non-deleted documents."""
        results = self._execute_sql_gracefully(
            f"SELECT count(*) FROM {self.table} WHERE is_deleted = false"
        )
        return results[0][0] if results else 0

    def _get_next_shard(self, doc_id: str) -> int:
        """Get the next shard to use for the document."""
        return hash(doc_id) % self.virtual_shards

    def _get_connection(self):
        """
        Get a connection from the pool and register the vector type.
        This extends the behavior of the mixin's _get_connection.
        """
        # Get a standard connection from the mixin
        conn = super()._get_connection()

        # Only attempt vector registration once per handler instance
        # This avoids repeated failures if pgvector isn't installed
        if not self._vector_registration_attempted:
            self._vector_registration_attempted = True
            self._vector_registered = register_vector(conn)
            if not self._vector_registered:
                self.logger.warning(
                    "pgvector extension not available; vector operations will not work"
                )

        return conn
