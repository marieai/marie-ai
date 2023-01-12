__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import datetime
import hashlib
import json
from typing import Any, Generator, List, Optional, Tuple

import jsons
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import pool  # noqa: F401

from marie import Document, DocumentArray
from marie.logging.logger import MarieLogger
from marie.numpyencoder import NumpyEncoder


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
    #
    # new_doc.ClearField('embedding')
    # return new_doc.SerializeToString()


def serialize_to_json(content: Any):
    serialized = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ": "),
        ensure_ascii=False,
        indent=2,
        cls=NumpyEncoder,
    )
    return serialized


SCHEMA_VERSION = 4
META_TABLE_NAME = "index_metas"
MODEL_TABLE_NAME = "index_models"


class PostgreSQLHandler:
    """
    Postgres Handler to connect to the database and
     can apply add, update, delete and query.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param collection: the collection name
    :param dry_run: If True, no database connection will be build
    :param virtual_shards: the number of shards to
    distribute the data (used when rolling update on Searcher side)
    :param args: other arguments
    :param kwargs: other keyword arguments
    """

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = 5432,
        username: str = "default_name",
        password: str = "default_pwd",
        database: str = "postgres",
        table: Optional[str] = "default_table",
        max_connections: int = 5,
        dump_dtype: type = np.float64,
        dry_run: bool = False,
        virtual_shards: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = MarieLogger("psq_handler")
        self.table = table
        self.dump_dtype = dump_dtype
        self.virtual_shards = virtual_shards
        self.snapshot_table = f"snapshot_{table}"

        if not dry_run:
            self.postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
            )
            self._init_table()

    def __enter__(self):
        self.connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self._close_connection(self.connection)

    def _init_table(self):
        """
        Use table if exists or create one if it doesn't.

        Create table if needed with id, vecs and metas.
        """
        with self:
            self._create_meta_table()
            self._create_model_table()

            if self._table_exists():
                self._assert_table_schema_version()
                self.logger.info("Using existing table")
            else:
                self._create_table()

    def _execute_sql_gracefully(self, statement, data=tuple()):
        try:
            cursor = self.connection.cursor()
            if data:
                cursor.execute(statement, data)
            else:
                cursor.execute(statement)
        except psycopg2.errors.UniqueViolation as error:
            self.logger.debug(f"Error while executing {statement}: {error}.")

        self.connection.commit()
        return cursor

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
            f"""CREATE TABLE IF NOT EXISTS {self.table} (
                doc_id VARCHAR PRIMARY KEY,
                ref_id VARCHAR(64) not null,
                ref_type VARCHAR(32) not null,
                store_mode VARCHAR(32) not null,
                tags JSONB,
                embedding BYTEA,
                blob BYTEA,
                content JSONB,
                doc BYTEA,
                shard int,
                created_at timestamp with time zone default current_timestamp,
                updated_at timestamp with time zone default current_timestamp,
                is_deleted BOOL DEFAULT FALSE
            );
            INSERT INTO {META_TABLE_NAME} (table_name, schema_version) VALUES (%s, %s);
            INSERT INTO {MODEL_TABLE_NAME} (table_name) VALUES (%s);""",
            (self.table, SCHEMA_VERSION, self.table),
        )

    def _table_exists(self):
        return self._execute_sql_gracefully(
            "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)",
            (self.table,),
        ).fetchall()[0][0]

    def _assert_table_schema_version(self):
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT schema_version FROM {META_TABLE_NAME} WHERE table_name=%s;",
            (self.table,),
        )
        result = cursor.fetchone()
        if result:
            if result[0] != SCHEMA_VERSION:
                raise RuntimeError(
                    "The schema versions of the database "
                    f"(version {result[0]}) and the Executor "
                    f"(version {SCHEMA_VERSION}) do not match. "
                    "Please migrate your data to the latest "
                    "version or use an Executor version with a "
                    "matching schema version."
                )
        else:
            raise RuntimeError(
                "The schema versions of the database "
                "(NO version number) and the Executor "
                f"(version {SCHEMA_VERSION}) do not match."
                "Please migrate your data to the latest version."
            )

    def add(self, docs: DocumentArray, store_mode="content", *args, **kwargs):
        """Insert the documents into the database.

        :param store_mode: how to store the document, valid options content|blob|embedding|doc
        :param docs: list of Documents
        :param args: other arguments
        :param kwargs: other keyword arguments
        :return record: List of Document's id added
        """

        ref_id = kwargs.pop("ref_id")
        ref_type = kwargs.pop("ref_type")

        with self:
            cursor = self.connection.cursor()
            try:
                psycopg2.extras.execute_batch(
                    cursor,
                    f"INSERT INTO {self.table} (doc_id, ref_id, ref_type, store_mode,tags, embedding, blob, "
                    " content, doc, shard, created_at, updated_at) VALUES (%s, %s, %s, %s, %s,"
                    " %s, %s, %s, %s,%s, current_timestamp, current_timestamp)",
                    [
                        (
                            doc.id,
                            ref_id,
                            ref_type,
                            store_mode,
                            serialize_to_json(doc.tags)
                            if doc.tags is not None
                            else None,
                            doc.embedding.astype(self.dump_dtype).tobytes()
                            if store_mode == "embedding" and doc.embedding is not None
                            else None,
                            doc.blob
                            if store_mode == "blob" and doc.blob is not None
                            else None,
                            serialize_to_json(doc.content)
                            if store_mode == "content" and doc.content is not None
                            else None,
                            None,  # TODO : Need to make serialization much faster than what JSON serializer can provider
                            # doc_without_embedding(doc),
                            self._get_next_shard(doc.id),
                        )
                        for doc in docs
                    ],
                )
            except psycopg2.errors.UniqueViolation as e:
                self.logger.warning(
                    f"Document already exists in PSQL database. {e}. Skipping entire transaction..."
                )
                self.connection.rollback()
            self.connection.commit()

    def update(self, docs: DocumentArray, *args, **kwargs):
        """Updated documents from the database.

        :param docs: list of Documents
        :param args: other arguments
        :param kwargs: other keyword arguments
        :return record: List of Document's id after update
        """
        cursor = self.connection.cursor()
        psycopg2.extras.execute_batch(
            cursor,
            f"UPDATE {self.table}             SET embedding = %s,             doc = %s,"
            "             is_deleted = false,              updated_at ="
            " current_timestamp             WHERE doc_id = %s",
            [
                (
                    doc.embedding.astype(self.dump_dtype).tobytes(),
                    doc_without_embedding(doc),
                    doc.id,
                )
                for doc in docs
            ],
        )
        self.connection.commit()

    def prune(self):
        """
        Full deletion of the entries that
        have been marked for soft-deletion
        """
        cursor = self.connection.cursor()
        psycopg2.extras.execute_batch(
            cursor,
            f"DELETE FROM {self.table} WHERE is_deleted = true",
        )
        self.connection.commit()
        return

    def clear(self):
        """
        Full hard-deletion of the entries
        :return:
        """
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {self.table}")
        self.connection.commit()
        return

    def delete(self, docs: DocumentArray, soft_delete=False, *args, **kwargs):
        """Delete document from the database.

        NOTE: This can be a soft-deletion, required by the snapshotting
        mechanism in the PSQLFaissCompound

        For a real delete, use the /cleanup endpoint

        :param docs: list of Documents
        :param args: other arguments
        :param soft_delete:
        :param kwargs: other keyword arguments
        :return record: List of Document's id after deletion
        """
        cursor = self.connection.cursor()

        if soft_delete:
            # self.logger.warning(
            #     'Performing soft-delete. Use /prune or a hard '
            #     'delete to delete the records'
            # )
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
        self.connection.commit()
        return

    def close(self):
        self.postgreSQL_pool.closeall()

    def search(self, docs: DocumentArray, return_embeddings: bool = True, **kwargs):
        """Use the Postgres db as a key-value engine,
        returning the metadata of a document id"""
        if return_embeddings:
            embeddings_field = ", embedding "
        else:
            embeddings_field = ""
        cursor = self.connection.cursor()
        for doc in docs:
            # retrieve metadata
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
                embedding = np.frombuffer(result[1], dtype=self.dump_dtype)
                retrieved_doc.embedding = embedding
            doc.MergeFrom(retrieved_doc)

    def get_trained_model(self):
        """Get the trained index parameters from the Postgres db

        :return: the trained index
        """
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT model_blob, model_checksum FROM {MODEL_TABLE_NAME} WHERE table_name=%s;",
            (self.table,),
        )

        result = cursor.fetchone()
        if result and (result[0] is not None):
            return bytes(result[0]), result[1]
        return None, None

    def save_trained_model(self, model: bytes, checksum: str):
        """Save the trained index parameters into PSQL db

        :param model: the dumps of the trained model
        :param checksum: the checksum of the trained model
        """
        cursor = self.connection.cursor()
        cursor.execute(
            f"UPDATE {MODEL_TABLE_NAME} "
            "SET model_blob = %s, "
            "model_checksum = %s, "
            "updated_at = current_timestamp "
            "where table_name = %s",
            (model, checksum, self.table),
        )
        self.connection.commit()
        self.logger.info("Successfully save model")

    def _close_connection(self, connection):
        # restore it to the pool
        self.postgreSQL_pool.putconn(connection)

    def _get_connection(self):
        # by default psycopg2 is not auto-committing
        # this means we can have rollbacks
        # and maintain ACID-ity
        connection = self.postgreSQL_pool.getconn()
        connection.autocommit = False
        return connection

    def get_size(self):
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table} WHERE is_deleted = false")
        records = cursor.fetchall()
        return records[0][0]

    def _get_next_shard(self, doc_id: str):
        sha = hashlib.sha256()
        sha.update(bytes(doc_id, "utf-8"))
        return int(sha.hexdigest(), 16) % self.virtual_shards

    def snapshot(self):
        """
        Saves the state of the data table in a new table

        Required to be done in two steps becauselast_updated
        1. create table like ... doesn't include data
        2. insert into .. (select ...) doesn't include primary key definitions
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"drop table if exists {self.snapshot_table}; "
                f"create table {self.snapshot_table} "
                f"(like {self.table} including all);"
            )
            self.connection.commit()
            cursor = self.connection.cursor()
            cursor.execute(
                f"insert into {self.snapshot_table} (select * from {self.table});"
            )
            self.connection.commit()
            self.logger.info("Successfully created snapshot")
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error snapshotting: {error}")
            self.connection.rollback()

    def get_snapshot(
        self,
        shards_to_get: List[int],
        include_metas: bool = False,
        filter_deleted: bool = True,
    ):
        """
        Get the data from the snapshot, for a specific range of virtual shards
        """
        return self.get_data_iterator(
            table_name=self.snapshot_table,
            include_metas=include_metas,
            filter_deleted=filter_deleted,
            shards_to_get=shards_to_get,
        )

    def get_document_iterator(
        self,
        limit: int = 0,
        return_embedding: bool = False,
        check_embedding: bool = False,
    ) -> Generator[Document, None, None]:
        """Get the documents from the PSQL database.

        :param limit: the maximal number docs to get
        :param return_embedding: whether filter out the documents without embedding
        :param check_embedding: whether to return embeddings on search
        :return:
        """
        try:
            cursor = self.connection.cursor("doc_iterator")
            cursor.itersize = 10000
            cursor.execute(
                f"SELECT doc_id, doc, embedding from {self.table} WHERE is_deleted = false"
                + (f" limit = {limit}" if limit > 0 else "")
            )
            for sample in cursor:
                doc_id = sample[0]
                if sample[1] is not None:
                    doc = Document(bytes(sample[1]))
                else:
                    doc = Document(id=doc_id)

                if return_embedding and sample[2] is not None:
                    embedding = np.frombuffer(sample[2], dtype=self.dump_dtype)
                    doc.embedding = embedding

                    yield doc
                elif not check_embedding:
                    yield doc

        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error importing snapshot: {error}")
            self.connection.rollback()
        self.connection.commit()

    def get_data_iterator(
        self,
        table_name: Optional[str] = None,
        include_metas: bool = True,
        filter_deleted: bool = True,
        shards_to_get: Optional[List[int]] = None,
    ) -> Generator[Tuple[str, bytes, Optional[bytes]], None, None]:
        """Get the rows from specific table

        :param table_name: the table name to use
        :param include_metas: whether to retrieval document's meta data
        :param filter_deleted: whether to filter documents which has been marked as soft-delete
        :param shards_to_get: the shards list to search
        :return:
        """
        connection = self._get_connection()
        cursor = connection.cursor("dump_iterator")  # server-side cursor
        cursor.itersize = 10000

        try:
            if shards_to_get is not None:
                shards_quoted = tuple(int(shard) for shard in shards_to_get)

                cursor.execute(
                    "SELECT doc_id, embedding"
                    + (", doc " if include_metas else " ")
                    + f"FROM {table_name or self.table} WHERE "
                    + "shard in %s "
                    + ("and is_deleted = false " if filter_deleted else ""),
                    (shards_quoted,),
                )
            else:
                cursor.execute(
                    "SELECT doc_id, embedding"
                    + (", doc " if include_metas else " ")
                    + f"FROM {table_name or self.table} "
                    + ("WHERE is_deleted = false " if filter_deleted else " ")
                )

            for record in cursor:
                yield record[0], np.frombuffer(
                    record[1], dtype=self.dump_dtype
                ) if record[1] is not None else None, record[
                    2
                ] if include_metas else None
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error executing sql statement: {error}")

        self._close_connection(connection)

    def get_snapshot_latest_timestamp(self):
        """Get the timestamp of the snapshot"""
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT MAX(updated_at) FROM {self.snapshot_table}")
            for rec in cursor:
                return rec[0]
        except Exception as e:
            self.logger.error(f"Could not obtain timestamp from snapshot: {e}")

    def get_delta_updates(
        self, shards_to_get, timestamp, filter_deleted: bool = False
    ) -> Generator[Tuple[str, bytes, datetime.datetime], None, None]:
        """
        Get the rows that have changed since the last timestamp

        :param shards_to_get: the shards list to search
        :param timestamp: the last timestamp
        :param filter_deleted: whether to filter out the data which has been marked as soft-delete
        """
        connection = self._get_connection()
        cursor = connection.cursor("delta_generator")  # server-side cursor
        cursor.itersize = 10000
        shards_quoted = tuple(int(shard) for shard in shards_to_get)
        cursor.execute(
            f"SELECT doc_id, embedding, updated_at, is_deleted from {self.table} WHERE shard in %s and updated_at > %s"
            + (" and is_deleted = false" if filter_deleted else ""),
            (shards_quoted, timestamp),
        )

        for rec in cursor:
            second_val = (
                np.frombuffer(rec[1], dtype=self.dump_dtype)
                if rec[1] is not None
                else None
            )
            yield rec[0], second_val, rec[2], rec[3]
        self._close_connection(connection)

    def get_snapshot_size(self):
        """
        Get the size of the snapshot, if it exists.
        else 0
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.snapshot_table} WHERE is_deleted = false"
            )
            records = cursor.fetchall()
            return records[0][0]
        except Exception as e:
            self.logger.warning(f"Could not get size of snapshot: {e}")
        return 0
