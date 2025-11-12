from typing import Dict

import numpy as np
from docarray import DocList

from marie.api.docs import StorageDoc
from marie.logging_core.logger import MarieLogger

from .postgreshandler import PostgreSQLHandler


class PostgreSQLStorage:
    """:class:`PostgreSQLStorage` PostgreSQL-based Storage Indexer."""

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = 5432,
        username: str = "postgres",
        password: str = "123456",
        database: str = "postgres",
        table: str = "default_table",
        min_connections=1,
        max_connections=5,
        traversal_paths: str = "@r",
        return_embeddings: bool = True,
        dry_run: bool = False,
        virtual_shards: int = 128,
        dump_dtype: type = np.float64,
        *args,
        **kwargs,
    ):
        """
        Initialize the PostgreSQLStorage.

        :param hostname: hostname of the machine
        :param port: the port
        :param username: the username to authenticate
        :param password: the password to authenticate
        :param database: the database name
        :param table: the table name to use
        :param traversal_paths: the default traversal path on docs used for indexing, updating and deleting, e.g. ['r'], ['c']
        :param return_embeddings: whether to return embeddings on search or not
        :param dry_run: If True, no database connection will be build.
        :param virtual_shards: the number of shards to distribute
         the data (used when rolling update on Searcher side)
        """
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.table = table
        self.logger = MarieLogger(self.__class__.__name__)
        self.virtual_shards = virtual_shards
        self.dry_run = dry_run
        self.handler = PostgreSQLHandler(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
            table=self.table,
            min_connections=min_connections,
            max_connections=max_connections,
            dry_run=dry_run,
            virtual_shards=virtual_shards,
            dump_dtype=dump_dtype,
        )

        self.default_traversal_paths = traversal_paths
        self.return_embeddings = return_embeddings

    @property
    def dump_dtype(self):
        return self.handler.dump_dtype

    @property
    def size(self):
        """Obtain the size of the table
        .. # noqa: DAR201
        """
        return self.handler.get_size()

    @property
    def snapshot_size(self):
        """Obtain the size of the table
        .. # noqa: DAR201
        """
        return self.handler.get_snapshot_size()

    def add(
        self, docs: DocList[StorageDoc], store_mode: str, parameters: Dict, **kwargs
    ):
        """Add Documents to Postgres
        :param store_mode: how to store the document, valid options content|blob|embedding|doc
        :param docs: list of Documents
        :param parameters: parameters to the request,
        """
        if docs is None:
            return
        self.handler.add(
            docs,
            store_mode,
            **{
                "ref_id": parameters.get("ref_id"),
                "ref_type": parameters.get("ref_type"),
            },
        )

    def similarity_search_with_score(self, query_vector, k=5):
        """
        Returns the top k similar vectors to the query vector.
        """
        return self.handler.similarity_search_with_score(query_vector, k)

    def similarity_search(self, query_vector, k=5):
        """
        Returns the top k similar vectors to the query vector.
        """
        return self.handler.similarity_search(query_vector, k)

    def _get_connection(self):
        """
        Get a database connection from the pool.
        Delegates to the underlying handler.
        """
        return self.handler._get_connection()

    def _close_connection(self, conn):
        """
        Return a database connection to the pool.
        Delegates to the underlying handler.
        """
        return self.handler._close_connection(conn)
