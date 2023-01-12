from typing import Dict, List, Optional

import numpy as np

from marie import DocumentArray, Executor, requests
from marie.logging.logger import MarieLogger

from .postgreshandler import PostgreSQLHandler

# https://github.com/jina-ai/executor-hnsw-postgres/blob/main/executor/postgres_indexer.py


class PostgreSQLStorage(Executor):
    """:class:`PostgreSQLStorage` PostgreSQL-based Storage Indexer."""

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        port: int = 5432,
        username: str = "postgres",
        password: str = "123456",
        database: str = "postgres",
        table: str = "default_table",
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

    def add(self, docs: DocumentArray, store_mode: str, parameters: Dict, **kwargs):
        """Add Documents to Postgres
        :param store_mode: how to store the document, valid options content|blob|embedding|doc
        :param docs: list of Documents
        :param parameters: parameters to the request,
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            "traversal_paths", self.default_traversal_paths
        )

        self.handler.add(
            docs[traversal_paths],
            store_mode,
            **{
                "ref_id": parameters.get("ref_id"),
                "ref_type": parameters.get("ref_type"),
            },
        )
