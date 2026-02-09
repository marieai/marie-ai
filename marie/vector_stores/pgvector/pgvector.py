"""PGVector store implementation for unified multimodal RAG.

This module provides a PostgreSQL-based vector store using the pgvector extension.
It supports unified text + image embeddings via jina-embeddings-v4 in the same space.

Key features:
- Single unified table for text + images
- HNSW index for fast similarity search
- Async support via asyncpg
- Metadata filtering
- Source-based filtering for multi-tenant RAG
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import PrivateAttr

from marie._core.schema import BaseNode, TextNode
from marie.logging_core.logger import MarieLogger
from marie.storage.pgvector.utils import from_db, to_db
from marie.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = MarieLogger("marie.vector_stores.pgvector").logger


class PGVectorStore(BasePydanticVectorStore):
    """PostgreSQL vector store using pgvector extension.

    Unified implementation for multimodal RAG that stores text and image
    embeddings in the same table (enabled by jina-embeddings-v4 which
    produces embeddings in the same vector space for both modalities).

    Example:
        ```python
        store = PGVectorStore(
            connection_string="postgresql://user:pass@localhost/db",
            table_name="rag_vectors",
            embedding_dim=2048,
        )

        # Initialize (creates table and indexes)
        await store.initialize()

        # Add nodes
        node_ids = await store.async_add(nodes)

        # Search
        result = await store.aquery(
            VectorStoreQuery(
                query_embedding=embedding,
                similarity_top_k=10,
            )
        )
        ```

    Attributes:
        stores_text: Always True - stores text content with embeddings.
        is_embedding_query: Always True - queries use embeddings.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    # Connection settings
    connection_string: str
    table_name: str = "rag_vectors"
    embedding_dim: int = 2048  # jina-v4 default (Matryoshka: 128/256/512/1024/2048)

    # Performance settings
    hnsw_m: int = 16  # HNSW M parameter (connections per node)
    hnsw_ef_construction: int = 64  # HNSW ef_construction parameter

    # Private attributes
    _pool: Optional[Any] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pool = None
        self._initialized = False

    @property
    def client(self) -> Any:
        """Get the asyncpg connection pool."""
        return self._pool

    async def _ensure_pool(self) -> None:
        """Ensure connection pool is created."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=10,
                )
                logger.info(f"Created asyncpg connection pool for {self.table_name}")
            except ImportError:
                raise ImportError(
                    "asyncpg is required for PGVectorStore. "
                    "Install with: pip install asyncpg"
                )

    async def initialize(self) -> None:
        """Initialize the vector store (create table and indexes).

        This should be called once before using the store.
        """
        if self._initialized:
            return

        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create unified table for text + images
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    node_id TEXT UNIQUE NOT NULL,
                    source_id TEXT NOT NULL,
                    ref_doc_id TEXT,
                    content TEXT,
                    node_type TEXT NOT NULL DEFAULT 'text',
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({self.embedding_dim}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """
            )

            # Create indexes
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source
                ON {self.table_name} (source_id);
            """
            )

            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_node_type
                ON {self.table_name} (node_type);
            """
            )

            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ref_doc
                ON {self.table_name} (ref_doc_id);
            """
            )

            # Create HNSW index for similarity search (cosine distance)
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction});
            """
            )

            logger.info(f"Initialized PGVectorStore table: {self.table_name}")

        self._initialized = True

    # -------------------------------------------------------------------------
    # BasePydanticVectorStore interface implementation
    # -------------------------------------------------------------------------

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes synchronously (wraps async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, **kwargs)
        )

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes with embeddings to the vector store.

        Args:
            nodes: Sequence of BaseNode objects with embeddings.
            **kwargs: Additional arguments:
                - source_id: Required. ID of the DocumentSource.
                - node_type: Node type (text/image/document). Default: text.

        Returns:
            List of inserted node IDs.
        """
        if not nodes:
            return []

        source_id = kwargs.get("source_id")
        if not source_id:
            raise ValueError("source_id is required for PGVectorStore.async_add")

        node_type = kwargs.get("node_type", "text")
        await self._ensure_pool()

        if not self._initialized:
            await self.initialize()

        inserted_ids = []
        async with self._pool.acquire() as conn:
            for node in nodes:
                node_id = node.node_id if hasattr(node, "node_id") else node.id
                ref_doc_id = (
                    node.ref_doc_id
                    if hasattr(node, "ref_doc_id")
                    else kwargs.get("ref_doc_id")
                )

                # Get content
                content = ""
                if hasattr(node, "get_content"):
                    content = node.get_content()
                elif hasattr(node, "text"):
                    content = node.text or ""

                # Get embedding
                embedding = node.embedding if hasattr(node, "embedding") else None
                if embedding is None:
                    logger.warning(f"Node {node_id} has no embedding, skipping")
                    continue

                # Get metadata
                metadata = node.metadata if hasattr(node, "metadata") else {}

                # Convert embedding to DB format
                embedding_db = to_db(embedding, dim=self.embedding_dim)

                try:
                    row = await conn.fetchrow(
                        f"""
                        INSERT INTO {self.table_name}
                        (node_id, source_id, ref_doc_id, content, node_type, metadata, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (node_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            source_id = EXCLUDED.source_id
                        RETURNING id
                        """,
                        node_id,
                        source_id,
                        ref_doc_id,
                        content,
                        node_type,
                        json.dumps(metadata),
                        embedding_db,
                    )
                    inserted_ids.append(str(row["id"]))
                except Exception as e:
                    logger.error(f"Failed to insert node {node_id}: {e}")

        logger.info(f"Added {len(inserted_ids)} nodes to {self.table_name}")
        return inserted_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id synchronously."""
        asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **delete_kwargs)
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id asynchronously."""
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE ref_doc_id = $1",
                ref_doc_id,
            )
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} nodes with ref_doc_id={ref_doc_id}")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query synchronously (wraps async)."""
        return asyncio.get_event_loop().run_until_complete(self.aquery(query, **kwargs))

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store asynchronously.

        Args:
            query: VectorStoreQuery with embedding and filters.
            **kwargs: Additional arguments:
                - source_ids: List of source IDs to filter by.
                - node_type: Filter by node type.

        Returns:
            VectorStoreQueryResult with matching nodes and similarities.
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for vector search")

        await self._ensure_pool()

        if not self._initialized:
            await self.initialize()

        embedding_db = to_db(query.query_embedding, dim=self.embedding_dim)

        # Build WHERE clause
        where_clauses = []
        params: List[Any] = [embedding_db]
        param_idx = 2

        # Source ID filtering
        source_ids = kwargs.get("source_ids")
        if source_ids:
            where_clauses.append(f"source_id = ANY(${param_idx})")
            params.append(source_ids)
            param_idx += 1

        # Node type filtering
        node_type = kwargs.get("node_type")
        if node_type:
            where_clauses.append(f"node_type = ${param_idx}")
            params.append(node_type)
            param_idx += 1

        # Doc ID filtering
        if query.doc_ids:
            where_clauses.append(f"ref_doc_id = ANY(${param_idx})")
            params.append(query.doc_ids)
            param_idx += 1

        # Node ID filtering
        if query.node_ids:
            where_clauses.append(f"node_id = ANY(${param_idx})")
            params.append(query.node_ids)
            param_idx += 1

        # Metadata filters
        if query.filters:
            filter_sql, filter_params = self._build_metadata_filter_sql(
                query.filters, param_idx
            )
            if filter_sql:
                where_clauses.append(filter_sql)
                params.extend(filter_params)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Add top_k param
        params.append(query.similarity_top_k)
        top_k_idx = len(params)

        # Execute query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, node_id, source_id, ref_doc_id, content,
                       node_type, metadata,
                       1 - (embedding <=> $1) as similarity
                FROM {self.table_name}
                WHERE {where_sql}
                ORDER BY embedding <=> $1
                LIMIT ${top_k_idx}
                """,
                *params,
            )

        # Convert to result
        nodes = []
        similarities = []
        ids = []

        for row in rows:
            # Create TextNode from row
            metadata = (
                json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"] or {}
            )

            node = TextNode(
                id=row["node_id"],
                text=row["content"] or "",
                metadata=metadata,
            )
            # Store additional info in metadata
            node.metadata["_source_id"] = row["source_id"]
            node.metadata["_node_type"] = row["node_type"]
            node.metadata["_ref_doc_id"] = row["ref_doc_id"]

            nodes.append(node)
            similarities.append(float(row["similarity"]))
            ids.append(row["node_id"])

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _build_metadata_filter_sql(
        self,
        filters: MetadataFilters,
        start_param_idx: int,
    ) -> tuple[str, List[Any]]:
        """Build SQL WHERE clause from MetadataFilters.

        Args:
            filters: MetadataFilters object.
            start_param_idx: Starting parameter index for placeholders.

        Returns:
            Tuple of (SQL string, list of parameters).
        """
        clauses = []
        params = []
        param_idx = start_param_idx

        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                # Nested filters
                nested_sql, nested_params = self._build_metadata_filter_sql(
                    f, param_idx
                )
                if nested_sql:
                    clauses.append(f"({nested_sql})")
                    params.extend(nested_params)
                    param_idx += len(nested_params)
            elif isinstance(f, MetadataFilter):
                sql, filter_params = self._metadata_filter_to_sql(f, param_idx)
                if sql:
                    clauses.append(sql)
                    params.extend(filter_params)
                    param_idx += len(filter_params)

        if not clauses:
            return "", []

        # Join with condition
        if filters.condition == FilterCondition.OR:
            return " OR ".join(clauses), params
        elif filters.condition == FilterCondition.NOT:
            return f"NOT ({' AND '.join(clauses)})", params
        else:  # AND
            return " AND ".join(clauses), params

    def _metadata_filter_to_sql(
        self,
        f: MetadataFilter,
        param_idx: int,
    ) -> tuple[str, List[Any]]:
        """Convert a single MetadataFilter to SQL.

        Uses JSONB operators for metadata field access.
        """
        key = f.key
        value = f.value
        op = f.operator

        # JSONB path
        json_path = f"metadata->>'{key}'"

        if op == FilterOperator.EQ:
            return f"{json_path} = ${param_idx}", [str(value)]
        elif op == FilterOperator.NE:
            return f"{json_path} != ${param_idx}", [str(value)]
        elif op == FilterOperator.GT:
            return f"({json_path})::numeric > ${param_idx}", [value]
        elif op == FilterOperator.GTE:
            return f"({json_path})::numeric >= ${param_idx}", [value]
        elif op == FilterOperator.LT:
            return f"({json_path})::numeric < ${param_idx}", [value]
        elif op == FilterOperator.LTE:
            return f"({json_path})::numeric <= ${param_idx}", [value]
        elif op == FilterOperator.IN:
            return f"{json_path} = ANY(${param_idx})", [
                [str(v) for v in value] if isinstance(value, list) else [str(value)]
            ]
        elif op == FilterOperator.NIN:
            return f"NOT ({json_path} = ANY(${param_idx}))", [
                [str(v) for v in value] if isinstance(value, list) else [str(value)]
            ]
        elif op == FilterOperator.TEXT_MATCH:
            return f"{json_path} LIKE ${param_idx}", [f"%{value}%"]
        elif op == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return f"{json_path} ILIKE ${param_idx}", [f"%{value}%"]
        elif op == FilterOperator.CONTAINS:
            return f"metadata->'{key}' ? ${param_idx}", [str(value)]
        elif op == FilterOperator.IS_EMPTY:
            return f"(metadata->'{key}' IS NULL OR metadata->>'{key}' = '')", []
        else:
            logger.warning(f"Unsupported filter operator: {op}")
            return "", []

    # -------------------------------------------------------------------------
    # Additional RAG-specific methods
    # -------------------------------------------------------------------------

    async def add_node(
        self,
        node_id: str,
        embedding: List[float],
        content: str,
        node_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ref_doc_id: Optional[str] = None,
    ) -> str:
        """Add a single node directly (convenience method).

        Args:
            node_id: Unique node identifier.
            embedding: Vector embedding.
            content: Text content.
            node_type: Type of node (text/image/document).
            source_id: Parent DocumentSource ID.
            metadata: Additional metadata.
            ref_doc_id: Reference document ID.

        Returns:
            Database UUID of inserted row.
        """
        await self._ensure_pool()

        if not self._initialized:
            await self.initialize()

        embedding_db = to_db(embedding, dim=self.embedding_dim)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self.table_name}
                (node_id, source_id, ref_doc_id, content, node_type, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (node_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
                RETURNING id
                """,
                node_id,
                source_id,
                ref_doc_id,
                content,
                node_type,
                json.dumps(metadata or {}),
                embedding_db,
            )
            return str(row["id"])

    async def search(
        self,
        query_embedding: List[float],
        source_ids: Optional[List[str]] = None,
        top_k: int = 10,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store (convenience method).

        Text query can match images since jina-v4 uses the same embedding space.

        Args:
            query_embedding: Query vector.
            source_ids: Filter to specific sources.
            top_k: Number of results.
            node_type: Filter by node type.

        Returns:
            List of matching documents as dictionaries.
        """
        await self._ensure_pool()

        if not self._initialized:
            await self.initialize()

        embedding_db = to_db(query_embedding, dim=self.embedding_dim)

        # Build WHERE clause
        where_clauses = []
        params: List[Any] = [embedding_db, top_k]

        if source_ids:
            where_clauses.append(f"source_id = ANY($3)")
            params.append(source_ids)

        if node_type:
            where_clauses.append(f"node_type = ${len(params) + 1}")
            params.append(node_type)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, node_id, source_id, ref_doc_id, content,
                       node_type, metadata,
                       1 - (embedding <=> $1) as similarity
                FROM {self.table_name}
                WHERE {where_sql}
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                *params,
            )

        return [dict(r) for r in rows]

    async def delete_by_source(self, source_id: str) -> int:
        """Delete all nodes for a DocumentSource.

        Args:
            source_id: DocumentSource ID.

        Returns:
            Number of deleted rows.
        """
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE source_id = $1",
                source_id,
            )
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} nodes for source_id={source_id}")
            return count

    async def get_source_stats(self, source_id: str) -> Dict[str, Any]:
        """Get statistics for a DocumentSource.

        Args:
            source_id: DocumentSource ID.

        Returns:
            Dictionary with node counts by type.
        """
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT node_type, COUNT(*) as count
                FROM {self.table_name}
                WHERE source_id = $1
                GROUP BY node_type
                """,
                source_id,
            )

        stats = {"total": 0, "by_type": {}}
        for row in rows:
            stats["by_type"][row["node_type"]] = row["count"]
            stats["total"] += row["count"]

        return stats

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info(f"Closed connection pool for {self.table_name}")
