"""PostgreSQL full-text indexing of one item's captured text."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from marie.executor.kb.chunking import chunk_text
from marie.kb.content import document_content_reference
from marie.kb.item_state import KbItemScope, KbItemStore
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

TEXT_BACKEND = 'postgresql_text'
TEXT_STORE = 'kb_text_section'


class TextIndexStore:
    def __init__(self, pool: AsyncPostgresConnectionPool) -> None:
        self.pool = pool

    async def replace(
        self,
        scope: KbItemScope,
        indexing_dag_id: str,
        text: str,
        reference: dict[str, Any],
        *,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        job_id: str | None = None,
        run_attempt_id: str | None = None,
        run_owner: str | None = None,
    ) -> int:
        reference = document_content_reference(reference)
        if not reference.get('sha256'):
            raise ValueError('Text indexing requires a captured content digest')
        if len(text.encode('utf-8')) > 32 * 1024 * 1024:
            raise ValueError('Text exceeds the supported artifact size')
        if not 100 <= chunk_size <= 8000 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError('Invalid text section size or overlap')
        step = chunk_size - chunk_overlap
        section_count = (
            0 if not text else 1 + (max(0, len(text) - chunk_size) + step - 1) // step
        )
        if section_count > 10_000 or section_count * chunk_size > 16 * 1024 * 1024:
            raise ValueError(
                'Text requires too many sections; reduce overlap or split the saved artifact'
            )
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        records = [
            (
                *scope.values(),
                hashlib.sha256(
                    json.dumps(
                        [*map(str, scope.values()), chunk.start, chunk.end]
                    ).encode()
                ).hexdigest(),
                UUID(indexing_dag_id),
                chunk.text,
                reference['sha256'],
                json.dumps(reference),
                chunk.start,
                chunk.end,
            )
            for chunk in chunks
        ]
        async with KbItemStore(self.pool).write(
            scope,
            indexing_dag_id,
            record_count=len(records),
            content_digest=reference['sha256'],
            backend=TEXT_BACKEND,
            store_name=TEXT_STORE,
            job_id=job_id,
            run_attempt_id=run_attempt_id,
            run_owner=run_owner,
        ) as connection:
            await connection.execute(
                '''DELETE FROM marie_scheduler.kb_text_section
                WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4''',
                *scope.values(),
            )
            if records:
                await connection.executemany(
                    '''INSERT INTO marie_scheduler.kb_text_section
                    (tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,content,source_digest,input_ref,start_offset,end_offset)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11)''',
                    records,
                )
        return len(records)

    async def search(
        self, tenant_id: str, index_id: str, query: str, *, top_k: int = 10
    ) -> list[dict[str, Any]]:
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 50:
            raise ValueError('A query and supported result limit are required')
        rows = await self.pool.fetch(
            '''SELECT section_id,item_id,source_id,indexed_dag_id,content,
                source_digest,start_offset,end_offset,item_state,refresh_error,
                indexing_dag_id <> indexed_dag_id AS refreshing,
                ts_rank_cd(search_vector,plainto_tsquery('english',$3)) AS score
            FROM marie_scheduler.kb_text_section_current
            WHERE tenant_id=$1 AND index_id=$2 AND search_vector @@ plainto_tsquery('english',$3)
            ORDER BY score DESC,item_id,section_id LIMIT $4''',
            UUID(tenant_id),
            UUID(index_id),
            query,
            top_k,
        )
        return [{**row, 'indexed_dag_id': str(row['indexed_dag_id'])} for row in rows]

    async def passage(
        self,
        scope: KbItemScope,
        section_id: str,
        indexed_dag_id: str,
        digest: str,
    ) -> dict[str, Any] | None:
        return await self.pool.fetchrow(
            '''SELECT * FROM marie_scheduler.kb_text_section_current
            WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
              AND section_id=$5 AND indexed_dag_id=$6 AND source_digest=$7''',
            *scope.values(),
            section_id,
            UUID(indexed_dag_id),
            digest,
        )

    async def delete_item(self, scope: KbItemScope) -> int:
        removed = await KbItemStore(self.pool).remove(
            scope, backend=TEXT_BACKEND, store_name=TEXT_STORE
        )
        return await self._cleanup_removed(scope.tenant_id, scope.index_id, [removed])

    async def delete_index(self, tenant_id: str, index_id: str) -> int:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                wrong = await connection.fetchval(
                    '''SELECT 1 FROM marie_scheduler.kb_item_state
                    WHERE tenant_id=$1 AND index_id=$2 AND (state <> 'removed' OR cleanup_state <> 'complete')
                      AND (backend <> $3 OR store_name IS DISTINCT FROM $4) LIMIT 1''',
                    UUID(tenant_id),
                    UUID(index_id),
                    TEXT_BACKEND,
                    TEXT_STORE,
                )
                if wrong:
                    raise PermissionError(
                        'KB index contains output owned by another backend'
                    )
                rows = await connection.fetch(
                    '''UPDATE marie_scheduler.kb_item_state
                    SET state='removed',cleanup_state=CASE WHEN state='removed' THEN cleanup_state ELSE 'pending' END,
                        updated_at=clock_timestamp()
                    WHERE tenant_id=$1 AND index_id=$2 AND backend=$3 AND store_name=$4
                      AND (state <> 'removed' OR cleanup_state <> 'complete') RETURNING source_id,item_id,indexing_dag_id''',
                    UUID(tenant_id),
                    UUID(index_id),
                    TEXT_BACKEND,
                    TEXT_STORE,
                )
        count = 0
        for offset in range(0, len(rows), 500):
            count += await self._cleanup_removed(
                tenant_id, index_id, rows[offset : offset + 500]
            )
        return count

    async def _cleanup_removed(
        self, tenant_id: str, index_id: str, items: list[dict[str, Any]]
    ) -> int:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                return await connection.fetchval(
                    '''WITH withdrawn AS MATERIALIZED (
                    SELECT k.* FROM marie_scheduler.kb_item_state k
                    JOIN unnest($3::text[],$4::text[],$5::uuid[]) captured(source_id,item_id,indexing_dag_id)
                      USING (source_id,item_id,indexing_dag_id)
                    WHERE k.tenant_id=$1 AND k.index_id=$2 AND k.state='removed'
                      AND k.backend=$6 AND k.store_name=$7 FOR UPDATE OF k
                ), deleted AS (
                    DELETE FROM marie_scheduler.kb_text_section section USING withdrawn k
                    WHERE (section.tenant_id,section.index_id,section.source_id,section.item_id)=
                          (k.tenant_id,k.index_id,k.source_id,k.item_id) RETURNING 1
                ), completed AS (
                    UPDATE marie_scheduler.kb_item_state k SET cleanup_state='complete',error=NULL,updated_at=clock_timestamp()
                    FROM withdrawn w WHERE (k.tenant_id,k.index_id,k.source_id,k.item_id)=
                                          (w.tenant_id,w.index_id,w.source_id,w.item_id) RETURNING 1
                ) SELECT count(*) FROM deleted''',
                    UUID(tenant_id),
                    UUID(index_id),
                    [row['source_id'] for row in items],
                    [row['item_id'] for row in items],
                    [row['indexing_dag_id'] for row in items],
                    TEXT_BACKEND,
                    TEXT_STORE,
                )
