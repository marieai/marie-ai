from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from marie.storage.database.postgres_pool import (
    AsyncPostgresConnection,
    AsyncPostgresConnectionPool,
)


class StaleIndexingRun(ValueError):
    pass


async def bind_upstream_input(
    connection: AsyncPostgresConnection,
    dag_id: str,
    metadata: dict[str, Any],
    planner: str,
) -> None:
    """Bind already registered waiting items in the source DAG's admission transaction."""
    if metadata.get('ref_type') == 'kb_item' or not metadata.get('uri'):
        return
    try:
        tenant_id = UUID(metadata.get('project_id') or metadata.get('tenant_id'))
    except (ValueError, TypeError, AttributeError):
        return
    if not await connection.fetchval(
        "SELECT to_regclass('marie_scheduler.kb_item_state')"
    ):
        return
    await connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(jsonb_build_array('kb_upstream'::text,$1::text,$2::text,$3::text)::text,0))",
        str(tenant_id),
        planner,
        metadata['uri'],
    )
    await connection.execute(
        '''UPDATE marie_scheduler.kb_item_state s
           SET input_ref=(s.input_ref - 'artifact' - 'completion_id' - 'completed_at') || $1::jsonb,
               indexing_dag_id=CASE WHEN s.admitted_at IS NOT NULL OR s.indexed_dag_id IS NOT NULL
                   OR EXISTS(SELECT 1 FROM marie_scheduler.dag d WHERE d.id=s.indexing_dag_id)
                   THEN gen_random_uuid() ELSE s.indexing_dag_id END,
               state='pending',error=NULL,admitted_at=NULL,updated_at=clock_timestamp()
           WHERE s.tenant_id=$2 AND s.uri=$3
             AND s.state <> 'removed' AND s.input_ref->>'kind'='upstream'
             AND s.input_ref->>'planner'=$4
             AND s.input_ref->>'dag_id' IS DISTINCT FROM $5
             AND (NOT s.input_ref ? 'fabric_group_id' OR s.input_ref->>'fabric_group_id' IS NOT DISTINCT FROM $6)
             AND (NOT s.input_ref ? 'fabric_group_ids' OR s.input_ref->'fabric_group_ids' ? $6)''',
        json.dumps(
            {
                'dag_id': dag_id,
                'ref_id': metadata.get('ref_id'),
                'ref_type': metadata.get('ref_type'),
                'fabric_group_id': metadata.get('fabric_group_id'),
            }
        ),
        tenant_id,
        metadata['uri'],
        planner,
        dag_id,
        metadata.get('fabric_group_id'),
    )
    await _sync_document_indexing_dags(connection, dag_id)


async def _sync_document_indexing_dags(
    connection: AsyncPostgresConnection, dag_id: str
) -> None:
    if await connection.fetchval("SELECT to_regclass('marie_studio.kb_document')"):
        await connection.execute(
            '''UPDATE marie_studio.kb_document doc
               SET dag_id=item.indexing_dag_id,status=CASE WHEN item.state='error' THEN 'error' ELSE 'pending' END,
                   stage=CASE WHEN item.input_ref ? 'artifact' THEN 'queued' ELSE 'waiting_upstream' END,
                   error_message=item.error,
                   metadata=jsonb_set(doc.metadata,'{kbItem,indexingDagId}',to_jsonb(item.indexing_dag_id::text))
               FROM marie_scheduler.kb_item_state item
               WHERE item.input_ref->>'dag_id'=$1
                 AND item.source_id=doc.source_id::text
                 AND item.item_id=doc.metadata->'kbItem'->>'itemId' AND item.state <> 'removed'
                 AND doc.dag_id IS DISTINCT FROM item.indexing_dag_id''',
            dag_id,
        )


@dataclass(frozen=True)
class KbItemScope:
    tenant_id: str
    index_id: str
    source_id: str
    item_id: str

    def values(self) -> tuple[UUID, UUID, str, str]:
        return UUID(self.tenant_id), UUID(self.index_id), self.source_id, self.item_id


def item_indexing_run(parameters: dict[str, Any]) -> tuple[KbItemScope, str]:
    """Resolve the complete identity required at an indexing boundary."""
    try:
        parameters = {
            **parameters,
            'kb_indexing_dag_id': parameters.get('kb_indexing_dag_id')
            or parameters.get('kb_attempt_id'),
        }
        fields = ('tenant_id', 'index_id', 'source_id', 'item_id', 'kb_indexing_dag_id')
        if any(
            not isinstance(parameters.get(key), str) or not parameters[key]
            for key in fields
        ):
            raise ValueError('Incomplete identity')
        tenant_id = str(UUID(parameters['tenant_id']))
        index_id = str(UUID(parameters['index_id']))
        indexing_dag_id = str(UUID(parameters['kb_indexing_dag_id']))
        if (
            parameters.get('index_name')
            and str(UUID(parameters['index_name'])) != index_id
        ):
            raise ValueError('Conflicting index names')
        return (
            KbItemScope(
                tenant_id, index_id, parameters['source_id'], parameters['item_id']
            ),
            indexing_dag_id,
        )
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(
            'Registered KB item scope and current indexing DAG are required'
        ) from error


def item_run_parameters(item: dict[str, Any]) -> dict[str, Any]:
    """Carry an admitted item's parameters and a DAG-owned artifact reference."""
    identity = [
        str(item[key]) for key in ('tenant_id', 'index_id', 'source_id', 'item_id')
    ]
    artifact_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    parameters = {
        **dict(zip(('tenant_id', 'index_id', 'source_id', 'item_id'), identity)),
        'project_id': identity[0],
        'index_name': identity[1],
        'kb_indexing_dag_id': str(item['indexing_dag_id']),
        'owner_type': item['owner_type'],
        'owner_id': item['owner_id'],
        'uri': item['uri'],
        'ref_id': f"{artifact_id}-{item['indexing_dag_id']}",
        'ref_type': 'kb_item',
        'run_params': item['run_params'],
    }
    if item.get('input_ref') is not None:
        parameters['input_ref'] = item['input_ref']
        workflow = item['input_ref'].get('processing_workflow')
        if workflow:
            parameters['planner'] = workflow
    if item['run_params'].get('fabric_group_id'):
        parameters['fabric_group_id'] = item['run_params']['fabric_group_id']
    return parameters


def scheduler_execution(parameters: dict[str, Any]) -> dict[str, str]:
    execution = {
        key: parameters.get(key) for key in ('job_id', 'run_attempt_id', 'run_owner')
    }
    try:
        UUID(execution['job_id'])
        UUID(execution['run_attempt_id'])
        if not execution['run_owner']:
            raise ValueError('Missing run owner')
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError('KB item requires a scheduler execution and lease') from error
    return execution


def resolve_indexing_parameters(
    item: dict[str, Any] | None, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a current registered run using its saved processing settings."""
    _, dag_id = item_indexing_run(parameters)
    if (
        item is None
        or item['state'] == 'removed'
        or str(item['indexing_dag_id']) != dag_id
    ):
        raise StaleIndexingRun('KB indexing DAG changed or was removed')
    saved = item_run_parameters(item)
    for key in (
        'run_params',
        'uri',
        'input_ref',
        'ref_id',
        'ref_type',
        'index_name',
    ):
        if key in parameters and parameters[key] != saved.get(key):
            raise ValueError(f'{key} conflicts with the saved KB indexing settings')
    planner = saved.get('planner', 'kb_indexing')
    if parameters.get('planner', planner) != planner:
        raise ValueError('Planner conflicts with the saved KB indexing workflow')
    return {**saved, 'planner': planner}


class KbItemStore:
    """Keep current item and indexing DAG beside the adapter's indexed rows."""

    def __init__(self, pool: AsyncPostgresConnectionPool) -> None:
        self.pool = pool

    async def resolve_indexing_parameters(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve a current registered run using its saved processing settings."""
        scope, _ = item_indexing_run(parameters)
        item = await self.get(scope)
        return resolve_indexing_parameters(item, parameters)

    async def register(
        self,
        scope: KbItemScope,
        *,
        indexing_dag_id: str,
        owner_type: str,
        owner_id: str,
        uri: str,
        run_params: dict[str, Any],
        backend: str = 'postgresql',
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Admit an authorized item, idempotently for the same input and indexing DAG."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    '''INSERT INTO marie_scheduler.kb_item_state
                       (tenant_id,index_id,source_id,item_id,indexing_dag_id,owner_type,owner_id,uri,run_params,backend,store_name)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11)
                       ON CONFLICT DO NOTHING''',
                    *scope.values(),
                    UUID(indexing_dag_id),
                    owner_type,
                    owner_id,
                    uri,
                    json.dumps(run_params),
                    backend,
                    store_name,
                )
                row = await self._lock(conn, scope)
                if (
                    row['state'] == 'removed'
                    or str(row['indexing_dag_id']) != indexing_dag_id
                    or (
                        row['owner_type'],
                        row['owner_id'],
                        row['uri'],
                        row['run_params'],
                    )
                    != (owner_type, owner_id, uri, run_params)
                ):
                    raise StaleIndexingRun(
                        'KB item registration changed or was removed'
                    )
                self._check_backend(row, backend, store_name)
                return row

    async def get(self, scope: KbItemScope) -> dict[str, Any] | None:
        return await self.pool.fetchrow(
            '''SELECT * FROM marie_scheduler.kb_item_state
               WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4''',
            *scope.values(),
        )

    async def reindex(
        self,
        scope: KbItemScope,
        *,
        expected_indexing_dag_id: str,
        indexing_dag_id: str,
        run_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a fresh indexing DAG while retaining the committed result."""
        if UUID(indexing_dag_id) == UUID(expected_indexing_dag_id):
            raise ValueError('Re-indexing requires a fresh DAG ID')
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    '''UPDATE marie_scheduler.kb_item_state
                       SET indexing_dag_id=$6, state='pending', error=NULL, admitted_at=NULL,
                           run_params=COALESCE($7::jsonb,run_params), updated_at=clock_timestamp()
                       WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
                         AND indexing_dag_id=$5 AND state <> 'removed'
                       RETURNING *''',
                    *scope.values(),
                    UUID(expected_indexing_dag_id),
                    UUID(indexing_dag_id),
                    json.dumps(run_params) if run_params is not None else None,
                )
                if row is None:
                    raise StaleIndexingRun(
                        'KB item indexing DAG changed or was removed'
                    )
                return row

    async def pending(
        self, owner_type: str | tuple[str, ...], limit: int
    ) -> list[dict[str, Any]]:
        return await self.pool.fetch(
            '''SELECT * FROM marie_scheduler.kb_item_state
               WHERE state='pending' AND admitted_at IS NULL AND owner_type=ANY($1)
                 AND backend IN ('pgvector','postgresql_text')
                 AND (input_ref->>'kind' IS DISTINCT FROM 'upstream' OR input_ref ? 'artifact')
               ORDER BY updated_at,tenant_id,index_id,source_id,item_id LIMIT $2''',
            [owner_type] if isinstance(owner_type, str) else list(owner_type),
            limit,
        )

    async def complete_upstream(
        self, completion: dict[str, Any], *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Attach saved output to current waiting destinations without creating membership."""
        from marie.kb.content import document_content_reference

        dag_id = UUID(completion['dag_id'])
        if not 1 <= limit <= 1000:
            raise ValueError('KB handoff limit must be between 1 and 1000')
        outputs = [completion.get('output')] + [
            node.get('output')
            for node in completion.get('outputs', [])
            if isinstance(node, dict)
        ]
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                source = await connection.fetchrow(
                    'SELECT state,completed_on FROM marie_scheduler.dag WHERE id=$1 FOR SHARE',
                    dag_id,
                )
                if (
                    source is None
                    or source['state'] != completion['state']
                    or source['completed_on'] != completion['completed_at']
                ):
                    return []
                items = await connection.fetch(
                    '''SELECT * FROM marie_scheduler.kb_item_state
                       WHERE input_ref->>'kind'='upstream' AND input_ref->>'dag_id'=$1
                         AND state <> 'removed' AND (
                           input_ref->>'completion_id' IS DISTINCT FROM $2 OR
                           (state='pending' AND admitted_at IS NULL))
                       ORDER BY tenant_id,index_id,source_id,item_id LIMIT $3 FOR UPDATE''',
                    str(dag_id),
                    str(completion['completion_id']),
                    limit,
                )
                ready = []
                for item in items:
                    reference = dict(item['input_ref'])
                    new_occurrence = reference.get('completion_id') != str(
                        completion['completion_id']
                    )
                    indexing_dag_id = item['indexing_dag_id']
                    accepted = await connection.fetchval(
                        'SELECT EXISTS(SELECT 1 FROM marie_scheduler.dag WHERE id=$1)',
                        indexing_dag_id,
                    )
                    if new_occurrence and (
                        item['admitted_at'] is not None
                        or item['indexed_dag_id'] is not None
                        or accepted
                    ):
                        indexing_dag_id = uuid4()
                    reference['completion_id'] = str(completion['completion_id'])
                    message = None
                    if source['state'] != 'completed':
                        message = f"Upstream workflow {source['state']}"
                        reference.pop('artifact', None)
                    else:
                        candidates = [
                            output['input_ref']
                            for output in outputs
                            if isinstance(output, dict)
                            and isinstance(output.get('input_ref'), dict)
                        ]
                        candidates = [
                            candidate
                            for candidate in candidates
                            if candidate.get('kind') != 'extraction'
                            or (candidate.get('ref_id'), candidate.get('ref_type'))
                            == (reference.get('ref_id'), reference.get('ref_type'))
                        ]
                        unique = {
                            json.dumps(candidate, sort_keys=True): candidate
                            for candidate in candidates
                        }
                        try:
                            if len(unique) != 1:
                                raise ValueError(
                                    'Upstream workflow must publish one unambiguous saved content reference'
                                )
                            artifact = next(iter(unique.values()))
                            if (
                                reference.get('processing_workflow')
                                == 'kb_index_content'
                            ):
                                artifact = document_content_reference(artifact)
                            reference['artifact'] = artifact
                            reference['completed_at'] = source[
                                'completed_on'
                            ].isoformat()
                        except ValueError as error:
                            message = str(error)
                            reference.pop('artifact', None)
                    scope = KbItemScope(
                        *(
                            str(item[key])
                            for key in ('tenant_id', 'index_id', 'source_id', 'item_id')
                        )
                    )
                    updated = await connection.fetchrow(
                        '''UPDATE marie_scheduler.kb_item_state SET input_ref=$5::jsonb,
                           state=$6,error=$7,indexing_dag_id=$8,admitted_at=NULL,updated_at=clock_timestamp()
                           WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4 RETURNING *''',
                        *scope.values(),
                        json.dumps(reference),
                        'error' if message else 'pending',
                        message,
                        indexing_dag_id,
                    )
                    if message is None:
                        ready.append(updated)
                await _sync_document_indexing_dags(connection, str(dag_id))
                return ready

    async def mark_admitted(self, scope: KbItemScope, indexing_dag_id: str) -> None:
        await self.pool.execute(
            '''UPDATE marie_scheduler.kb_item_state SET admitted_at=COALESCE(admitted_at,clock_timestamp())
               WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
                 AND indexing_dag_id=$5 AND state = 'pending' ''',
            *scope.values(),
            UUID(indexing_dag_id),
        )

    async def fail(
        self, scope: KbItemScope, indexing_dag_id: str, message: str
    ) -> None:
        await self.pool.execute(
            '''UPDATE marie_scheduler.kb_item_state SET state='error',error=$6
               WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
                 AND indexing_dag_id=$5 AND state = 'pending' ''',
            *scope.values(),
            UUID(indexing_dag_id),
            message,
        )

    @asynccontextmanager
    async def write(
        self,
        scope: KbItemScope,
        indexing_dag_id: str,
        *,
        record_count: int,
        content_digest: str | None = None,
        backend: str | None = None,
        store_name: str | None = None,
        job_id: str | None = None,
        run_attempt_id: str | None = None,
        run_owner: str | None = None,
    ) -> AsyncIterator[AsyncPostgresConnection]:
        """Replace adapter-owned rows and completion in one checked transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await self._lock(conn, scope)
                if row['state'] != 'pending' or row['indexing_dag_id'] != UUID(
                    indexing_dag_id
                ):
                    raise StaleIndexingRun('KB indexing DAG is no longer current')
                self._check_backend(row, backend, store_name)
                reference = row.get('input_ref') or {}
                if reference.get('kind') == 'upstream':
                    upstream = await conn.fetchrow(
                        'SELECT state,completed_on FROM marie_scheduler.dag WHERE id=$1 FOR SHARE',
                        UUID(reference['dag_id']),
                    )
                    if (
                        upstream is None
                        or upstream['state'] != 'completed'
                        or upstream['completed_on']
                        != datetime.fromisoformat(reference['completed_at'])
                    ):
                        raise StaleIndexingRun(
                            'Upstream workflow input was rerun or is no longer complete'
                        )
                if job_id is not None:
                    job = await conn.fetchrow(
                        '''SELECT data,dag_id,state,run_attempt_id,run_owner,
                                  run_lease_expires_at > clock_timestamp() AS lease_valid
                           FROM marie_scheduler.job WHERE id=$1 FOR UPDATE''',
                        UUID(job_id),
                    )
                    expected = item_run_parameters(row)
                    metadata = (job['data'] or {}).get('metadata', {}) if job else {}
                    metadata = {
                        **metadata,
                        'kb_indexing_dag_id': metadata.get('kb_indexing_dag_id')
                        or metadata.get('kb_attempt_id'),
                    }
                    if (
                        job is None
                        or str(job['state']) != 'active'
                        or not job['lease_valid']
                        or str(job['dag_id']) != indexing_dag_id
                        or not run_attempt_id
                        or str(job['run_attempt_id']) != run_attempt_id
                        or not run_owner
                        or job['run_owner'] != run_owner
                        or metadata.get('input_ref') != expected.get('input_ref')
                        or any(
                            metadata.get(key) != expected[key]
                            for key in (
                                'tenant_id',
                                'index_id',
                                'source_id',
                                'item_id',
                                'kb_indexing_dag_id',
                                'owner_type',
                                'owner_id',
                                'uri',
                                'ref_id',
                            )
                        )
                    ):
                        raise StaleIndexingRun(
                            'KB scheduler execution is no longer current'
                        )
                yield conn
                if job_id is not None and not await conn.fetchval(
                    'SELECT run_lease_expires_at > clock_timestamp() FROM marie_scheduler.job WHERE id=$1',
                    UUID(job_id),
                ):
                    raise StaleIndexingRun(
                        'KB scheduler lease expired during replacement'
                    )
                await conn.execute(
                    '''UPDATE marie_scheduler.kb_item_state
                       SET state='ready', indexed_dag_id=indexing_dag_id,
                           record_count=$5, content_digest=$6, error=NULL,
                           updated_at=clock_timestamp()
                       WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4''',
                    *scope.values(),
                    record_count,
                    content_digest,
                )

    async def remove(
        self,
        scope: KbItemScope,
        *,
        backend: str | None = None,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Durably withdraw delivery before independently retryable cleanup."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                self._check_backend(await self._lock(conn, scope), backend, store_name)
                row = await conn.fetchrow(
                    '''UPDATE marie_scheduler.kb_item_state
                       SET state='removed',
                           cleanup_state=CASE WHEN state='removed' THEN cleanup_state ELSE 'pending' END,
                           updated_at=clock_timestamp()
                       WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
                       RETURNING *''',
                    *scope.values(),
                )
                if row is None:
                    raise StaleIndexingRun('KB item is not registered')
                return row

    @staticmethod
    def _check_backend(
        row: dict[str, Any], backend: str | None, store_name: str | None
    ) -> None:
        if backend is not None and (row['backend'], row['store_name']) != (
            backend,
            store_name,
        ):
            raise PermissionError('KB item backend does not match this store')

    async def _lock(
        self, conn: AsyncPostgresConnection, scope: KbItemScope
    ) -> dict[str, Any]:
        row = await conn.fetchrow(
            '''SELECT * FROM marie_scheduler.kb_item_state
               WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4
               FOR UPDATE''',
            *scope.values(),
        )
        if row is None:
            raise StaleIndexingRun('KB item is not registered')
        return row
