"""Authoritative KB catalog and transactional lifecycle operations."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, AsyncIterator
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from psycopg.rows import dict_row

from marie.kb import catalog_models as models
from marie.storage.database.postgres_pool import (
    AsyncPostgresConnection,
    AsyncPostgresConnectionPool,
)


class CatalogError(ValueError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


def json_safe(value: Any) -> Any:
    if isinstance(value, (UUID, datetime, date)):
        return str(value) if isinstance(value, UUID) else value.isoformat()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def camel_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        re.sub(r'_([a-z])', lambda match: match[1].upper(), key): json_safe(value)
        for key, value in row.items()
    }
    for key in ('size', 'fileSize'):
        if key in result:
            result[key] = str(result[key])
    return result


def merge_metadata(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(current) if isinstance(current, dict) else {}
    for key, value in patch.items():
        result[key] = (
            merge_metadata(result.get(key) or {}, value)
            if isinstance(value, dict)
            else value
        )
    return result


def backend(index: dict[str, Any]) -> str:
    value = (index['metadata'] or {}).get('retrievalBackend', 'pgvector')
    if value not in ('pgvector', 'postgresql_text'):
        raise CatalogError('Unsupported knowledge base backend', 400)
    return value


def needs_reindex(document: dict[str, Any]) -> bool:
    if document['metadata'].get('removalPending'):
        return False
    item = document.get('itemState')
    if not item:
        return True
    if item['state'] == 'removed':
        return False
    if item['state'] == 'error':
        return True
    if item['indexed_dag_id'] == item['indexing_dag_id']:
        return False
    reference = item.get('input_ref') or {}
    if reference.get('kind') == 'upstream' and not reference.get('artifact'):
        return False
    rows = item['rows']
    if item['admitted_at'] and not rows:
        return True
    stages = [
        row for row in rows if re.search('extract|embed|index', row['name'] or '', re.I)
    ]
    if any(row['state'] in ('failed', 'cancelled', 'expired') for row in stages):
        return True
    if any(row['state'] in ('active', 'retry') for row in stages):
        return False
    return bool(stages) and all(
        row['state'] in ('completed', 'skipped') for row in stages
    )


def run_params(
    index: dict[str, Any],
    previous: dict[str, Any],
    supplied: models.ProcessingParams | None = None,
) -> dict[str, Any]:
    canonical = {
        'chunk_size': index['chunk_size'],
        'chunk_overlap': index['chunk_overlap'],
        'segmentation_mode': index['segmentation_mode'],
        'multimodal': (
            False
            if backend(index) == 'postgresql_text'
            else index['multimodal_enabled']
        ),
        'embedding_model': index['embedding_model'],
        'embedding_dim': index['embedding_dim'],
    }
    if backend(index) == 'postgresql_text':
        canonical['retrieval_backend'] = 'postgresql_text'
    patch = supplied.model_dump(exclude_none=True) if supplied else {}
    for key, value in {**canonical, 'retrieval_backend': backend(index)}.items():
        if key in patch and patch[key] != value:
            raise CatalogError(
                f'{key} must match the saved knowledge base settings', 400
            )
    result = merge_metadata(
        {'parse_mode': index['segmentation_mode'], **previous}, patch
    )
    result.pop('retrieval_backend', None)
    return {**result, **canonical}


class KbCatalog:
    def __init__(
        self,
        pool: AsyncPostgresConnectionPool,
        *,
        schema: str = 'marie_scheduler',
        platform_schema: str = 'marie_studio',
    ) -> None:
        if not all(
            re.fullmatch(r'[a-z_][a-z0-9_]*', name)
            for name in (schema, platform_schema)
        ):
            raise ValueError('Invalid catalog schema')
        self.pool, self.schema = pool, schema
        self.platform_schema = platform_schema

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncPostgresConnection]:
        async with self.pool.acquire() as connection:
            previous = connection._connection.row_factory
            connection._connection.row_factory = dict_row
            try:
                async with connection.transaction():
                    yield connection
            finally:
                connection._connection.row_factory = previous

    async def call(self, operation: str, body: dict[str, Any]) -> Any:
        if operation not in models.COMMANDS:
            raise CatalogError('Unknown KB catalog operation', 404)
        command = models.COMMANDS[operation].model_validate(body)
        async with self._connection() as connection:
            documents = await self._prelock(connection, operation, command)
            if operation == 'reindex_failed':
                return await self._reindex_failed(
                    connection, command, document_ids=documents
                )
            return await getattr(self, f'_{operation}')(connection, command)

    async def _item_rows(
        self,
        connection: AsyncPostgresConnection,
        tenant_id: UUID,
        index_ids: list[UUID],
        *,
        document_ids: list[UUID] | None = None,
        owner_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        lock: bool = False,
    ) -> list[dict[str, Any]]:
        if owner_ids is not None:
            condition = "item.owner_type='submission_document' AND item.owner_id=ANY($2::text[])"
            arguments = (tenant_id, owner_ids)
        elif document_ids is not None:
            condition = f'''EXISTS(SELECT 1 FROM {self.schema}.kb_document d WHERE d.id=ANY($2::uuid[])
                AND d.source_id::text=item.source_id AND (NOT(d.metadata ? 'kbItem') OR d.metadata->'kbItem'->>'itemId'=item.item_id
                    OR item.owner_type='kb_document' AND item.owner_id=d.id::text))'''
            arguments = (tenant_id, document_ids)
        else:
            condition = 'item.index_id=ANY($2::uuid[]) AND ($3::text[] IS NULL OR item.source_id=ANY($3))'
            arguments = (tenant_id, index_ids, source_ids)
        return await connection.fetch(
            f'''SELECT item.* FROM {self.schema}.kb_item_state item WHERE item.tenant_id=$1 AND {condition}
                ORDER BY item.tenant_id,item.index_id,item.source_id,item.item_id'''
            + (' FOR UPDATE OF item' if lock else ''),
            *arguments,
        )

    async def _prelock(
        self, connection: AsyncPostgresConnection, operation: str, command: Any
    ) -> list[UUID] | None:
        document_operations = {
            'reindex_document',
            'withdraw_document',
            'finalize_document_removal',
        }
        index_operations = {
            'admit_documents',
            'reindex_failed',
            'update_index_configuration',
            'withdraw_source',
            'finalize_source_removal',
            'withdraw_index',
            'finalize_index_removal',
        }
        submission_operations = {
            'set_submission_destinations',
            'register_submission_documents',
            'admit_submission_documents',
            'withdraw_submission',
        }
        if (
            operation
            not in document_operations | index_operations | submission_operations
        ):
            return None
        parents: set[UUID] = set()
        advisories: set[tuple[str, str]] = set()
        owner_ids = None
        source_ids = None
        document_ids = None
        submission = None
        snapshot: list[dict[str, Any]] = []
        legacy_submission_sources: dict[UUID, str] = {}
        if operation in submission_operations:
            submission = await self._submission(connection, command)
            parents.add(submission['id'])
            owners = await connection.fetch(
                f'SELECT id,storage_key FROM {self.schema}.submission_documents WHERE submission_id=$1',
                submission['id'],
            )
            owner_ids = [str(row['id']) for row in owners]
            indexes = set(await self._submission_destinations(connection, submission))
            if operation == 'set_submission_destinations':
                indexes.update(command.kbIndexIds)
            owned_items = await self._item_rows(
                connection, command.tenant_id, [], owner_ids=owner_ids
            )
            indexes.update(item['index_id'] for item in owned_items)
            index_ids = [
                row['id']
                for row in await connection.fetch(
                    f'SELECT id FROM {self.schema}.kb_index WHERE id=ANY($1::uuid[]) AND tenant_id=$2 AND organization_id=$3',
                    list(indexes),
                    command.tenant_id,
                    command.organization_id,
                )
            ]
            if operation == 'set_submission_destinations' and not set(
                command.kbIndexIds
            ).issubset(index_ids):
                raise CatalogError('Knowledge base destination not found', 404)
            register = (
                operation
                in ('register_submission_documents', 'admit_submission_documents')
                or operation == 'set_submission_destinations'
                and command.registerExistingDocuments
            )
            enabled = (
                command.enableSemanticSearch
                if operation == 'set_submission_destinations'
                else submission['enable_semantic_search']
            )
            if register and enabled:
                effective = dict(submission)
                if (
                    operation == 'set_submission_destinations'
                    and command.data
                    and 'queryPlanTemplateId' in command.data.model_fields_set
                ):
                    effective['query_plan_template_id'] = (
                        command.data.queryPlanTemplateId
                    )
                planner, _ = await self._upstream_settings(
                    connection,
                    models.RegisterSubmission(
                        organization_id=command.organization_id,
                        actor_id=command.actor_id,
                        tenant_id=command.tenant_id,
                        submissionId=submission['id'],
                        bucket=command.bucket,
                        upstreamPlanner=command.upstreamPlanner,
                        upstreamFabrics=command.upstreamFabrics,
                    ),
                    effective,
                )
                if planner:
                    keys = {row['storage_key'] for row in owners}
                    if operation == 'admit_submission_documents':
                        keys.update(file.storageKey for file in command.files)
                    advisories.update(
                        (planner, f's3://{command.bucket}/{key}') for key in keys
                    )
        elif operation in document_operations:
            index = await self._document_index(connection, command, removing=True)
            index_ids = [index['id']]
            snapshot = [await self._document(connection, index, command.documentId)]
            document_ids = [command.documentId]
        else:
            index = await self._index(
                connection, command, command.indexId, removing=True
            )
            index_ids = [index['id']]
            if operation in ('withdraw_source', 'finalize_source_removal'):
                source_ids = [str(command.sourceId)]
            if operation == 'admit_documents':
                parents.update(
                    file.origin.submissionId
                    for file in command.files
                    if file.origin.kind == 'submission'
                )
                advisories.update(
                    (file.upstreamPlanner, f's3://{command.bucket}/{file.s3Key}')
                    for file in command.files
                    if file.upstreamPlanner
                )
                snapshot = await connection.fetch(
                    f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                        WHERE s.index_id=$1 AND d.source_id=ANY($2::uuid[]) AND d.s3_key=ANY($3::text[])''',
                    index['id'],
                    [file.sourceId for file in command.files],
                    [file.s3Key for file in command.files],
                )
                document_ids = [row['id'] for row in snapshot]
            else:
                snapshot = await connection.fetch(
                    f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                        WHERE s.index_id=$1 AND ($2::text[] IS NULL OR s.id::text=ANY($2)) ORDER BY d.id''',
                    index['id'],
                    source_ids,
                )
                if operation == 'reindex_failed':
                    document_ids = [row['id'] for row in snapshot]
        item_snapshot = await self._item_rows(
            connection,
            command.tenant_id,
            index_ids,
            document_ids=document_ids,
            owner_ids=owner_ids,
            source_ids=source_ids,
        )
        if owner_ids is None:
            owner_ids_for_parents = [
                item['owner_id']
                for item in item_snapshot
                if item['owner_type'] == 'submission_document'
            ]
            if owner_ids_for_parents:
                parents.update(
                    row['submission_id']
                    for row in await connection.fetch(
                        f'SELECT DISTINCT submission_id FROM {self.schema}.submission_documents WHERE id::text=ANY($1::text[])',
                        owner_ids_for_parents,
                    )
                )
            for document in snapshot:
                origin = document['metadata'].get('origin') or {}
                if origin.get('kind') == 'submission':
                    parents.add(UUID(origin['submissionId']))
                elif not origin:
                    source = await self._source(
                        connection, index, document['source_id'], removing=True
                    )
                    if source['source_type'] == 'submission':
                        parents.add(UUID(source['metadata']['submissionId']))
                        legacy_submission_sources[source['id']] = source['metadata'][
                            'submissionId'
                        ]
        if operation == 'finalize_index_removal' and await self._exists(
            connection, 'submissions'
        ):
            parents.update(
                row['id']
                for row in await connection.fetch(
                    f'''SELECT s.id FROM {self.schema}.submissions s WHERE s.tenant_id=$1 AND
                    (s.kb_index_id=$2 OR EXISTS(SELECT 1 FROM {self.schema}.submission_kb_indexes l
                        WHERE l.submission_id=s.id AND l.kb_index_id=$2))''',
                    command.tenant_id,
                    command.indexId,
                )
            )
        if parents:
            await connection.fetch(
                f'''SELECT id FROM {self.schema}.submissions WHERE tenant_id=$1 AND id=ANY($2::uuid[])
                    ORDER BY id FOR UPDATE''',
                command.tenant_id,
                sorted(parents, key=str),
            )
        for item in item_snapshot:
            reference = item['input_ref'] or {}
            if reference.get('kind') == 'upstream':
                advisories.add((reference['planner'], item['uri']))
        for planner, uri in sorted(advisories):
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(jsonb_build_array('kb_upstream'::text,$1::text,$2::text,$3::text)::text,0))",
                str(command.tenant_id),
                planner,
                uri,
            )
        for index_id in sorted(index_ids, key=str):
            await self._index(connection, command, index_id, lock=True, removing=True)
        if operation == 'finalize_index_removal' and await self._exists(
            connection, 'submissions'
        ):
            current_parents = {
                row['id']
                for row in await connection.fetch(
                    f'''SELECT s.id FROM {self.schema}.submissions s WHERE s.tenant_id=$1 AND
                    (s.kb_index_id=$2 OR EXISTS(SELECT 1 FROM {self.schema}.submission_kb_indexes l
                        WHERE l.submission_id=s.id AND l.kb_index_id=$2))''',
                    command.tenant_id,
                    command.indexId,
                )
            }
            if not current_parents.issubset(parents):
                raise CatalogError(
                    'Submission membership changed; retry knowledge base removal'
                )
        await connection.fetch(
            f'SELECT id FROM {self.schema}.kb_source WHERE index_id=ANY($1::uuid[]) ORDER BY index_id,id FOR UPDATE',
            index_ids,
        )
        for source_id, submission_id in legacy_submission_sources.items():
            current = await connection.fetchval(
                f"SELECT metadata->>'submissionId' FROM {self.schema}.kb_source WHERE id=$1",
                source_id,
            )
            if current != submission_id:
                raise CatalogError('Submission source ownership changed')
        if operation == 'admit_documents':
            document_ids = [
                row['id']
                for row in await connection.fetch(
                    f'''SELECT d.id FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                    WHERE s.index_id=$1 AND d.source_id=ANY($2::uuid[]) AND d.s3_key=ANY($3::text[])''',
                    command.indexId,
                    [file.sourceId for file in command.files],
                    [file.s3Key for file in command.files],
                )
            ]
        await self._item_rows(
            connection,
            command.tenant_id,
            index_ids,
            document_ids=document_ids,
            owner_ids=owner_ids,
            source_ids=source_ids,
            lock=True,
        )
        if submission:
            locked_documents = await connection.fetch(
                f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                    WHERE s.index_id=ANY($1::uuid[]) AND s.source_type='submission' AND s.metadata->>'submissionId'=$2
                    ORDER BY d.id FOR UPDATE OF d''',
                index_ids,
                str(submission['id']),
            )
        elif document_ids is not None:
            locked_documents = await connection.fetch(
                f'SELECT * FROM {self.schema}.kb_document WHERE id=ANY($1::uuid[]) ORDER BY id FOR UPDATE',
                document_ids,
            )
        else:
            locked_documents = await connection.fetch(
                f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                    WHERE s.index_id=ANY($1::uuid[]) AND ($2::text[] IS NULL OR s.id::text=ANY($2))
                    ORDER BY d.id FOR UPDATE OF d''',
                index_ids,
                source_ids,
            )
        if operation in ('reindex_document', 'reindex_failed'):
            observed = {row['id']: row['metadata'].get('kbItem') for row in snapshot}
            for row in locked_documents:
                if row['metadata'].get('kbItem') != observed[row['id']]:
                    raise CatalogError('KB item indexing DAG changed')
        return document_ids if operation == 'reindex_failed' else None

    async def _exists(self, connection: AsyncPostgresConnection, table: str) -> bool:
        return bool(
            await connection.fetchval(
                'SELECT to_regclass($1)', f'{self.schema}.{table}'
            )
        )

    async def _index(
        self,
        connection: AsyncPostgresConnection,
        scope: models.Scope,
        index_id: UUID | None = None,
        name: str | None = None,
        *,
        lock: bool = False,
        removing: bool = False,
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            f'''SELECT * FROM {self.schema}.kb_index WHERE tenant_id=$1 AND organization_id=$2
                AND ($3::uuid IS NULL OR id=$3) AND ($4::text IS NULL OR name=$4)'''
            + (' FOR UPDATE' if lock else ''),
            scope.tenant_id,
            scope.organization_id,
            index_id,
            name,
        )
        if row is None:
            raise CatalogError('Knowledge base not found', 404)
        if row['metadata'].get('removalPending') and not removing:
            raise CatalogError('Knowledge base removal is pending')
        return row

    async def _source(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        source_id: UUID,
        *,
        lock: bool = False,
        removing: bool = False,
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            f'SELECT * FROM {self.schema}.kb_source WHERE id=$1 AND index_id=$2'
            + (' FOR UPDATE' if lock else ''),
            source_id,
            index['id'],
        )
        if row is None:
            raise CatalogError('Knowledge base source not found', 404)
        if not removing and (
            row['status'] == 'disabled' or row['metadata'].get('removalPending')
        ):
            raise CatalogError('Source removal is pending')
        return row

    async def _document(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        document_id: UUID,
        *,
        lock: bool = False,
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                WHERE d.id=$1 AND s.index_id=$2'''
            + (' FOR UPDATE OF d' if lock else ''),
            document_id,
            index['id'],
        )
        if row is None:
            raise CatalogError('Knowledge base document not found', 404)
        return row

    async def _source_dto(
        self, connection: AsyncPostgresConnection, source: dict[str, Any]
    ) -> dict[str, Any]:
        dto = camel_row(source)
        dto['documentCount'] = await connection.fetchval(
            f'''SELECT count(*) FROM {self.schema}.kb_document d WHERE d.source_id=$1
                AND NOT EXISTS(SELECT 1 FROM {self.schema}.kb_item_state item
                  WHERE item.index_id=$2 AND item.source_id=d.source_id::text
                  AND item.item_id=d.metadata->'kbItem'->>'itemId'
                  AND item.indexing_dag_id::text=d.metadata->'kbItem'->>'indexingDagId'
                  AND item.state='removed' AND item.cleanup_state='complete')''',
            source['id'],
            source['index_id'],
        )
        dto['schedule'] = None
        if source.get('trigger_config_id') and await self._exists(connection, 'sensor'):
            sensor = await connection.fetchrow(
                f'SELECT id,external_id,config,status::text FROM {self.schema}.sensor WHERE external_id=$1',
                source['trigger_config_id'],
            )
            if sensor:
                dto['schedule'] = {
                    **json_safe(sensor['config']),
                    'config': json_safe(sensor['config']),
                    'status': sensor['status'],
                    'sensorId': str(sensor['id']),
                    'externalId': str(sensor['external_id']),
                }
        dto['latestRun'] = None
        if source['source_type'] in ('datasource', 'repository') and await self._exists(
            connection, 'dag'
        ):
            planner = (
                'kb_datasource_sync'
                if source['source_type'] == 'datasource'
                else 'kb_repository_refresh'
            )
            run = await connection.fetchrow(
                f'''SELECT d.id AS dag_id,d.state::text,d.completed_on FROM {self.schema}.dag d
                    JOIN {self.schema}.kb_index k ON k.id=$1 WHERE d.project_id=k.tenant_id::text
                    AND d.ref_id=$2 AND d.planner=$3 ORDER BY d.created_on DESC,d.id DESC LIMIT 1''',
                source['index_id'],
                str(source['id']),
                planner,
            )
            dto['latestRun'] = camel_row(run) if run else None
            if run:
                job_id = None
                if source['source_type'] == 'repository' and await self._exists(
                    connection, 'job_search_document'
                ):
                    job_id = await connection.fetchval(
                        f'''SELECT job_id FROM {self.schema}.job_search_document
                            WHERE dag_id=$1 AND endpoint=$2 ORDER BY job_id LIMIT 1''',
                        run['dag_id'],
                        'repository_executor://refresh_repository',
                    )
                dto['latestRun']['jobId'] = json_safe(job_id)
        return dto

    async def _index_dto(
        self, connection: AsyncPostgresConnection, index: dict[str, Any]
    ) -> dict[str, Any]:
        dto = camel_row(index)
        dto['configurationRevision'] = int(
            index['metadata'].get('configurationRevision', 0)
        )
        dto['retrievalBackend'] = backend(index)
        sources = await connection.fetch(
            f'SELECT * FROM {self.schema}.kb_source WHERE index_id=$1 ORDER BY created_at DESC',
            index['id'],
        )
        dto['sources'] = [
            await self._source_dto(connection, source) for source in sources
        ]
        dto['sourceCount'] = len(sources)
        dto['documentCount'] = sum(source['documentCount'] for source in dto['sources'])
        binding = await connection.fetchrow(
            f"SELECT * FROM {self.schema}.resource_workflow_binding WHERE resource_type='kb_index' AND resource_id=$1 AND tenant_id=$2",
            index['id'],
            index['tenant_id'],
        )
        dto['workflowBinding'] = camel_row(binding) if binding else None
        return dto

    async def _jobs(
        self, connection: AsyncPostgresConnection, dag_id: UUID
    ) -> list[dict[str, Any]]:
        if not await self._exists(connection, 'job_search_document'):
            return []
        return json_safe(
            await connection.fetch(
                f'''SELECT jsd.node_label AS name,j.state::text,j.started_on,j.completed_on,
                   (kv.value->>'runtime_env_json')::jsonb->'error'->>'message' AS error
                FROM {self.schema}.job_search_document jsd
                JOIN {self.schema}.job j ON j.id=jsd.job_id AND j.name=jsd.queue_name
                LEFT JOIN {self.schema}.kv_store_worker kv ON kv.namespace='job'
                  AND kv.key='marie_internal/job_info_' || j.id::text
                WHERE jsd.dag_id=$1''',
                dag_id,
            )
        )

    async def _document_dto(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        document: dict[str, Any],
    ) -> dict[str, Any]:
        dto = camel_row(document)
        dto['indexId'] = str(index['id'])
        mapping = document['metadata'].get('kbItem') or {}
        item = await connection.fetchrow(
            f'''SELECT * FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2
                  AND source_id=$3 AND item_id=$4 AND indexing_dag_id::text=$5''',
            index['tenant_id'],
            index['id'],
            str(document['source_id']),
            mapping.get('itemId'),
            mapping.get('indexingDagId'),
        )
        if item:
            item['rows'] = await self._jobs(connection, item['indexing_dag_id'])
            item['indexed_at'] = None
            if item['indexed_dag_id'] and await self._exists(connection, 'dag'):
                item['indexed_at'] = await connection.fetchval(
                    f"SELECT completed_on FROM {self.schema}.dag WHERE id=$1 AND state='completed'",
                    item['indexed_dag_id'],
                )
        dto['itemState'] = json_safe(item)
        return dto

    async def _document_origin(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        source: dict[str, Any],
        document: dict[str, Any],
    ) -> models.Origin:
        saved = document['metadata'].get('origin')
        if saved:
            return models.Origin.model_validate(saved)
        if source['source_type'] == 'submission':
            submission_id = source['metadata'].get('submissionId')
            owner = await connection.fetchrow(
                f'''SELECT d.id,d.submission_id FROM {self.schema}.submission_documents d
                    JOIN {self.schema}.submissions s ON s.id=d.submission_id WHERE s.id::text=$1
                    AND s.tenant_id=$2 AND d.storage_key=$3''',
                submission_id,
                index['tenant_id'],
                document['s3_key'],
            )
            if owner is None:
                raise CatalogError(
                    'Submission storage ownership cannot be authorized', 403
                )
            return models.Origin(
                kind='submission',
                submissionId=owner['submission_id'],
                submissionDocumentId=owner['id'],
            )
        return models.Origin()

    async def _list_indexes(
        self, connection: AsyncPostgresConnection, command: models.ListIndexes
    ) -> list[dict[str, Any]]:
        rows = await connection.fetch(
            f'''SELECT * FROM {self.schema}.kb_index WHERE organization_id=$1
                AND ($2::uuid IS NULL OR tenant_id=$2) AND ($3::uuid[] IS NULL OR tenant_id=ANY($3))
                AND ($4::uuid[] IS NULL OR id=ANY($4)) ORDER BY updated_at DESC''',
            command.organization_id,
            command.tenant_id,
            command.tenantIds,
            command.indexIds,
        )
        return [await self._index_dto(connection, row) for row in rows]

    async def _get_index(
        self, connection: AsyncPostgresConnection, command: models.GetIndex
    ) -> dict[str, Any]:
        return await self._index_dto(
            connection,
            await self._index(
                connection, command, command.indexId, command.name, removing=True
            ),
        )

    async def _binding(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        await connection.execute(
            f'''INSERT INTO {self.schema}.resource_workflow_binding(resource_type,resource_id,tenant_id,workflow_name,run_params)
                VALUES ('kb_index',$1,$2,'kb_indexing',$3::jsonb)
                ON CONFLICT (resource_type,resource_id) DO UPDATE SET tenant_id=EXCLUDED.tenant_id,
                workflow_name=EXCLUDED.workflow_name,run_params=EXCLUDED.run_params,updated_at=clock_timestamp()''',
            index['id'],
            index['tenant_id'],
            json.dumps(params),
        )

    async def _create_index(
        self, connection: AsyncPostgresConnection, command: models.CreateIndex
    ) -> dict[str, Any]:
        size, overlap = (
            command.chunkSize or 1024,
            (200 if command.chunkOverlap is None else command.chunkOverlap),
        )
        if overlap >= size:
            raise CatalogError('Chunk overlap must be smaller than chunk size', 400)
        index = await connection.fetchrow(
            f'''INSERT INTO {self.schema}.kb_index(id,organization_id,tenant_id,workspace_id,name,embedding_model,
                embedding_dim,chunk_size,chunk_overlap,segmentation_mode,hybrid_enabled,hybrid_alpha,multimodal_enabled,metadata,created_by)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb,$15) RETURNING *''',
            command.indexId or uuid4(),
            command.organization_id,
            command.tenant_id,
            command.workspaceId,
            command.name,
            command.embeddingModel,
            command.embeddingDim,
            size,
            overlap,
            command.segmentationMode or 'semantic',
            True if command.hybridEnabled is None else command.hybridEnabled,
            0.5 if command.hybridAlpha is None else command.hybridAlpha,
            True if command.multimodalEnabled is None else command.multimodalEnabled,
            json.dumps(
                {
                    **(command.metadata or {}),
                    'retrievalBackend': command.retrievalBackend or 'pgvector',
                    'configurationRevision': 0,
                }
            ),
            command.actor_id,
        )
        await self._binding(connection, index, run_params(index, {}, command.runParams))
        return await self._index_dto(connection, index)

    def _revision(self, index: dict[str, Any], expected: int) -> int:
        current = int(index['metadata'].get('configurationRevision', 0))
        if current != expected:
            raise CatalogError(
                'Knowledge base configuration changed. Reload before saving.'
            )
        return current + 1

    async def _update_index_configuration(
        self, connection: AsyncPostgresConnection, command: models.UpdateIndex
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId, lock=True)
        revision = self._revision(index, command.expectedConfigurationRevision)
        binding = await connection.fetchrow(
            f"SELECT run_params FROM {self.schema}.resource_workflow_binding WHERE resource_type='kb_index' AND resource_id=$1",
            index['id'],
        )
        previous = run_params(index, binding['run_params'] if binding else {})
        values = command.data
        metadata = merge_metadata(index['metadata'], values.metadata or {})
        if values.retrievalBackend and values.retrievalBackend != backend(index):
            occupied = await connection.fetchval(
                f'''SELECT EXISTS(SELECT 1 FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id WHERE s.index_id=$1)
                    OR EXISTS(SELECT 1 FROM {self.schema}.kb_item_state WHERE index_id=$1 AND tenant_id=$2
                      AND (state <> 'removed' OR cleanup_state <> 'complete'))''',
                index['id'],
                index['tenant_id'],
            )
            if occupied:
                raise CatalogError(
                    'Changing the search backend requires an empty knowledge base or a separate rebuild'
                )
            metadata['retrievalBackend'] = values.retrievalBackend
        metadata['configurationRevision'] = revision
        size = index['chunk_size'] if values.chunkSize is None else values.chunkSize
        overlap = (
            index['chunk_overlap']
            if values.chunkOverlap is None
            else values.chunkOverlap
        )
        if overlap >= size:
            raise CatalogError('Chunk overlap must be smaller than chunk size', 400)
        index = await connection.fetchrow(
            f'''UPDATE {self.schema}.kb_index SET chunk_size=$2,chunk_overlap=$3,segmentation_mode=COALESCE($4,segmentation_mode),
                hybrid_enabled=COALESCE($5,hybrid_enabled),hybrid_alpha=COALESCE($6,hybrid_alpha),
                multimodal_enabled=COALESCE($7,multimodal_enabled),metadata=$8::jsonb,updated_by=$9,updated_at=clock_timestamp()
                WHERE id=$1 RETURNING *''',
            index['id'],
            size,
            overlap,
            values.segmentationMode,
            values.hybridEnabled,
            values.hybridAlpha,
            values.multimodalEnabled,
            json.dumps(metadata),
            command.actor_id,
        )
        params = run_params(index, previous, values.runParams)
        await self._binding(connection, index, params)
        consumed = (
            'chunk_size',
            'chunk_overlap',
            'segmentation_mode',
            'multimodal',
            'retrieval_backend',
            'embedding_model',
            'embedding_dim',
        )
        if any(previous.get(key) != params.get(key) for key in consumed):
            await connection.execute(
                f'''WITH changed AS (
                    UPDATE {self.schema}.kb_item_state SET indexing_dag_id=gen_random_uuid(),state='pending',admitted_at=NULL,
                    error=NULL,run_params=run_params || $3::jsonb,updated_at=clock_timestamp()
                    WHERE tenant_id=$1 AND index_id=$2 AND state <> 'removed'
                      AND owner_type IN ('kb_document','submission_document','datasource')
                      AND NOT(state='error' AND COALESCE(input_ref->>'kind','')='upstream' AND NOT input_ref ? 'artifact')
                    RETURNING source_id,item_id,indexing_dag_id,input_ref)
                    UPDATE {self.schema}.kb_document doc SET dag_id=c.indexing_dag_id,status='pending',error_message=NULL,
                    stage=CASE WHEN c.input_ref->>'kind'='upstream' AND NOT c.input_ref ? 'artifact' THEN 'waiting_upstream' ELSE 'queued' END,
                    metadata=jsonb_set(doc.metadata,'{{kbItem,indexingDagId}}',to_jsonb(c.indexing_dag_id::text))
                    FROM changed c WHERE doc.source_id::text=c.source_id AND doc.metadata->'kbItem'->>'itemId'=c.item_id''',
                index['tenant_id'],
                index['id'],
                json.dumps(params),
            )
            await connection.execute(
                f'''UPDATE {self.schema}.kb_item_state SET run_params=run_params || $3::jsonb,updated_at=clock_timestamp()
                    WHERE tenant_id=$1 AND index_id=$2 AND state='error' AND input_ref->>'kind'='upstream'
                      AND NOT input_ref ? 'artifact' AND owner_type IN ('kb_document','submission_document','datasource')''',
                index['tenant_id'],
                index['id'],
                json.dumps(params),
            )
        return await self._index_dto(connection, index)

    async def _list_sources(
        self, connection: AsyncPostgresConnection, command: models.ListSources
    ) -> list[dict[str, Any]]:
        await self._index(connection, command, command.indexId, removing=True)
        rows = await connection.fetch(
            f'SELECT * FROM {self.schema}.kb_source WHERE index_id=$1 AND ($2::text IS NULL OR source_type=$2) ORDER BY created_at',
            command.indexId,
            command.sourceType,
        )
        return [await self._source_dto(connection, row) for row in rows]

    async def _get_source(
        self, connection: AsyncPostgresConnection, command: models.SourceScope
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId, removing=True)
        return await self._source_dto(
            connection,
            await self._source(connection, index, command.sourceId, removing=True),
        )

    async def _source_runs(
        self, connection: AsyncPostgresConnection, command: models.SourceRuns
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId)
        source = await self._source(connection, index, command.sourceId)
        if source['source_type'] != 'repository':
            raise CatalogError('Refresh history requires a repository source', 400)
        if not await self._exists(connection, 'dag'):
            return {'runs': [], 'total': 0}
        arguments = (
            str(index['tenant_id']),
            str(source['id']),
            'kb_repository_refresh',
        )
        rows = await connection.fetch(
            f'''SELECT id AS dag_id,state::text,created_on,started_on,completed_on
                FROM {self.schema}.dag WHERE project_id=$1 AND ref_id=$2 AND planner=$3
                ORDER BY created_on DESC,id DESC LIMIT $4 OFFSET $5''',
            *arguments,
            command.pageSize,
            (command.page - 1) * command.pageSize,
        )
        total = await connection.fetchval(
            f'''SELECT count(*) FROM {self.schema}.dag
                WHERE project_id=$1 AND ref_id=$2 AND planner=$3''',
            *arguments,
        )
        jobs = {}
        endpoint = 'repository_executor://refresh_repository'
        if rows and await self._exists(connection, 'job'):
            jobs = {
                row['dag_id']: row
                for row in await connection.fetch(
                    f'''SELECT DISTINCT ON (dag_id) dag_id,id AS job_id,
                        data->'metadata'->>'on' AS endpoint,
                        data->'metadata'->>'user_id' AS actor_id,
                        COALESCE(data->>'trigger',data->'metadata'->>'trigger') AS trigger,
                        COALESCE(data->>'source_sensor_id',data->>'sensor_id') AS sensor_id,
                        data->>'sensor_name' AS sensor_name,
                        data->'run_config'->>'scheduled_at' AS scheduled_at
                        FROM {self.schema}.job WHERE dag_id=ANY($1::uuid[])
                        ORDER BY dag_id,(data->'metadata'->>'on'=$2) DESC NULLS LAST,id''',
                    [row['dag_id'] for row in rows],
                    endpoint,
                )
            }
        runs = []
        for row in rows:
            job = jobs.get(row['dag_id']) or {}
            trigger = job.get('trigger')
            if trigger == 'schedule':
                trigger = 'schedule'
            elif job.get('sensor_id'):
                trigger = 'sensor'
            elif trigger == 'manual' or job.get('actor_id'):
                trigger = 'manual'
            else:
                trigger = 'unknown'
            runs.append(
                {
                    **camel_row(row),
                    'jobId': (
                        json_safe(job.get('job_id'))
                        if job.get('endpoint') == endpoint
                        else None
                    ),
                    'trigger': trigger,
                    'actorId': job.get('actor_id'),
                    'sensorId': job.get('sensor_id'),
                    'sensorName': job.get('sensor_name'),
                    'scheduledAt': job.get('scheduled_at'),
                }
            )
        return {'runs': runs, 'total': total}

    async def _prepare_upload(
        self, connection: AsyncPostgresConnection, command: models.Upload
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId, lock=True)
        source = await connection.fetchrow(
            f"INSERT INTO {self.schema}.kb_source(index_id,name,source_type,created_by) VALUES($1,$2,'file',$3) RETURNING *",
            index['id'],
            command.fileName,
            command.actor_id,
        )
        return {
            'source': await self._source_dto(connection, source),
            's3Key': f"tenants/{index['tenant_id']}/kb-indexes/{index['id']}/sources/{source['id']}/{command.fileName}",
        }

    async def _save_source(
        self, connection: AsyncPostgresConnection, command: models.SaveSource
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId, lock=True)
        revision = self._revision(index, command.expectedConfigurationRevision)
        if command.workspaceId is not None:
            if command.sourceType not in ('datasource', 'repository'):
                raise CatalogError(
                    'Only executable sources can assign a workspace', 400
                )
            if index['workspace_id'] not in (None, command.workspaceId):
                raise CatalogError('Knowledge base workspace cannot change', 409)
            if index['workspace_id'] is None:
                index = await connection.fetchrow(
                    f'''UPDATE {self.schema}.kb_index SET workspace_id=$2,updated_by=$3,
                        updated_at=clock_timestamp() WHERE id=$1 RETURNING *''',
                    index['id'],
                    command.workspaceId,
                    command.actor_id,
                )
        existing = (
            await self._source(connection, index, command.sourceId, lock=True)
            if command.sourceId
            else None
        )
        if existing and existing['source_type'] != command.sourceType:
            raise CatalogError('Source kind cannot change')
        if (
            'triggerConfigId' in command.model_fields_set
            and command.triggerConfigId
            != (existing['trigger_config_id'] if existing else None)
        ):
            raise CatalogError(
                'Manage schedule ownership through the source schedule command'
            )
        if (
            command.sourceType in ('datasource', 'repository')
            and not index['workspace_id']
        ):
            raise CatalogError('This source requires a knowledge base workspace', 400)
        if existing:
            await self._require_inactive_schedule(connection, existing)
            values = command.model_dump(exclude_unset=True)
            source = await connection.fetchrow(
                f'''UPDATE {self.schema}.kb_source SET name=$2,extension_install_id=$3,datasource_ref=$4,dataset_inputs=$5::jsonb,
                    sync_mode=$6,trigger_config_id=$7,status=$8,metadata=$9::jsonb,updated_at=clock_timestamp() WHERE id=$1 RETURNING *''',
                existing['id'],
                command.name,
                values.get('extensionInstallId', existing['extension_install_id']),
                values.get('datasourceRef', existing['datasource_ref']),
                json.dumps(values.get('datasetInputs', existing['dataset_inputs'])),
                values.get('syncMode', existing['sync_mode']),
                values.get('triggerConfigId', existing['trigger_config_id']),
                values.get('status', existing['status']),
                json.dumps(
                    merge_metadata(existing['metadata'], command.metadata or {})
                ),
            )
        else:
            source = await connection.fetchrow(
                f'''INSERT INTO {self.schema}.kb_source(id,index_id,name,source_type,extension_install_id,datasource_ref,
                    dataset_inputs,sync_mode,trigger_config_id,status,metadata,created_by)
                    VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11::jsonb,$12) RETURNING *''',
                uuid4(),
                index['id'],
                command.name,
                command.sourceType,
                command.extensionInstallId,
                command.datasourceRef,
                json.dumps(command.datasetInputs or {}),
                command.syncMode or 'manual',
                command.triggerConfigId,
                command.status
                or ('registered' if command.sourceType == 'datasource' else 'pending'),
                json.dumps(command.metadata or {}),
                command.actor_id,
            )
        if (
            'schedule' in command.model_fields_set
            or command.syncMode == 'manual'
            and source['source_type'] in ('datasource', 'repository')
        ):
            source = await self._save_schedule(
                connection, command, index, source, command.schedule
            )
        await self._set_revision(connection, index, revision)
        return {
            'source': await self._source_dto(connection, source),
            'configurationRevision': revision,
        }

    async def _set_revision(
        self, connection: AsyncPostgresConnection, index: dict[str, Any], revision: int
    ) -> None:
        await connection.execute(
            f"UPDATE {self.schema}.kb_index SET metadata=jsonb_set(metadata,'{{configurationRevision}}',$2::jsonb),updated_at=clock_timestamp() WHERE id=$1",
            index['id'],
            json.dumps(revision),
        )

    async def _ownership(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        source: dict[str, Any],
        file: models.FileAdmission,
    ) -> None:
        if file.origin.kind == 'submission':
            if source['source_type'] != 'submission' or source['metadata'].get(
                'submissionId'
            ) != str(file.origin.submissionId):
                raise CatalogError('Document source ownership changed', 403)
            authorized = await connection.fetchval(
                f'''SELECT d.id FROM {self.schema}.submission_documents d JOIN {self.schema}.submissions s ON s.id=d.submission_id
                    WHERE d.id=$1 AND s.id=$2 AND s.tenant_id=$3 AND d.storage_key=$4
                    AND d.indexing_status IS DISTINCT FROM 'removing' AND s.enable_semantic_search
                    AND (s.kb_index_id=$5 OR EXISTS(SELECT 1 FROM {self.schema}.submission_kb_indexes l
                      WHERE l.submission_id=s.id AND l.kb_index_id=$5)) FOR UPDATE OF d''',
                file.origin.submissionDocumentId,
                file.origin.submissionId,
                index['tenant_id'],
                file.s3Key,
                index['id'],
            )
            if not authorized:
                raise CatalogError(
                    'Submission document is no longer available to this knowledge base',
                    403,
                )
        else:
            prefixes = [
                f"tenants/{index['tenant_id']}/kb-indexes/{index['id']}/",
                f"kb-indexes/{index['id']}/",
            ]
            if index['workspace_id']:
                prefixes.append(
                    f"workspaces/{index['workspace_id']}/kb-indexes/{index['id']}/"
                )
            if source['source_type'] in (
                'submission',
                'datasource',
                'repository',
            ) or not any(
                file.s3Key.startswith(f"{prefix}sources/{source['id']}/")
                and not file.s3Key.endswith('/')
                for prefix in prefixes
            ):
                raise CatalogError(
                    'Document storage ownership cannot be authorized', 403
                )

    async def _admit_file(
        self,
        connection: AsyncPostgresConnection,
        command: models.AdmitDocuments,
        index: dict[str, Any],
        file: models.FileAdmission,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        source = await self._source(connection, index, file.sourceId, lock=True)
        await self._ownership(connection, index, source, file)
        document = await connection.fetchrow(
            f'SELECT * FROM {self.schema}.kb_document WHERE source_id=$1 AND s3_key=$2 FOR UPDATE',
            file.sourceId,
            file.s3Key,
        )
        if document and document['metadata'].get('removalPending'):
            raise CatalogError('Document removal is pending')
        origin = file.origin.model_dump(mode='json', exclude_none=True)
        if not document:
            document = await connection.fetchrow(
                f'''INSERT INTO {self.schema}.kb_document(source_id,name,s3_key,size,content_type,created_by,metadata)
                    VALUES($1,$2,$3,$4,$5,$6,$7::jsonb) RETURNING *''',
                file.sourceId,
                file.name,
                file.s3Key,
                file.size,
                file.contentType,
                command.actor_id,
                json.dumps({'origin': origin}),
            )
            await connection.execute(
                f'UPDATE {self.schema}.kb_source SET file_count=file_count+1,updated_at=clock_timestamp() WHERE id=$1',
                source['id'],
            )
        mapping = document['metadata'].get('kbItem') or {}
        item_id = mapping.get('itemId') or str(document['id'])
        previous = await connection.fetchrow(
            f'SELECT * FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4 FOR UPDATE',
            index['tenant_id'],
            index['id'],
            str(source['id']),
            item_id,
        )
        owner_type = (
            'submission_document' if file.origin.kind == 'submission' else 'kb_document'
        )
        owner_id = (
            str(file.origin.submissionDocumentId)
            if file.origin.kind == 'submission'
            else str(document['id'])
        )
        if previous and previous['state'] == 'removed':
            if previous['owner_type'] != owner_type or previous['owner_id'] != owner_id:
                raise CatalogError('KB item owner changed')
            item_id = str(uuid4())
        reference = None
        if file.upstreamPlanner:
            reference = {
                'kind': 'upstream',
                'planner': file.upstreamPlanner,
                'processing_workflow': 'kb_index_content',
            }
            if file.upstreamFabrics:
                reference['fabric_group_ids'] = file.upstreamFabrics
        item = await connection.fetchrow(
            f'''INSERT INTO {self.schema}.kb_item_state(tenant_id,index_id,source_id,item_id,owner_type,owner_id,uri,
                indexing_dag_id,run_params,backend,store_name,input_ref)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12::jsonb)
                ON CONFLICT(tenant_id,index_id,source_id,item_id) DO UPDATE SET updated_at=kb_item_state.updated_at
                WHERE kb_item_state.state <> 'removed' AND kb_item_state.owner_type=EXCLUDED.owner_type
                AND kb_item_state.owner_id=EXCLUDED.owner_id AND kb_item_state.uri=EXCLUDED.uri
                AND (kb_item_state.run_params-'embedding_model'-'embedding_dim')=(EXCLUDED.run_params-'embedding_model'-'embedding_dim')
                AND (NOT kb_item_state.run_params ? 'embedding_model' OR kb_item_state.run_params->'embedding_model'=EXCLUDED.run_params->'embedding_model')
                AND (NOT kb_item_state.run_params ? 'embedding_dim' OR kb_item_state.run_params->'embedding_dim'=EXCLUDED.run_params->'embedding_dim')
                AND kb_item_state.backend=EXCLUDED.backend AND kb_item_state.store_name=EXCLUDED.store_name
                AND kb_item_state.input_ref->>'planner' IS NOT DISTINCT FROM EXCLUDED.input_ref->>'planner'
                AND kb_item_state.input_ref->'fabric_group_ids' IS NOT DISTINCT FROM EXCLUDED.input_ref->'fabric_group_ids'
                RETURNING *''',
            index['tenant_id'],
            index['id'],
            str(source['id']),
            item_id,
            owner_type,
            owner_id,
            f's3://{command.bucket}/{file.s3Key}',
            uuid4(),
            json.dumps(params),
            backend(index),
            'kb_text_section' if backend(index) == 'postgresql_text' else 'kb_vectors',
            json.dumps(reference) if reference else None,
        )
        if item is None:
            raise CatalogError(
                'KB item settings or owner changed; re-index to create a fresh indexing DAG'
            )
        if file.upstreamPlanner:
            await connection.execute(
                f'''WITH upstream AS (SELECT d.id,d.ref_id,d.ref_type,j.data->'metadata'->>'fabric_group_id' AS fabric
                    FROM {self.schema}.dag d JOIN {self.schema}.job j ON j.dag_id=d.id
                    WHERE d.project_id=$1 AND d.planner=$2 AND j.data->'metadata'->>'uri'=$3
                    AND ($4::text[]='{{}}' OR j.data->'metadata'->>'fabric_group_id'=ANY($4))
                    ORDER BY d.created_on DESC,d.id DESC LIMIT 1)
                    UPDATE {self.schema}.kb_item_state item SET input_ref=item.input_ref || jsonb_build_object(
                      'dag_id',u.id::text,'ref_id',u.ref_id,'ref_type',u.ref_type,'fabric_group_id',u.fabric)
                    FROM upstream u WHERE item.tenant_id=$5 AND item.index_id=$6 AND item.source_id=$7 AND item.item_id=$8
                    AND item.admitted_at IS NULL AND item.input_ref->>'dag_id' IS NULL''',
                str(index['tenant_id']),
                file.upstreamPlanner,
                f's3://{command.bucket}/{file.s3Key}',
                file.upstreamFabrics,
                index['tenant_id'],
                index['id'],
                str(source['id']),
                item_id,
            )
        document = await connection.fetchrow(
            f'''UPDATE {self.schema}.kb_document SET metadata=$2::jsonb,dag_id=$3,updated_at=clock_timestamp()
                WHERE id=$1 RETURNING *''',
            document['id'],
            json.dumps(
                {
                    **document['metadata'],
                    'origin': origin,
                    'removalPending': False,
                    'kbItem': {
                        'itemId': item_id,
                        'indexingDagId': str(item['indexing_dag_id']),
                    },
                }
            ),
            item['indexing_dag_id'],
        )
        return await self._document_dto(connection, index, document)

    async def _admit_documents(
        self, connection: AsyncPostgresConnection, command: models.AdmitDocuments
    ) -> list[dict[str, Any]]:
        index = await self._index(connection, command, command.indexId, lock=True)
        binding = await connection.fetchrow(
            f"SELECT run_params FROM {self.schema}.resource_workflow_binding WHERE resource_type='kb_index' AND resource_id=$1",
            index['id'],
        )
        params = run_params(index, binding['run_params'] if binding else {})
        return [
            await self._admit_file(connection, command, index, file, params)
            for file in command.files
        ]

    async def _get_document(
        self, connection: AsyncPostgresConnection, command: models.DocumentScope
    ) -> dict[str, Any]:
        index = await self._document_index(connection, command, removing=True)
        return await self._document_dto(
            connection,
            index,
            await self._document(connection, index, command.documentId),
        )

    async def _list_documents(
        self, connection: AsyncPostgresConnection, command: models.ListDocuments
    ) -> list[dict[str, Any]]:
        index = await self._index(connection, command, command.indexId, removing=True)
        documents = await connection.fetch(
            f'''SELECT d.* FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                WHERE s.index_id=$1 AND ($2::uuid IS NULL OR s.id=$2)
                AND ($3::text IS NULL OR strpos(lower(d.name),lower($3))>0)
                AND NOT EXISTS(SELECT 1 FROM {self.schema}.kb_item_state item
                  WHERE item.index_id=s.index_id AND item.source_id=d.source_id::text
                  AND item.item_id=d.metadata->'kbItem'->>'itemId'
                  AND item.indexing_dag_id::text=d.metadata->'kbItem'->>'indexingDagId'
                  AND item.state='removed' AND item.cleanup_state='complete')
                ORDER BY d.created_at DESC''',
            index['id'],
            command.sourceId,
            command.search,
        )
        return [
            await self._document_dto(connection, index, document)
            for document in documents
        ]

    async def _reindex_document(
        self, connection: AsyncPostgresConnection, command: models.DocumentMutation
    ) -> dict[str, Any]:
        snapshot_index = await self._document_index(connection, command)
        snapshot = await self._document(connection, snapshot_index, command.documentId)
        snapshot_mapping = snapshot['metadata'].get('kbItem') or {}
        previous = await connection.fetchrow(
            f'''SELECT * FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2
                AND source_id=$3 AND item_id=$4''',
            command.tenant_id,
            snapshot_index['id'],
            str(snapshot['source_id']),
            snapshot_mapping.get('itemId'),
        )
        reference = (previous or {}).get('input_ref') or {}
        index = await self._index(connection, command, snapshot_index['id'], lock=True)
        document = await self._document(
            connection, index, command.documentId, lock=True
        )
        if document['metadata'].get('kbItem', {}) != snapshot_mapping:
            raise CatalogError('KB item indexing DAG changed')
        if document['metadata'].get('removalPending'):
            raise CatalogError('Document removal is pending')
        source = await self._source(connection, index, document['source_id'], lock=True)
        origin = await self._document_origin(connection, index, source, document)
        file = models.FileAdmission(
            sourceId=source['id'],
            name=document['name'],
            s3Key=document['s3_key'],
            size=document['size'],
            contentType=document['content_type'],
            origin=origin,
        )
        await self._ownership(connection, index, source, file)
        mapping = document['metadata'].get('kbItem') or {}
        if command.expectedIndexingDagId and str(
            command.expectedIndexingDagId
        ) != mapping.get('indexingDagId'):
            raise CatalogError('KB item indexing DAG changed')
        current = await connection.fetchrow(
            f'''SELECT * FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2
                AND source_id=$3 AND item_id=$4 FOR UPDATE''',
            command.tenant_id,
            index['id'],
            str(source['id']),
            mapping.get('itemId'),
        )
        if not mapping or current is None or current['state'] == 'removed':
            bucket = command.bucket or (
                urlsplit(current['uri']).netloc if current else None
            )
            if not bucket:
                raise CatalogError(
                    'Storage bucket is required to register this legacy document', 400
                )
            if reference.get('kind') == 'upstream':
                file = file.model_copy(
                    update={
                        'upstreamPlanner': reference['planner'],
                        'upstreamFabrics': reference.get('fabric_group_ids', []),
                    }
                )
            admitted = await self._admit_documents(
                connection,
                models.AdmitDocuments(
                    organization_id=command.organization_id,
                    actor_id=command.actor_id,
                    tenant_id=command.tenant_id,
                    indexId=index['id'],
                    bucket=bucket,
                    files=[file],
                ),
            )
            return admitted[0]
        binding = await connection.fetchrow(
            f"SELECT run_params FROM {self.schema}.resource_workflow_binding WHERE resource_type='kb_index' AND resource_id=$1",
            index['id'],
        )
        dag_id = uuid4()
        item = await connection.fetchrow(
            f'''UPDATE {self.schema}.kb_item_state SET indexing_dag_id=$6,state='pending',error=NULL,admitted_at=NULL,
                run_params=$7::jsonb,updated_at=clock_timestamp() WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3
                AND item_id=$4 AND indexing_dag_id=$5 AND state <> 'removed' AND backend=$8 RETURNING *''',
            index['tenant_id'],
            index['id'],
            str(source['id']),
            mapping['itemId'],
            UUID(mapping['indexingDagId']),
            dag_id,
            json.dumps(run_params(index, binding['run_params'] if binding else {})),
            backend(index),
        )
        if item is None:
            raise CatalogError('KB item indexing DAG changed or was removed')
        mapping = {**mapping, 'indexingDagId': str(dag_id)}
        document = await connection.fetchrow(
            f'''UPDATE {self.schema}.kb_document SET metadata=$2::jsonb,dag_id=$3,status='pending',stage='queued',error_message=NULL,
                updated_at=clock_timestamp() WHERE id=$1 RETURNING *''',
            document['id'],
            json.dumps({**document['metadata'], 'kbItem': mapping}),
            dag_id,
        )
        return await self._document_dto(connection, index, document)

    async def _reindex_failed(
        self,
        connection: AsyncPostgresConnection,
        command: models.ReindexFailed,
        *,
        document_ids: list[UUID] | None = None,
    ) -> list[dict[str, Any]]:
        await self._index(connection, command, command.indexId)
        documents = await self._list_documents(
            connection, models.ListDocuments(**command.model_dump(exclude={'bucket'}))
        )
        if document_ids is not None:
            documents = [
                document
                for document in documents
                if UUID(document['id']) in document_ids
            ]
        results = []
        for document in documents:
            item = document['itemState']
            failed = needs_reindex(document)
            if failed:
                results.append(
                    await self._reindex_document(
                        connection,
                        models.DocumentMutation(
                            **command.model_dump(),
                            documentId=document['id'],
                            expectedIndexingDagId=(
                                item['indexing_dag_id'] if item else None
                            ),
                        ),
                    )
                )
        return results

    async def _withdraw_items(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        *,
        source_id: UUID | None = None,
        item_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return json_safe(
            await connection.fetch(
                f'''UPDATE {self.schema}.kb_item_state SET state='removed',
                cleanup_state=CASE WHEN state='removed' THEN cleanup_state ELSE 'pending' END,updated_at=clock_timestamp()
                WHERE tenant_id=$1 AND index_id=$2 AND ($3::text IS NULL OR source_id=$3)
                AND ($4::text IS NULL OR item_id=$4) RETURNING *''',
                index['tenant_id'],
                index['id'],
                str(source_id) if source_id else None,
                item_id,
            )
        )

    async def _withdraw_document(
        self, connection: AsyncPostgresConnection, command: models.DocumentMutation
    ) -> dict[str, Any]:
        index = await self._document_index(
            connection, command, lock=True, removing=True
        )
        document = await self._document(
            connection, index, command.documentId, lock=True
        )
        mapping = document['metadata'].get('kbItem') or {}
        if not mapping:
            raise CatalogError(
                'Runtime item registration is required before document removal'
            )
        if command.expectedIndexingDagId and str(
            command.expectedIndexingDagId
        ) != mapping.get('indexingDagId'):
            raise CatalogError('KB item indexing DAG changed')
        items = (
            await self._withdraw_items(
                connection,
                index,
                source_id=document['source_id'],
                item_id=mapping['itemId'],
            )
            if mapping
            else []
        )
        if not items:
            raise CatalogError(
                'Runtime item registration is unavailable; re-index before removal'
            )
        document = await connection.fetchrow(
            f"UPDATE {self.schema}.kb_document SET metadata=metadata || '{{\"removalPending\":true}}'::jsonb,updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
            document['id'],
        )
        return {
            'document': await self._document_dto(connection, index, document),
            'items': items,
        }

    async def _cleanup_complete(
        self,
        connection: AsyncPostgresConnection,
        index: dict[str, Any],
        source_id: UUID | None = None,
        item_id: str | None = None,
    ) -> None:
        pending = await connection.fetchval(
            f'''SELECT EXISTS(SELECT 1 FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2
                AND ($3::text IS NULL OR source_id=$3) AND ($4::text IS NULL OR item_id=$4)
                AND (state <> 'removed' OR cleanup_state <> 'complete'))''',
            index['tenant_id'],
            index['id'],
            str(source_id) if source_id else None,
            item_id,
        )
        if pending:
            raise CatalogError('Indexed removal is pending runtime confirmation')

    async def _finalize_document_removal(
        self, connection: AsyncPostgresConnection, command: models.DocumentScope
    ) -> dict[str, bool]:
        index = await self._document_index(
            connection, command, lock=True, removing=True
        )
        document = await self._document(
            connection, index, command.documentId, lock=True
        )
        if not document['metadata'].get('removalPending'):
            raise CatalogError('Document has not been withdrawn')
        mapping = document['metadata'].get('kbItem') or {}
        if mapping:
            await self._cleanup_complete(
                connection, index, document['source_id'], mapping['itemId']
            )
        await connection.execute(
            f'DELETE FROM {self.schema}.kb_document WHERE id=$1', document['id']
        )
        await connection.execute(
            f'UPDATE {self.schema}.kb_source SET file_count=GREATEST(0,file_count-1),updated_at=clock_timestamp() WHERE id=$1',
            document['source_id'],
        )
        return {'deleted': True}

    async def _withdraw_source(
        self, connection: AsyncPostgresConnection, command: models.WithdrawSource
    ) -> dict[str, Any]:
        index = await self._index(
            connection, command, command.indexId, lock=True, removing=True
        )
        source = await self._source(
            connection, index, command.sourceId, lock=True, removing=True
        )
        revision = int(index['metadata'].get('configurationRevision', 0))
        if command.expectedConfigurationRevision is not None:
            revision = self._revision(index, command.expectedConfigurationRevision)
            await self._set_revision(connection, index, revision)
        source = await connection.fetchrow(
            f"UPDATE {self.schema}.kb_source SET status='disabled',metadata=metadata || '{{\"removalPending\":true}}'::jsonb,updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
            source['id'],
        )
        await self._pause_sensor(connection, source)
        return {
            'source': await self._source_dto(connection, source),
            'items': await self._withdraw_items(
                connection, index, source_id=source['id']
            ),
            'configurationRevision': revision,
        }

    async def _finalize_source_removal(
        self, connection: AsyncPostgresConnection, command: models.SourceScope
    ) -> dict[str, bool]:
        index = await self._index(
            connection, command, command.indexId, lock=True, removing=True
        )
        source = await self._source(
            connection, index, command.sourceId, lock=True, removing=True
        )
        if not source['metadata'].get('removalPending'):
            raise CatalogError('Source has not been withdrawn')
        await self._cleanup_complete(connection, index, source['id'])
        if source['metadata'].get('codeIndex'):
            raise CatalogError('Repository index cleanup is pending')
        if source['trigger_config_id'] and await self._exists(connection, 'sensor'):
            await connection.execute(
                f'DELETE FROM {self.schema}.sensor WHERE external_id=$1',
                source['trigger_config_id'],
            )
        await connection.execute(
            f'DELETE FROM {self.schema}.kb_source WHERE id=$1', source['id']
        )
        return {'deleted': True}

    async def _pause_sensor(
        self, connection: AsyncPostgresConnection, source: dict[str, Any]
    ) -> None:
        if source['trigger_config_id'] and await self._exists(connection, 'sensor'):
            await connection.execute(
                f"UPDATE {self.schema}.sensor SET status='inactive',updated_at=clock_timestamp() WHERE external_id=$1",
                source['trigger_config_id'],
            )

    async def _withdraw_index(
        self, connection: AsyncPostgresConnection, command: models.IndexScope
    ) -> dict[str, Any]:
        index = await self._index(
            connection, command, command.indexId, lock=True, removing=True
        )
        index = await connection.fetchrow(
            f"UPDATE {self.schema}.kb_index SET metadata=metadata || '{{\"removalPending\":true}}'::jsonb,updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
            index['id'],
        )
        sources = await connection.fetch(
            f'SELECT * FROM {self.schema}.kb_source WHERE index_id=$1', index['id']
        )
        for source in sources:
            await self._pause_sensor(connection, source)
        return {
            'index': await self._index_dto(connection, index),
            'items': await self._withdraw_items(connection, index),
        }

    async def _finalize_index_removal(
        self, connection: AsyncPostgresConnection, command: models.IndexScope
    ) -> dict[str, bool]:
        index = await self._index(
            connection, command, command.indexId, lock=True, removing=True
        )
        if not index['metadata'].get('removalPending'):
            raise CatalogError('Knowledge base has not been withdrawn')
        await self._cleanup_complete(connection, index)
        if await connection.fetchval(
            f"SELECT EXISTS(SELECT 1 FROM {self.schema}.kb_source WHERE index_id=$1 AND metadata ? 'codeIndex')",
            index['id'],
        ):
            raise CatalogError('Repository index cleanup is pending')
        sources = await connection.fetch(
            f'SELECT * FROM {self.schema}.kb_source WHERE index_id=$1', index['id']
        )
        for source in sources:
            if source['trigger_config_id'] and await self._exists(connection, 'sensor'):
                await connection.execute(
                    f'DELETE FROM {self.schema}.sensor WHERE external_id=$1',
                    source['trigger_config_id'],
                )
        if await self._exists(connection, 'submission_kb_indexes'):
            await connection.execute(
                f'DELETE FROM {self.schema}.submission_kb_indexes WHERE kb_index_id=$1',
                index['id'],
            )
            await connection.execute(
                f'UPDATE {self.schema}.submissions SET kb_index_id=NULL WHERE kb_index_id=$1',
                index['id'],
            )
        await connection.execute(
            f"DELETE FROM {self.schema}.resource_workflow_binding WHERE resource_type='kb_index' AND resource_id=$1",
            index['id'],
        )
        await connection.execute(
            f'DELETE FROM {self.schema}.kb_index WHERE id=$1', index['id']
        )
        return {'deleted': True}

    async def _document_index(
        self,
        connection: AsyncPostgresConnection,
        command: models.DocumentScope,
        *,
        lock: bool = False,
        removing: bool = False,
    ) -> dict[str, Any]:
        index_id = command.indexId
        if index_id is None:
            index_id = await connection.fetchval(
                f'''SELECT k.id FROM {self.schema}.kb_index k JOIN {self.schema}.kb_source s ON s.index_id=k.id
                    JOIN {self.schema}.kb_document d ON d.source_id=s.id
                    WHERE d.id=$1 AND k.organization_id=$2 AND k.tenant_id=$3''',
                command.documentId,
                command.organization_id,
                command.tenant_id,
            )
            if index_id is None:
                raise CatalogError('Knowledge base document not found', 404)
        return await self._index(
            connection, command, index_id, lock=lock, removing=removing
        )

    async def _storage_references(
        self, connection: AsyncPostgresConnection, command: models.StorageReferences
    ) -> dict[str, list[str]]:
        if command.excludeIndexId:
            await self._index(
                connection, command, command.excludeIndexId, removing=True
            )
        if command.excludeDocumentId:
            await self._document_index(
                connection,
                models.DocumentScope(
                    organization_id=command.organization_id,
                    actor_id=command.actor_id,
                    tenant_id=command.tenant_id,
                    documentId=command.excludeDocumentId,
                ),
                removing=True,
            )
        rows = await connection.fetch(
            f'''SELECT DISTINCT d.s3_key FROM {self.schema}.kb_document d JOIN {self.schema}.kb_source s ON s.id=d.source_id
                WHERE d.s3_key=ANY($1::text[]) AND ($2::uuid IS NULL OR d.id <> $2)
                  AND ($3::uuid IS NULL OR s.index_id <> $3)''',
            command.keys,
            command.excludeDocumentId,
            command.excludeIndexId,
        )
        return {'keys': [row['s3_key'] for row in rows]}

    async def _check_text_citation(
        self, connection: AsyncPostgresConnection, command: models.CheckCitation
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId)
        citation = command.citation
        source = await self._source(connection, index, citation.sourceId)
        section = await connection.fetchrow(
            f'''SELECT * FROM {self.schema}.kb_text_section_current WHERE tenant_id=$1 AND index_id=$2
                AND source_id=$3 AND item_id=$4 AND section_id=$5 AND indexed_dag_id=$6 AND source_digest=$7''',
            command.tenant_id,
            command.indexId,
            str(citation.sourceId),
            citation.itemId,
            citation.sectionId,
            citation.indexedDagId,
            citation.sourceDigest,
        )
        if section is None:
            raise CatalogError(
                'This passage is no longer available. Run the search again.', 404
            )
        document = await connection.fetchrow(
            f"SELECT * FROM {self.schema}.kb_document WHERE source_id=$1 AND metadata->'kbItem'->>'itemId'=$2",
            citation.sourceId,
            citation.itemId,
        )
        if document:
            if document['metadata'].get('removalPending'):
                raise CatalogError('This source is no longer available', 404)
            origin = await self._document_origin(connection, index, source, document)
            await self._ownership(
                connection,
                index,
                source,
                models.FileAdmission(
                    sourceId=source['id'],
                    name=document['name'],
                    s3Key=document['s3_key'],
                    size=document['size'],
                    origin=origin,
                ),
            )
        elif source['source_type'] != 'datasource':
            raise CatalogError('This source is no longer available', 404)
        elif not await connection.fetchval(
            f'''SELECT EXISTS(SELECT 1 FROM {self.platform_schema}.extension_installs i
                JOIN {self.platform_schema}.extension_packages p ON p.id=i.package_id
                WHERE i.id=$1 AND i.organization_id=$2 AND (i.workspace_id IS NULL OR i.workspace_id=$3)
                  AND i.enabled AND i.status='enabled' AND p.trust_level <> 'blocked')''',
            source['extension_install_id'],
            index['organization_id'],
            index['workspace_id'],
        ):
            raise CatalogError('Datasource installation is no longer available', 404)
        item = await connection.fetchrow(
            f'SELECT * FROM {self.schema}.kb_item_state WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4',
            index['tenant_id'],
            index['id'],
            str(source['id']),
            citation.itemId,
        )
        return {
            'document': (
                await self._document_dto(connection, index, document)
                if document
                else None
            ),
            'source': await self._source_dto(connection, source),
            'itemState': json_safe(item),
        }

    async def _document_evidence(
        self, connection: AsyncPostgresConnection, command: models.DocumentScope
    ) -> dict[str, Any] | None:
        document = await self._get_document(connection, command)
        mapping = document['metadata'].get('kbItem') or {}
        if not mapping:
            return None
        row = await connection.fetchrow(
            f'''SELECT source_id,item_id,section_id,indexed_dag_id,source_digest FROM {self.schema}.kb_text_section_current
                WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4 ORDER BY start_offset,section_id LIMIT 1''',
            command.tenant_id,
            UUID(document['indexId']),
            document['sourceId'],
            mapping['itemId'],
        )
        if row is None:
            return None
        citation = camel_row(row)
        await self._check_text_citation(
            connection,
            models.CheckCitation(
                organization_id=command.organization_id,
                actor_id=command.actor_id,
                tenant_id=command.tenant_id,
                indexId=document['indexId'],
                citation=citation,
            ),
        )
        return citation

    async def _document_timeline(
        self, connection: AsyncPostgresConnection, command: models.DocumentScope
    ) -> dict[str, Any]:
        document = await self._get_document(connection, command)
        return {
            'document': document,
            'jobs': (document['itemState'] or {}).get('rows', []),
        }

    async def _record_audit(
        self, connection: AsyncPostgresConnection, command: models.Audit
    ) -> dict[str, Any]:
        if command.entityType == 'kb_index':
            index_id = command.entityId
        elif command.entityType == 'kb_source':
            index_id = await connection.fetchval(
                f'SELECT index_id FROM {self.schema}.kb_source WHERE id=$1',
                command.entityId,
            )
        else:
            index_id = await connection.fetchval(
                f'SELECT s.index_id FROM {self.schema}.kb_source s JOIN {self.schema}.kb_document d ON d.source_id=s.id WHERE d.id=$1',
                command.entityId,
            )
        if index_id:
            await self._index(connection, command, index_id, removing=True)
        else:
            index_id = await connection.fetchval(
                f'''SELECT index_id FROM {self.schema}.kb_audit_log WHERE organization_id=$1 AND tenant_id=$2
                    AND entity_type=$3 AND entity_id=$4 ORDER BY timestamp DESC LIMIT 1''',
                command.organization_id,
                command.tenant_id,
                command.entityType,
                command.entityId,
            )
            if not index_id:
                raise CatalogError('Audit entity not found', 404)
        row = await connection.fetchrow(
            f'''INSERT INTO {self.schema}.kb_audit_log(organization_id,tenant_id,index_id,entity_type,entity_id,action,
                previous_value,new_value,user_id,ip_address,user_agent) VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10,$11) RETURNING *''',
            command.organization_id,
            command.tenant_id,
            index_id,
            command.entityType,
            command.entityId,
            command.action,
            json.dumps(command.previousValue),
            json.dumps(command.newValue),
            command.actor_id,
            command.ipAddress,
            command.userAgent,
        )
        return camel_row(row)

    async def _list_audit(
        self, connection: AsyncPostgresConnection, command: models.ListAudit
    ) -> dict[str, Any]:
        rows = await connection.fetch(
            f'''SELECT * FROM {self.schema}.kb_audit_log WHERE organization_id=$1 AND tenant_id=$2 AND index_id=$3
                AND ($4::text IS NULL OR action=$4) ORDER BY timestamp DESC,id DESC LIMIT $5 OFFSET $6''',
            command.organization_id,
            command.tenant_id,
            command.indexId,
            command.action,
            command.pageSize,
            (command.page - 1) * command.pageSize,
        )
        count = await connection.fetchval(
            f'''SELECT count(*) FROM {self.schema}.kb_audit_log WHERE organization_id=$1 AND tenant_id=$2 AND index_id=$3
                AND ($4::text IS NULL OR action=$4)''',
            command.organization_id,
            command.tenant_id,
            command.indexId,
            command.action,
        )
        return {'entries': [camel_row(row) for row in rows], 'total': count}

    async def _require_inactive_schedule(
        self, connection: AsyncPostgresConnection, source: dict[str, Any]
    ) -> None:
        if source['trigger_config_id'] and await self._exists(connection, 'sensor'):
            state = await connection.fetchval(
                f'SELECT status::text FROM {self.schema}.sensor WHERE external_id=$1 FOR UPDATE',
                source['trigger_config_id'],
            )
            if state not in (None, 'inactive', 'paused'):
                raise CatalogError(
                    'Pause the source schedule before changing its configuration'
                )

    async def _save_schedule(
        self,
        connection: AsyncPostgresConnection,
        command: models.Scope,
        index: dict[str, Any],
        source: dict[str, Any],
        schedule: models.ScheduleConfig | None,
    ) -> dict[str, Any]:
        if (
            source['source_type'] not in ('datasource', 'repository')
            or not index['workspace_id']
        ):
            raise CatalogError(
                'A source schedule requires a datasource or repository and workspace',
                400,
            )
        await self._require_inactive_schedule(connection, source)
        external = source['trigger_config_id']
        if schedule is None:
            if external:
                await connection.execute(
                    f'DELETE FROM {self.schema}.sensor WHERE external_id=$1', external
                )
            return await connection.fetchrow(
                f"UPDATE {self.schema}.kb_source SET trigger_config_id=NULL,sync_mode='manual',updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
                source['id'],
            )
        external = external or uuid4()
        config = {
            **schedule.model_dump(),
            'kbIndexId': str(index['id']),
            'kbSourceId': str(source['id']),
        }
        await connection.execute(
            f'''INSERT INTO {self.schema}.sensor(external_id,name,config,target_job_name,target_dag_id,sensor_type,status)
                VALUES($1,$2,$3::jsonb,NULL,NULL,'schedule','inactive') ON CONFLICT(external_id) DO UPDATE
                SET name=EXCLUDED.name,config=EXCLUDED.config,target_job_name=NULL,target_dag_id=NULL,
                cursor=NULL,status='inactive',updated_at=clock_timestamp()''',
            external,
            f"{source['name']} discovery",
            json.dumps(config),
        )
        return await connection.fetchrow(
            f"UPDATE {self.schema}.kb_source SET trigger_config_id=$2,sync_mode='scheduled',updated_at=clock_timestamp() WHERE id=$1 RETURNING *",
            source['id'],
            external,
        )

    async def _set_source_schedule(
        self, connection: AsyncPostgresConnection, command: models.SetSourceSchedule
    ) -> dict[str, Any]:
        index = await self._index(connection, command, command.indexId, lock=True)
        revision = self._revision(index, command.expectedConfigurationRevision)
        source = await self._source(connection, index, command.sourceId, lock=True)
        if command.action == 'save':
            source = await self._save_schedule(
                connection, command, index, source, command.schedule
            )
        else:
            if not source['trigger_config_id'] or not index['workspace_id']:
                raise CatalogError('Save a source schedule first')
            sensor = await connection.fetchrow(
                f'SELECT config,status::text FROM {self.schema}.sensor WHERE external_id=$1 FOR UPDATE',
                source['trigger_config_id'],
            )
            if (
                not sensor
                or sensor['config'].get('kbIndexId') != str(index['id'])
                or sensor['config'].get('kbSourceId') != str(source['id'])
            ):
                raise CatalogError('Source schedule ownership changed')
            config = dict(sensor['config'])
            if command.action == 'activate':
                await self._require_inactive_schedule(connection, source)
                planner = (
                    'kb_datasource_sync'
                    if source['source_type'] == 'datasource'
                    else 'kb_repository_refresh'
                )
                if source['source_type'] not in ('datasource', 'repository'):
                    raise CatalogError('Unsupported source schedule', 400)
                config['run_config'] = {
                    'planner': planner,
                    'project_id': str(index['tenant_id']),
                    'tenant_id': str(index['tenant_id']),
                    'index_id': str(index['id']),
                    'source_id': str(source['id']),
                    'organization_id': str(command.organization_id),
                    'workspace_id': str(index['workspace_id']),
                    'user_id': str(command.actor_id),
                    'ref_type': (
                        'kb_datasource'
                        if source['source_type'] == 'datasource'
                        else 'kb_repository'
                    ),
                    'ref_id': str(source['id']),
                }
                await connection.execute(
                    f"UPDATE {self.schema}.sensor SET config=$2::jsonb,target_job_name=$3,status='active',cursor=NULL,updated_at=clock_timestamp() WHERE external_id=$1",
                    source['trigger_config_id'],
                    json.dumps(config),
                    planner,
                )
            else:
                await self._pause_sensor(connection, source)
        await self._set_revision(connection, index, revision)
        return {
            'source': await self._source_dto(connection, source),
            'configurationRevision': revision,
        }

    async def _owns_schedule(
        self, connection: AsyncPostgresConnection, command: models.OwnsSchedule
    ) -> dict[str, bool]:
        return {
            'owned': bool(
                await connection.fetchval(
                    f'''SELECT EXISTS(SELECT 1 FROM {self.schema}.kb_source s JOIN {self.schema}.kb_index k ON k.id=s.index_id
                WHERE s.trigger_config_id=$1 AND ($2::uuid IS NULL OR k.tenant_id=$2) AND k.organization_id=$3)''',
                    command.externalId,
                    command.tenant_id,
                    command.organization_id,
                )
            )
        }

    async def _source_schedules(
        self, connection: AsyncPostgresConnection, command: models.SourceSchedules
    ) -> list[dict[str, Any]]:
        rows = await connection.fetch(
            f'''SELECT sensor.*,k.id AS index_id,s.id AS source_id,k.workspace_id,
                COALESCE((SELECT jsonb_agg(tick ORDER BY tick.started_at DESC,tick.id DESC) FROM
                    (SELECT t.* FROM {self.schema}.sensor_tick t WHERE t.sensor_id=sensor.id
                     ORDER BY t.started_at DESC,t.id DESC LIMIT 10) tick),'[]'::jsonb) AS executions
                FROM {self.schema}.sensor sensor
                JOIN {self.schema}.kb_source s ON s.trigger_config_id=sensor.external_id
                JOIN {self.schema}.kb_index k ON k.id=s.index_id
                WHERE k.organization_id=$1 AND k.workspace_id=$2 AND ($3::uuid IS NULL OR k.tenant_id=$3)
                ORDER BY sensor.created_at DESC,sensor.id''',
            command.organization_id,
            command.workspaceId,
            command.tenant_id,
        )
        return [
            {
                **camel_row(row),
                'executions': [camel_row(tick) for tick in row['executions']],
            }
            for row in rows
        ]

    async def _submission(
        self, connection: AsyncPostgresConnection, command: models.SubmissionScope
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            f'SELECT * FROM {self.schema}.submissions WHERE id=$1 AND tenant_id=$2 FOR UPDATE',
            command.submissionId,
            command.tenant_id,
        )
        if row is None:
            raise CatalogError('Submission not found', 404)
        return row

    async def _withdraw_submission_items(
        self, connection: AsyncPostgresConnection, command: models.WithdrawSubmission
    ) -> list[dict[str, Any]]:
        return json_safe(
            await connection.fetch(
                f'''UPDATE {self.schema}.kb_item_state k SET state='removed',
                cleanup_state=CASE WHEN state='removed' THEN cleanup_state ELSE 'pending' END,updated_at=clock_timestamp()
                WHERE tenant_id=$1 AND owner_type='submission_document' AND EXISTS(SELECT 1 FROM {self.schema}.submission_documents d
                  WHERE d.id::text=k.owner_id AND d.submission_id=$2 AND ($3::uuid IS NULL OR d.id=$3))
                AND NOT(k.index_id=ANY($4::uuid[])) AND (k.state <> 'removed' OR k.cleanup_state <> 'complete') RETURNING *''',
                command.tenant_id,
                command.submissionId,
                command.documentId,
                command.retainedIndexIds,
            )
        )

    async def _withdraw_submission(
        self, connection: AsyncPostgresConnection, command: models.WithdrawSubmission
    ) -> list[dict[str, Any]]:
        await self._submission(connection, command)
        if command.documentId:
            document = await connection.fetchval(
                f'''UPDATE {self.schema}.submission_documents SET indexing_status='removing',indexing_error='Indexed removal is pending',
                    updated_at=clock_timestamp() WHERE id=$1 AND submission_id=$2 RETURNING id''',
                command.documentId,
                command.submissionId,
            )
            if document is None:
                raise CatalogError('Submission document not found', 404)
        elif not command.retainedIndexIds:
            await connection.execute(
                f'''UPDATE {self.schema}.submissions SET enable_semantic_search=false,kb_index_id=NULL,
                    updated_at=clock_timestamp() WHERE id=$1''',
                command.submissionId,
            )
            await connection.execute(
                f'DELETE FROM {self.schema}.submission_kb_indexes WHERE submission_id=$1',
                command.submissionId,
            )
        return await self._withdraw_submission_items(connection, command)

    async def _submission_destinations(
        self, connection: AsyncPostgresConnection, submission: dict[str, Any]
    ) -> list[UUID]:
        if not submission['enable_semantic_search']:
            return []
        rows = await connection.fetch(
            f'SELECT kb_index_id FROM {self.schema}.submission_kb_indexes WHERE submission_id=$1',
            submission['id'],
        )
        return sorted(
            {row['kb_index_id'] for row in rows}
            | ({submission['kb_index_id']} if submission.get('kb_index_id') else set()),
            key=str,
        )

    async def _register_submission(
        self,
        connection: AsyncPostgresConnection,
        command: models.RegisterSubmission,
        submission: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if (
            'expectedQueryPlanTemplateId' in command.model_fields_set
            and command.expectedQueryPlanTemplateId
            != submission.get('query_plan_template_id')
        ):
            raise CatalogError(
                'Submission processing workflow changed. Reload before registering documents.'
            )
        destinations = await self._submission_destinations(connection, submission)
        if command.indexIds is not None:
            if not set(command.indexIds).issubset(destinations):
                raise CatalogError(
                    'Submission destination is no longer authorized', 403
                )
            destinations = sorted(set(command.indexIds), key=str)
        if not destinations:
            return []
        planner, fabrics = await self._upstream_settings(
            connection, command, submission
        )
        documents = await connection.fetch(
            f'''SELECT * FROM {self.schema}.submission_documents WHERE submission_id=$1
                AND indexing_status IS DISTINCT FROM 'removing' AND ($2::uuid[] IS NULL OR id=ANY($2)) ORDER BY id''',
            submission['id'],
            command.documentIds,
        )
        if command.documentIds is not None and {row['id'] for row in documents} != set(
            command.documentIds
        ):
            raise CatalogError('Submission document is unavailable', 403)
        results = []
        for index_id in destinations:
            index = await self._index(connection, command, index_id, lock=True)
            source = await connection.fetchrow(
                f"SELECT * FROM {self.schema}.kb_source WHERE index_id=$1 AND source_type='submission' AND metadata->>'submissionId'=$2 FOR UPDATE",
                index_id,
                str(submission['id']),
            )
            if source is None:
                source = await connection.fetchrow(
                    f'''INSERT INTO {self.schema}.kb_source(index_id,name,source_type,status,metadata)
                        VALUES($1,$2,'submission','active',$3::jsonb) RETURNING *''',
                    index_id,
                    f"Submission: {submission['name']}",
                    json.dumps({'submissionId': str(submission['id'])}),
                )
            files = [
                models.FileAdmission(
                    sourceId=source['id'],
                    name=document['file_name'],
                    s3Key=document['storage_key'],
                    size=document['file_size'],
                    contentType=document['content_type'],
                    origin=models.Origin(
                        kind='submission',
                        submissionId=submission['id'],
                        submissionDocumentId=document['id'],
                    ),
                    upstreamPlanner=planner,
                    upstreamFabrics=fabrics,
                )
                for document in documents
            ]
            if files:
                admission = models.AdmitDocuments(
                    **{
                        key: value
                        for key, value in command.model_dump().items()
                        if key in ('organization_id', 'tenant_id', 'actor_id')
                    },
                    indexId=index_id,
                    bucket=command.bucket,
                    files=files,
                )
                results.extend(await self._admit_documents(connection, admission))
        return results

    async def _register_submission_documents(
        self, connection: AsyncPostgresConnection, command: models.RegisterSubmission
    ) -> list[dict[str, Any]]:
        submission = await self._submission(connection, command)
        return await self._register_submission(connection, command, submission)

    async def _set_submission_destinations(
        self,
        connection: AsyncPostgresConnection,
        command: models.SubmissionDestinations,
    ) -> dict[str, Any]:
        submission = await self._submission(connection, command)
        if submission['status'] != 'open':
            raise CatalogError('Can only update open submissions')
        if command.expectedUpdatedAt and int(
            submission['updated_at'].timestamp() * 1000
        ) != int(command.expectedUpdatedAt.timestamp() * 1000):
            raise CatalogError(
                'Submission configuration changed. Reload before saving.'
            )
        if (
            'expectedQueryPlanTemplateId' in command.model_fields_set
            and command.expectedQueryPlanTemplateId
            != submission.get('query_plan_template_id')
        ):
            raise CatalogError(
                'Submission processing workflow changed. Reload before saving.'
            )
        if command.data:
            patch = command.data.model_dump(exclude_unset=True)
            if 'description' in patch:
                await connection.execute(
                    f'UPDATE {self.schema}.submissions SET description=$2 WHERE id=$1',
                    submission['id'],
                    patch['description'],
                )
            if 'name' in patch:
                await connection.execute(
                    f'UPDATE {self.schema}.submissions SET name=$2 WHERE id=$1',
                    submission['id'],
                    patch['name'],
                )
            if 'queryPlanTemplateId' in patch:
                await connection.execute(
                    f'UPDATE {self.schema}.submissions SET query_plan_template_id=$2 WHERE id=$1',
                    submission['id'],
                    patch['queryPlanTemplateId'],
                )
        destinations = sorted(set(command.kbIndexIds), key=str)
        for index_id in destinations:
            await self._index(connection, command, index_id, lock=True)
        await connection.execute(
            f'DELETE FROM {self.schema}.submission_kb_indexes WHERE submission_id=$1',
            submission['id'],
        )
        for index_id in destinations:
            await connection.execute(
                f'INSERT INTO {self.schema}.submission_kb_indexes(id,submission_id,kb_index_id) VALUES($1,$2,$3)',
                uuid4(),
                submission['id'],
                index_id,
            )
        submission = await connection.fetchrow(
            f'''UPDATE {self.schema}.submissions SET kb_index_id=$2,enable_semantic_search=$3,updated_at=clock_timestamp()
                WHERE id=$1 RETURNING *''',
            submission['id'],
            destinations[0] if destinations else None,
            command.enableSemanticSearch,
        )
        withdrawn = await self._withdraw_submission_items(
            connection,
            models.WithdrawSubmission(
                organization_id=command.organization_id,
                actor_id=command.actor_id,
                tenant_id=command.tenant_id,
                submissionId=submission['id'],
                retainedIndexIds=destinations if command.enableSemanticSearch else [],
            ),
        )
        registered = []
        if command.registerExistingDocuments:
            registered = await self._register_submission(
                connection,
                models.RegisterSubmission(
                    organization_id=command.organization_id,
                    actor_id=command.actor_id,
                    tenant_id=command.tenant_id,
                    submissionId=submission['id'],
                    bucket=command.bucket,
                    upstreamPlanner=command.upstreamPlanner,
                    upstreamFabrics=command.upstreamFabrics,
                ),
                submission,
            )
        return {
            'submission': camel_row(submission),
            'documents': registered,
            'items': withdrawn,
        }

    async def _admit_submission_documents(
        self, connection: AsyncPostgresConnection, command: models.AdmitSubmission
    ) -> dict[str, Any]:
        submission = await self._submission(connection, command)
        if submission['status'] != 'open':
            raise CatalogError('Submission is no longer open')
        documents = []
        added = 0
        for file in command.files:
            prefix = f"tenants/{command.tenant_id}/submissions/{submission['id']}/"
            borrowed = not file.storageKey.startswith(prefix)
            if (
                file.storageKey.endswith('/')
                or borrowed
                and not file.storageImportAuthorized
            ):
                raise CatalogError(
                    'Submission storage ownership cannot be authorized', 403
                )
            if borrowed:
                parts = file.storageKey.split('/')
                if (
                    parts[0] == 'tenants' and parts[1:2] != [str(command.tenant_id)]
                ) or parts[0] == 'workspaces':
                    raise CatalogError(
                        'Storage object is outside the authorized scope', 403
                    )
                if parts[0] == 'kb-indexes':
                    try:
                        owner_id = UUID(parts[1])
                    except (ValueError, IndexError) as error:
                        raise CatalogError(
                            'Storage object owner cannot be authorized', 403
                        ) from error
                    owner = await self._index(
                        connection, command, owner_id, removing=True
                    )
                    if owner['workspace_id'] is not None:
                        raise CatalogError(
                            'Storage object owner cannot be authorized', 403
                        )
            previous = (
                await connection.fetchrow(
                    f'SELECT * FROM {self.schema}.submission_documents WHERE id=$1',
                    file.id,
                )
                if file.id
                else None
            )
            if previous:
                if (
                    previous['submission_id'],
                    previous['storage_key'],
                    previous['file_name'],
                    previous['file_size'],
                    previous['content_type'],
                ) != (
                    submission['id'],
                    file.storageKey,
                    file.fileName,
                    file.fileSize,
                    file.contentType,
                ):
                    raise CatalogError('Submission document identity changed')
                document = previous
            else:
                document = await connection.fetchrow(
                    f'''INSERT INTO {self.schema}.submission_documents(id,submission_id,file_name,file_size,content_type,storage_key,status)
                        VALUES($1,$2,$3,$4,$5,$6,'pending') RETURNING *''',
                    file.id or uuid4(),
                    submission['id'],
                    file.fileName,
                    file.fileSize,
                    file.contentType,
                    file.storageKey,
                )
                added += 1
            documents.append(document)
        if added:
            await connection.execute(
                f'UPDATE {self.schema}.submissions SET total_documents=total_documents+$2,updated_at=clock_timestamp() WHERE id=$1',
                submission['id'],
                added,
            )
        memberships = await self._register_submission(
            connection,
            models.RegisterSubmission(
                organization_id=command.organization_id,
                actor_id=command.actor_id,
                tenant_id=command.tenant_id,
                submissionId=submission['id'],
                documentIds=[row['id'] for row in documents],
                bucket=command.bucket,
                upstreamPlanner=command.upstreamPlanner,
                upstreamFabrics=command.upstreamFabrics,
                **(
                    {'expectedQueryPlanTemplateId': command.expectedQueryPlanTemplateId}
                    if 'expectedQueryPlanTemplateId' in command.model_fields_set
                    else {}
                ),
            ),
            submission,
        )
        return {
            'documents': [camel_row(row) for row in documents],
            'kbDocuments': memberships,
        }

    async def _upstream_settings(
        self,
        connection: AsyncPostgresConnection,
        command: models.RegisterSubmission,
        submission: dict[str, Any],
    ) -> tuple[str | None, list[str]]:
        template_id = submission.get('query_plan_template_id')
        if template_id is None:
            if command.upstreamPlanner:
                raise CatalogError('Submission processing workflow changed')
            return None, []
        template = await connection.fetchrow(
            f'''SELECT p.name,p.published_to_groups FROM {self.platform_schema}.query_plan_templates p
               JOIN {self.platform_schema}.workspaces w ON w.id=p.workspace_id
               WHERE p.id=$1 AND w.organization_id=$2''',
            template_id,
            command.organization_id,
        )
        if template is None:
            raise CatalogError('Submission processing workflow is unavailable', 403)
        fabrics = list(template['published_to_groups'] or [])
        if command.upstreamPlanner and (
            command.upstreamPlanner != template['name']
            or set(command.upstreamFabrics) != set(fabrics)
        ):
            raise CatalogError('Submission processing workflow settings changed')
        return template['name'], fabrics
