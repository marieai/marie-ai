"""Map a datasource's declared document output to existing KB items."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from marie.kb.content import document_content_reference
from marie.plugins.datasource import datasource_job
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


def datasource_items(records: list[Any]) -> list[dict[str, Any]]:
    """Validate the document adapter without flattening custom domain records."""
    seen = set()
    items = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get('id'), str)
            or not record['id'].strip()
        ):
            raise ValueError('Document adapter requires a stable record id')
        if record['id'] in seen:
            raise ValueError('Document adapter received duplicate record ids')
        seen.add(record['id'])
        if record.get('deleted') is True:
            items.append({'id': record['id'], 'deleted': True})
            continue
        uri = record.get('uri')
        if not isinstance(uri, str) or not uri.startswith('s3://'):
            raise ValueError('Document adapter requires a saved S3 content reference')
        item = {'id': record['id'], 'uri': uri}
        if record.get('input_ref') is not None:
            item['input_ref'] = document_content_reference(record['input_ref'])
            if not item['input_ref'].get('sha256'):
                raise ValueError('Datasource saved content requires its digest')
            if item['input_ref']['kind'] == 'text' and item['input_ref']['uri'] != uri:
                raise ValueError('Document and text artifact references differ')
        else:
            digest = record.get('sha256')
            if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
                raise ValueError(
                    'Datasource original document requires its SHA-256 digest'
                )
            item['sha256'] = digest
        items.append(item)
    return items


async def admit_datasource_items(
    pool: AsyncPostgresConnectionPool, binding: dict[str, Any], records: list[Any]
) -> int:
    """Admit delivered output atomically; existing item sensors schedule indexing."""
    items = datasource_items(records)
    for item in items:
        if (
            not item.get('deleted')
            and urlsplit(item['uri']).netloc != binding['bucket_name']
        ):
            raise ValueError(
                'Datasource artifact bucket differs from the existing storage connection'
            )
    run_params = (
        binding.get('run_params')
        if binding.get('run_params') is not None
        else {
            'chunk_size': binding['chunk_size'],
            'chunk_overlap': binding['chunk_overlap'],
            'segmentation_mode': binding['segmentation_mode'],
            'multimodal': binding['multimodal_enabled'],
        }
    )
    run_params = dict(run_params)
    run_params.update(
        chunk_size=binding['chunk_size'],
        chunk_overlap=binding['chunk_overlap'],
        segmentation_mode=binding['segmentation_mode'],
        multimodal=binding['multimodal_enabled'],
        embedding_model=binding['embedding_model'],
        embedding_dim=binding['embedding_dim'],
    )
    if binding.get('fabric_group_id'):
        run_params['fabric_group_id'] = binding['fabric_group_id']
    backend = (binding['index_metadata'] or {}).get('retrievalBackend', 'pgvector')
    if backend not in ('pgvector', 'postgresql_text'):
        raise ValueError(
            'Datasource document adapter requires a compatible retrieval backend'
        )
    store = 'kb_text_section' if backend == 'postgresql_text' else 'kb_vectors'
    if backend == 'postgresql_text':
        run_params.update(retrieval_backend=backend, multimodal=False)
    async with pool.acquire() as connection:
        async with connection.transaction():
            job = await datasource_job(connection, binding['execution'], lock=True)
            metadata = job['data']['metadata']
            if any(
                str(binding[key]) != metadata.get(key)
                for key in ('tenant_id', 'index_id', 'source_id')
            ):
                raise ValueError('Datasource scheduler scope changed')
            source = await connection.fetchrow(
                '''SELECT s.dataset_inputs,s.extension_install_id,s.datasource_ref,k.metadata,
                          k.chunk_size,k.chunk_overlap,k.segmentation_mode,k.multimodal_enabled,k.embedding_model,k.embedding_dim
                        ,s.metadata AS source_metadata
                   FROM marie_studio.kb_index k JOIN marie_studio.kb_source s ON s.index_id=k.id
                   WHERE k.id=$1 AND k.tenant_id=$2 AND s.id=$3 AND s.status <> 'disabled'
                     AND NOT COALESCE((k.metadata->>'removalPending')::boolean,false)
                   FOR UPDATE OF k,s''',
                binding['index_id'],
                binding['tenant_id'],
                binding['source_id'],
            )
            if (
                source is None
                or any(
                    source[key] != binding[key]
                    for key in (
                        'dataset_inputs',
                        'extension_install_id',
                        'datasource_ref',
                    )
                )
                or any(
                    source[key] != binding[key]
                    for key in (
                        'chunk_size',
                        'chunk_overlap',
                        'segmentation_mode',
                        'multimodal_enabled',
                        'embedding_model',
                        'embedding_dim',
                    )
                )
                or (source['metadata'] or {}).get('retrievalBackend', 'pgvector')
                != (binding['index_metadata'] or {}).get('retrievalBackend', 'pgvector')
            ):
                raise ValueError('Datasource binding changed during execution')
            workflow = await connection.fetchrow(
                '''SELECT run_params FROM marie_scheduler.resource_workflow_binding
                   WHERE resource_type='kb_index' AND resource_id=$1 AND tenant_id=$2''',
                binding['index_id'],
                binding['tenant_id'],
            )
            current_params = (workflow['run_params'] or {}) if workflow else {}
            previous_params = binding.get('run_params') or {}
            if not binding.get('fabric_group_id') and current_params.get(
                'fabric_group_id'
            ) != previous_params.get('fabric_group_id'):
                raise ValueError(
                    'Datasource processing settings changed during execution'
                )
            previous_dag = (source['source_metadata'] or {}).get('syncDagId')
            if previous_dag and await connection.fetchval(
                '''SELECT (SELECT min(created_on) FROM marie_scheduler.job WHERE dag_id=$1) >
                          (SELECT min(created_on) FROM marie_scheduler.job WHERE dag_id=$2)''',
                UUID(previous_dag),
                job['dag_id'],
            ):
                raise ValueError(
                    'Datasource sync was superseded by a newer completed run'
                )
            for item in items:
                identity = (
                    binding['tenant_id'],
                    binding['index_id'],
                    str(binding['source_id']),
                    item['id'],
                )
                if item.get('deleted'):
                    await connection.execute(
                        '''UPDATE marie_scheduler.kb_item_state SET state='removed',cleanup_state='pending',
                           updated_at=clock_timestamp() WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3
                           AND item_id=$4 AND owner_type='datasource' AND owner_id=$3 AND state <> 'removed' ''',
                        *identity,
                    )
                    continue
                reference = item.get('input_ref')
                path = urlsplit(item['uri']).path
                if not path.startswith(f"/tenants/{binding['tenant_id']}/") or any(
                    part in ('.', '..') for part in path.split('/')
                ):
                    raise ValueError('Datasource artifact is outside its tenant scope')
                if (
                    reference
                    and reference['kind'] == 'extraction'
                    and not await connection.fetchval(
                        '''SELECT EXISTS(SELECT 1 FROM marie_scheduler.dag d JOIN marie_scheduler.job j ON j.dag_id=d.id
                       WHERE d.project_id=$1 AND d.state='completed' AND d.ref_id=$2 AND d.ref_type=$3
                         AND j.data->'metadata'->>'uri'=$4)''',
                        str(binding['tenant_id']),
                        reference['ref_id'],
                        reference['ref_type'],
                        item['uri'],
                    )
                ):
                    raise ValueError(
                        'Saved extraction is not owned by the delivered document'
                    )
                if backend == 'postgresql_text' and reference is None:
                    raise ValueError(
                        'Text indexing requires saved text or extraction output'
                    )
                if reference is not None:
                    reference = {**reference, 'processing_workflow': 'kb_index_content'}
                item_params = (
                    {**run_params, 'source_digest': item['sha256']}
                    if 'sha256' in item
                    else run_params
                )
                current = await connection.fetchrow(
                    '''SELECT * FROM marie_scheduler.kb_item_state
                       WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4 FOR UPDATE''',
                    *identity,
                )
                if current:
                    if (
                        current['owner_type'],
                        current['owner_id'],
                        current['backend'],
                        current['store_name'],
                    ) != (
                        'datasource',
                        identity[2],
                        backend,
                        store,
                    ):
                        raise ValueError('Datasource item adapter ownership changed')
                    if (
                        current['state'] == 'removed'
                        and current['cleanup_state'] != 'complete'
                    ):
                        raise ValueError(
                            'Datasource item cleanup must complete before reintroduction'
                        )
                    previous_params = {
                        key: value
                        for key, value in current['run_params'].items()
                        if key != 'fabric_group_id'
                    }
                    content_params = {
                        key: value
                        for key, value in item_params.items()
                        if key != 'fabric_group_id'
                    }
                    for key in ('embedding_model', 'embedding_dim'):
                        if key in content_params:
                            previous_params.setdefault(key, content_params[key])
                    changed = current['state'] in ('error', 'removed') or (
                        current['uri'],
                        current['input_ref'],
                        previous_params,
                    ) != (item['uri'], reference, content_params)
                    if not changed:
                        continue
                    await connection.execute(
                        '''UPDATE marie_scheduler.kb_item_state SET indexing_dag_id=$5,state='pending',
                           admitted_at=NULL,error=NULL,uri=$6,run_params=$7::jsonb,input_ref=$8::jsonb,
                           indexed_dag_id=CASE WHEN state='removed' THEN NULL ELSE indexed_dag_id END,
                           record_count=CASE WHEN state='removed' THEN 0 ELSE record_count END,
                           cleanup_state='none',updated_at=clock_timestamp()
                           WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4''',
                        *identity,
                        uuid4(),
                        item['uri'],
                        json.dumps(item_params),
                        json.dumps(reference) if reference is not None else None,
                    )
                else:
                    await connection.execute(
                        '''INSERT INTO marie_scheduler.kb_item_state
                           (tenant_id,index_id,source_id,item_id,indexing_dag_id,owner_type,owner_id,uri,
                            run_params,backend,store_name,input_ref)
                           VALUES ($1,$2,$3,$4,$5,'datasource',$3,$6,$7::jsonb,$8,$9,$10::jsonb)''',
                        *identity,
                        uuid4(),
                        item['uri'],
                        json.dumps(item_params),
                        backend,
                        store,
                        json.dumps(reference) if reference is not None else None,
                    )
            removed = await connection.fetch(
                '''SELECT source_id,item_id,indexing_dag_id FROM marie_scheduler.kb_item_state
                   WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3
                     AND owner_type='datasource' AND owner_id=$3
                     AND state='removed' AND cleanup_state <> 'complete'
                     AND backend=$4 AND store_name=$5''',
                binding['tenant_id'],
                binding['index_id'],
                str(binding['source_id']),
                backend,
                store,
            )
            await datasource_job(connection, binding['execution'])
            await connection.execute(
                '''UPDATE marie_studio.kb_source SET status='ready',error_message=NULL,
                   metadata=COALESCE(metadata,'{}'::jsonb) || jsonb_build_object('syncDagId',$2::text),
                   updated_at=clock_timestamp() WHERE id=$1''',
                binding['source_id'],
                binding['dag_id'],
            )
    if removed:
        tenant_id, index_id = str(binding['tenant_id']), str(binding['index_id'])
        if backend == 'postgresql_text':
            from marie.kb.text import TextIndexStore

            await TextIndexStore(pool)._cleanup_removed(tenant_id, index_id, removed)
        else:
            from marie.vector_stores.pgvector.pgvector import cleanup_removed_items

            await cleanup_removed_items(pool, tenant_id, index_id, removed)
    return len(items)
