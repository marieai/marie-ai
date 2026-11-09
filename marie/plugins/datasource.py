"""Resolve a saved datasource binding for a normal scheduler execution."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from uuid import UUID

from marie.kb.item_state import scheduler_execution
from marie.plugins.embedded import inspect_archive
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


async def datasource_job(
    connection: Any, execution: dict[str, Any], *, lock: bool = False
) -> dict[str, Any]:
    job = await connection.fetchrow(
        '''SELECT data,dag_id,created_on FROM marie_scheduler.job
           WHERE id=$1 AND state='active' AND run_attempt_id=$2 AND run_owner=$3
             AND run_lease_expires_at > clock_timestamp()'''
        + (' FOR UPDATE' if lock else ''),
        UUID(execution['job_id']),
        UUID(execution['run_attempt_id']),
        execution['run_owner'],
    )
    if (
        job is None
        or job['data'].get('metadata', {}).get('planner') != 'kb_datasource_sync'
    ):
        raise ValueError('Datasource requires a current scheduler execution')
    return job


async def load_datasource_binding(
    pool: AsyncPostgresConnectionPool,
    parameters: dict[str, Any],
    package_root: Path,
) -> dict[str, Any]:
    """Resolve installed configuration and credential references once for this run."""
    execution = scheduler_execution(parameters)
    async with pool.acquire() as connection:
        job = await datasource_job(connection, execution)
        metadata = job['data']['metadata']
        binding = await connection.fetchrow(
            '''SELECT s.id AS source_id,s.dataset_inputs,s.extension_install_id,s.datasource_ref,
                      s.created_by,k.id AS index_id,k.tenant_id,k.workspace_id,k.metadata AS index_metadata,
                      k.chunk_size,k.chunk_overlap,k.segmentation_mode,k.multimodal_enabled,k.embedding_model,k.embedding_dim,
                      t.organization_id,p.id AS package_id,p.package_ref,p.source_digest,p.trust_level,
                      p.manifest_snapshot,d.output_schema,d.declaration,
                      provider.id AS provider_id
               FROM marie_studio.kb_source s
               JOIN marie_studio.kb_index k ON k.id=s.index_id
               JOIN marie_studio.tenant t ON t.id=k.tenant_id
               JOIN marie_studio.extension_installs i ON i.id=s.extension_install_id
               JOIN marie_studio.extension_packages p ON p.id=i.package_id
               JOIN marie_studio.extension_datasources d ON d.package_id=p.id AND d.datasource_ref=s.datasource_ref
               LEFT JOIN marie_studio.extension_providers provider ON provider.package_id=p.id
                    AND provider.provider_ref=d.declaration->>'providerRef' AND provider.type='datasource'
               WHERE s.id=$1 AND k.id=$2 AND t.id=$3 AND t.organization_id=$4 AND k.workspace_id=$5
                 AND s.source_type='datasource' AND s.status <> 'disabled' AND t.status='active'
                 AND NOT COALESCE((k.metadata->>'removalPending')::boolean,false)
                 AND i.organization_id=t.organization_id AND (i.workspace_id IS NULL OR i.workspace_id=k.workspace_id)
                 AND i.enabled AND i.status='enabled' AND p.trust_level <> 'blocked' ''',
            UUID(metadata['source_id']),
            UUID(metadata['index_id']),
            UUID(metadata['tenant_id']),
            UUID(metadata['organization_id']),
            UUID(metadata['workspace_id']),
        )
        if binding is None:
            raise ValueError('Saved datasource binding is unavailable')
        if binding['declaration'].get('indexingAdapter') != 'document':
            raise ValueError(
                'Datasource requires a compatible indexingAdapter; document is supported'
            )
        if binding['declaration'].get('providerRef') and not binding['provider_id']:
            raise ValueError('Datasource credential provider is unavailable')
        user_id = metadata.get('user_id') or str(binding['created_by'] or '')
        if not user_id:
            raise ValueError('Datasource invocation requires its principal')
        if not await connection.fetchval(
            '''SELECT EXISTS(SELECT 1 FROM marie_studio.organization_membership m
               JOIN marie_studio."user" u ON u.id=m.user_id
               WHERE m.user_id=$1 AND m.organization_id=$2 AND m.status='active' AND u.is_active)''',
            UUID(user_id),
            binding['organization_id'],
        ):
            raise ValueError(
                'Datasource principal is not an active organization member'
            )
        credentials = await connection.fetch(
            '''SELECT r.name,r.required,b.id AS binding_id,b.secret_ref
               FROM marie_studio.extension_credential_requirements r
               LEFT JOIN marie_studio.extension_credential_bindings b ON b.requirement_id=r.id
                 AND b.organization_id=$2 AND b.install_id=$3 AND b.status='active'
                 AND (b.workspace_id IS NULL OR b.workspace_id=$4)
                 AND (b.owner_type NOT IN ('user','owner','owner_only') OR b.owner_id=$5)
               WHERE r.provider_id=$1''',
            binding['provider_id'],
            binding['organization_id'],
            binding['extension_install_id'],
            binding['workspace_id'],
            UUID(user_id),
        )
        if len({row['name'] for row in credentials}) != len(credentials):
            raise ValueError('Datasource has ambiguous credential bindings')
        if any(row['required'] and not row['secret_ref'] for row in credentials):
            raise ValueError('Datasource credential binding is unavailable')
        workflow = await connection.fetchrow(
            '''SELECT run_params FROM marie_scheduler.resource_workflow_binding
               WHERE resource_type='kb_index' AND resource_id=$1 AND tenant_id=$2''',
            binding['index_id'],
            binding['tenant_id'],
        )
    digest = binding['source_digest']
    if not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Datasource package archive digest is invalid')
    archive = package_root / str(binding['organization_id']) / f'{digest}.zip'
    package_ref, package_digest = await asyncio.to_thread(inspect_archive, str(archive))
    if package_ref != binding['package_ref']:
        raise ValueError('Datasource package archive does not match its installation')
    return {
        **binding,
        'user_id': user_id,
        'package_digest': package_digest,
        'credentials': credentials,
        'execution': execution,
        'dag_id': str(job['dag_id']),
        'run_params': workflow['run_params'] if workflow else None,
        'fabric_group_id': metadata.get('fabric_group_id'),
    }
