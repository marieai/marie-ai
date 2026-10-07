"""Versioned operator endpoint policy in existing scheduler metadata."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from marie.engine.llm_queue.admission_policy import AdmissionPolicy
from marie.engine.llm_queue.endpoint import (
    RegisteredEndpoint,
    RegisteredEndpointGroup,
    RegisteredReplica,
)
from marie.engine.llm_queue.request_dispatcher import DispatchLane
from marie.engine.llm_queue.store import StoreLimits


def validate_dispatch_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate policy before resolving any named credential bindings."""
    if not isinstance(policy, dict) or set(policy) - {
        'endpoints',
        'endpoint_groups',
        'lanes',
        'limits',
        'policy',
        'total_concurrent_dispatch',
    }:
        raise ValueError('Invalid dispatch policy')
    limits = StoreLimits(**policy.get('limits', {}))
    scheduling = policy.get('policy', 'drr')
    total = policy.get('total_concurrent_dispatch', limits.max_execution_items)
    if (
        scheduling not in {'fifo', 'drr'}
        or type(total) is not int
        or not 1 <= total <= limits.max_execution_items
    ):
        raise ValueError('Invalid dispatch scheduling policy')
    endpoints = policy.get('endpoints', [])
    endpoint_groups = policy.get('endpoint_groups', [])
    lanes = policy.get('lanes', [])
    if (
        not isinstance(endpoints, list)
        or not isinstance(endpoint_groups, list)
        or not isinstance(lanes, list)
        or bool(endpoints) == bool(endpoint_groups)
        or not 0 < len(lanes) <= limits.max_routes
    ):
        raise ValueError('Invalid dispatch policy size')
    registered_groups: dict[str, RegisteredEndpointGroup] = {}
    registered_replicas: dict[str, RegisteredReplica] = {}
    if endpoints:
        for endpoint in endpoints:
            if not isinstance(endpoint, dict) or 'api_key' in endpoint:
                raise ValueError('Inline endpoint credentials are forbidden')
            values = dict(endpoint)
            binding = values.pop('credential_env', None)
            if binding is not None and (
                not isinstance(binding, str)
                or not re.fullmatch(r'[A-Z_][A-Z0-9_]{0,127}', binding)
            ):
                raise ValueError('Invalid endpoint credential binding')
            instance = RegisteredEndpoint(**values)
            replica = RegisteredReplica(
                replica_id=instance.endpoint_id,
                base_url=instance.base_url,
                credential_env=binding,
                allow_private=instance.allow_private,
                allow_loopback=instance.allow_loopback,
                execution_limit=instance.execution_limit,
                execution_bytes=instance.execution_bytes,
                call_timeout_seconds=instance.call_timeout_seconds,
                max_response_bytes=instance.max_response_bytes,
                retry_429=instance.retry_429,
                enabled=instance.enabled,
            )
            if (
                instance.endpoint_id in registered_groups
                or replica.replica_id in registered_replicas
            ):
                raise ValueError('Duplicate endpoint identity')
            registered_groups[instance.endpoint_id] = RegisteredEndpointGroup(
                instance.endpoint_id, 'r1', (replica,)
            )
            registered_replicas[replica.replica_id] = replica
    else:
        for record in endpoint_groups:
            if not isinstance(record, dict) or set(record) - {
                'group_id',
                'revision',
                'replicas',
            }:
                raise ValueError('Invalid endpoint group')
            raw_replicas = record.get('replicas')
            if not isinstance(raw_replicas, list):
                raise ValueError('Invalid endpoint group replicas')
            replicas = []
            for values in raw_replicas:
                if not isinstance(values, dict) or 'api_key' in values:
                    raise ValueError('Inline replica credentials are forbidden')
                replica = RegisteredReplica(**values)
                if replica.replica_id in registered_replicas:
                    raise ValueError('Duplicate replica identity')
                registered_replicas[replica.replica_id] = replica
                replicas.append(replica)
            group = RegisteredEndpointGroup(
                group_id=record.get('group_id'),
                revision=record.get('revision'),
                replicas=tuple(replicas),
            )
            if group.group_id in registered_groups:
                raise ValueError('Duplicate endpoint group identity')
            registered_groups[group.group_id] = group
    if not 0 < len(registered_replicas) <= limits.max_endpoints:
        raise ValueError('Invalid replica policy size')
    targets = [replica.target_fingerprint() for replica in registered_replicas.values()]
    if len(targets) != len(set(targets)):
        raise ValueError('Replica targets must be unique')
    for replica in registered_replicas.values():
        if (
            replica.execution_limit > limits.max_execution_items
            or replica.execution_bytes > limits.max_execution_bytes
        ):
            raise ValueError('Replica exceeds immutable fabric limits')
    seen = set()
    normalized_lanes = []
    for row in lanes:
        lane = DispatchLane(**row)
        group = registered_groups.get(lane.endpoint_group_id)
        if (
            group is None
            or lane.pool_id in seen
            or (endpoint_groups and lane.revision != group.revision)
        ):
            raise ValueError('Invalid lane endpoint binding')
        seen.add(lane.pool_id)
        group_limit = sum(replica.execution_limit for replica in group.replicas)
        group_bytes = sum(replica.execution_bytes for replica in group.replicas)
        if (
            not 0 < lane.execution_limit <= group_limit
            or not 0 < lane.execution_bytes <= group_bytes
        ):
            raise ValueError('Lane exceeds registered group limits')
        normalized = dict(row)
        normalized['endpoint_group_id'] = lane.endpoint_group_id
        if endpoint_groups:
            normalized.pop('endpoint_id', None)
        normalized_lanes.append(normalized)
    if (
        sum(row.get('min_concurrent', 0) for row in lanes if row.get('enabled', True))
        > total
    ):
        raise ValueError('Protected slots exceed dispatch capacity')
    return {
        'endpoints': endpoints,
        'endpoint_groups': endpoint_groups,
        'lanes': normalized_lanes,
        'limits': asdict(limits),
        'policy': scheduling,
        'total_concurrent_dispatch': total,
    }


def persisted_dispatch_policy(data: dict[str, Any]) -> dict[str, Any]:
    """Read schema version 1; ordinary pool URLs never become trusted endpoints."""
    fabric = (data.get('metadata') or {}).get('llm_dispatch')
    if (
        not isinstance(fabric, dict)
        or type(fabric.get('schema_version')) is not int
        or fabric['schema_version'] != 1
        or set(fabric) - {'schema_version', 'endpoints', 'endpoint_groups', 'limits'}
    ):
        raise ValueError('Invalid persisted dispatch fabric policy')
    lanes = []
    for row in data.get('lanes', []):
        metadata = (row.get('metadata') or {}).get('llm_dispatch')
        if (
            not isinstance(metadata, dict)
            or type(metadata.get('schema_version')) is not int
            or metadata['schema_version'] != 1
            or set(metadata)
            - {
                'schema_version',
                'endpoint_id',
                'endpoint_group_id',
                'revision',
                'execution_bytes',
            }
            or bool(metadata.get('endpoint_id'))
            == bool(metadata.get('endpoint_group_id'))
        ):
            raise ValueError('Invalid persisted dispatch pool policy')
        lane = {
            'pool_id': row['pool_id'],
            'enabled': row['enabled'],
            'endpoint_group_id': metadata.get('endpoint_group_id')
            or metadata['endpoint_id'],
            'revision': metadata['revision'],
            'execution_limit': row['max_concurrent'],
            'execution_bytes': metadata['execution_bytes'],
            'quantum': row.get('quantum', 1),
            'min_concurrent': row.get('min_concurrent', 0),
            'max_burst_per_visit': row.get('max_burst_per_visit'),
        }
        if metadata.get('endpoint_id'):
            lane['endpoint_id'] = metadata['endpoint_id']
        lanes.append(lane)
    return validate_dispatch_policy(
        {
            'endpoints': fabric.get('endpoints', []),
            'endpoint_groups': fabric.get('endpoint_groups', []),
            'lanes': lanes,
            'limits': fabric.get('limits', {}),
            'policy': data.get('policy', 'drr'),
            'total_concurrent_dispatch': data.get('total_concurrent_dispatch'),
        }
    )


def build_policy_generation_snapshot(
    fabric_group_id: str,
    generation: int,
    data: dict[str, Any],
) -> tuple[dict[str, Any], AdmissionPolicy]:
    dispatch = persisted_dispatch_policy(data)
    admission = AdmissionPolicy.from_rows(
        fabric_group_id,
        generation,
        data.get('lanes', []),
    )
    dispatch_pools = {lane['pool_id'] for lane in dispatch['lanes']}
    if any(rule.pool_id not in dispatch_pools for rule in admission.rules):
        raise ValueError('Admission pool has no dispatch lane')
    snapshot = {
        'schema_version': 1,
        'fabric_group_id': fabric_group_id,
        'scheduler': {
            'enabled': data.get('enabled', True),
            'policy': dispatch['policy'],
            'total_concurrent_dispatch': dispatch['total_concurrent_dispatch'],
        },
        'dispatch': dispatch,
        'admission': admission.to_snapshot(),
    }
    return snapshot, admission


def validate_legacy_lane_endpoints(
    scheduler_config: Any, runtime_base_url: str | None
) -> None:
    """V2 database lanes may only reuse the explicitly trusted runtime destination."""

    def canonical(value: str) -> tuple[str, str, int, str]:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {'http', 'https'}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or '?' in value
            or '#' in value
        ):
            raise ValueError('v2_endpoint_policy_requires_v3')
        return (
            parsed.scheme,
            parsed.hostname.lower(),
            parsed.port or (443 if parsed.scheme == 'https' else 80),
            parsed.path.rstrip('/'),
        )

    base = canonical(runtime_base_url or 'https://api.openai.com/v1')
    for lane in scheduler_config.lanes:
        if lane.endpoint_url and canonical(lane.endpoint_url) != base:
            raise ValueError('v2_endpoint_policy_requires_v3')
