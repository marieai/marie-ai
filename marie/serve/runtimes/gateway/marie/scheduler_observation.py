"""Read-only scheduler configuration from an activated policy generation."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from urllib.parse import urlsplit

from marie.engine.llm_queue.endpoint import RegisteredReplica

_REPLICA_DEFAULTS = {field.name: field.default for field in fields(RegisteredReplica)}


def scheduler_observation(
    snapshot: dict[str, Any], generation: int, *, limit: int
) -> dict[str, Any]:
    """Project configuration without credential bindings or arbitrary metadata."""
    dispatch = snapshot.get('dispatch', {})
    rules = {
        rule['pool_id']: rule.get('admission')
        for rule in snapshot.get('admission', {}).get('rules', [])
    }
    lanes = dispatch.get('lanes', [])
    pools = [
        {
            'pool_id': lane['pool_id'],
            'enabled': lane.get('enabled', True),
            'quantum': lane.get('quantum', 1),
            'min_concurrent': lane.get('min_concurrent', 0),
            'max_concurrent': lane.get('execution_limit'),
            'max_burst_per_visit': lane.get('max_burst_per_visit'),
            'execution_bytes': lane.get('execution_bytes'),
            'endpoint_group_id': lane.get('endpoint_group_id')
            or lane.get('endpoint_id'),
            'revision': lane.get('revision'),
            'admission': rules.get(lane['pool_id']),
        }
        for lane in lanes[:limit]
    ]
    groups = dispatch.get('endpoint_groups', []) or [
        {
            'group_id': endpoint['endpoint_id'],
            'revision': 'r1',
            'replicas': [{**endpoint, 'replica_id': endpoint['endpoint_id']}],
        }
        for endpoint in dispatch.get('endpoints', [])
    ]
    visible_groups = []
    replicas_left = limit
    replicas_truncated = False
    for group in groups[:limit]:
        replicas = group.get('replicas', [])
        visible_replicas = []
        for replica in replicas[:replicas_left]:
            address = replica.get('base_url')
            try:
                parsed = urlsplit(address or '')
                if (
                    parsed.scheme not in {'http', 'https'}
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                ):
                    address = None
            except ValueError:
                address = None
            visible_replicas.append(
                {
                    'replica_id': replica['replica_id'],
                    'enabled': replica.get('enabled', _REPLICA_DEFAULTS['enabled']),
                    'address': address,
                    'execution_limit': replica.get(
                        'execution_limit', _REPLICA_DEFAULTS['execution_limit']
                    ),
                    'execution_bytes': replica.get(
                        'execution_bytes', _REPLICA_DEFAULTS['execution_bytes']
                    ),
                    'call_timeout_seconds': replica.get(
                        'call_timeout_seconds',
                        _REPLICA_DEFAULTS['call_timeout_seconds'],
                    ),
                }
            )
        replicas_truncated |= len(replicas) > replicas_left
        replicas_left -= len(visible_replicas)
        visible_groups.append(
            {
                'group_id': group['group_id'],
                'revision': group.get('revision'),
                'replicas': visible_replicas,
            }
        )
    scheduler = snapshot.get('scheduler', {})
    return {
        'generation': generation,
        'enabled': scheduler.get('enabled'),
        'policy': scheduler.get('policy') or dispatch.get('policy'),
        'total_concurrent_dispatch': scheduler.get('total_concurrent_dispatch'),
        'pools': pools,
        'pools_truncated': len(lanes) > limit,
        'endpoint_groups': visible_groups,
        'endpoint_groups_truncated': len(groups) > limit or replicas_truncated,
    }
