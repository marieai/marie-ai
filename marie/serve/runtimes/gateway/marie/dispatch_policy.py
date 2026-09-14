"""Versioned operator endpoint policy in existing scheduler metadata."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from marie.engine.llm_queue.endpoint import RegisteredEndpoint
from marie.engine.llm_queue.request_dispatcher import DispatchLane
from marie.engine.llm_queue.store import StoreLimits


def validate_dispatch_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate policy before resolving any named credential bindings."""
    if not isinstance(policy, dict) or set(policy) - {
        'endpoints',
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
    lanes = policy.get('lanes', [])
    if (
        not isinstance(endpoints, list)
        or not isinstance(lanes, list)
        or not 0 < len(endpoints) <= limits.max_endpoints
        or not 0 < len(lanes) <= limits.max_routes
    ):
        raise ValueError('Invalid dispatch policy size')
    registered = {}
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
        if instance.endpoint_id in registered:
            raise ValueError('Duplicate endpoint identity')
        registered[instance.endpoint_id] = instance
        if (
            instance.execution_limit > limits.max_execution_items
            or instance.execution_bytes > limits.max_execution_bytes
        ):
            raise ValueError('Endpoint exceeds immutable fabric limits')
    seen = set()
    for row in lanes:
        lane = DispatchLane(**row)
        endpoint = registered.get(lane.endpoint_id)
        if endpoint is None or lane.pool_id in seen:
            raise ValueError('Invalid lane endpoint binding')
        seen.add(lane.pool_id)
        if (
            not 0 < lane.execution_limit <= endpoint.execution_limit
            or not 0 < lane.execution_bytes <= endpoint.execution_bytes
        ):
            raise ValueError('Lane exceeds registered endpoint limits')
    if (
        sum(row.get('min_concurrent', 0) for row in lanes if row.get('enabled', True))
        > total
    ):
        raise ValueError('Protected slots exceed dispatch capacity')
    return {
        'endpoints': endpoints,
        'lanes': lanes,
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
        or set(fabric) - {'schema_version', 'endpoints', 'limits'}
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
            - {'schema_version', 'endpoint_id', 'revision', 'execution_bytes'}
        ):
            raise ValueError('Invalid persisted dispatch pool policy')
        lanes.append(
            {
                'pool_id': row['pool_id'],
                'enabled': row['enabled'],
                'endpoint_id': metadata['endpoint_id'],
                'revision': metadata['revision'],
                'execution_limit': row['max_concurrent'],
                'execution_bytes': metadata['execution_bytes'],
                'quantum': row.get('quantum', 1),
                'min_concurrent': row.get('min_concurrent', 0),
                'max_burst_per_visit': row.get('max_burst_per_visit'),
            }
        )
    return validate_dispatch_policy(
        {
            'endpoints': fabric['endpoints'],
            'lanes': lanes,
            'limits': fabric.get('limits', {}),
            'policy': data.get('policy', 'drr'),
            'total_concurrent_dispatch': data.get('total_concurrent_dispatch'),
        }
    )


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
