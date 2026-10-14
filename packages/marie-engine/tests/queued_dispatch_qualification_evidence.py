"""Optional sanitized local qualification output, outside normal test artifacts."""

import json
import os
from pathlib import Path


def record_evidence(name, store, data):
    destination = os.getenv('MARIE_LLM_TASK6_EVIDENCE_DIR')
    if not destination:
        return
    server = store.client.info('server')
    store_name = 'valkey' if 'valkey_version' in server else 'redis'
    payload = {
        'boundary': 'local synthetic fixtures; no deployment or model qualification',
        'server': {
            k: v
            for k, v in server.items()
            if k in {'redis_version', 'valkey_version', 'server_name'}
        },
        'persistence': store.client.config_get(
            'save', 'appendonly', 'maxmemory', 'maxmemory-policy'
        ),
        **data,
    }
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    (path / f'{name}-{store_name}.json').write_text(
        json.dumps(payload, indent=2, default=str) + '\n'
    )
