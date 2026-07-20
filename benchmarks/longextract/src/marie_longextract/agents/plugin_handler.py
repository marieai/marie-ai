from __future__ import annotations

import asyncio
from typing import Any

from marie_longextract.agents.application import (
    propose_boundary_repair,
    propose_leaf_repair,
)
from marie_longextract.agents.plugin_models import (
    AgentInvocation,
    BoundaryRepairInput,
    LeafRepairInput,
)
from marie_plugins.runtime import SessionFrame, error_frame, session_frame
from pydantic import ValidationError


def dispatch_request(request: dict[str, Any]) -> list[SessionFrame]:
    session_id = str(request.get('session_id') or '')
    payload = request.get('data')
    action = payload.get('action') if isinstance(payload, dict) else None

    try:
        if action == 'capabilities':
            result = {
                'ready': True,
                'actions': ['run_leaf_repair', 'run_boundary_repair'],
            }
        else:
            invocation = AgentInvocation.model_validate(payload)
            if action == 'run_boundary_repair':
                values = BoundaryRepairInput.model_validate(invocation.input)
                result = asyncio.run(
                    propose_boundary_repair(
                        asset_dir=invocation.workspace.path(),
                        page_number=values.page_number,
                        record_index=values.record_index,
                        api_base=invocation.model_profile.base_url,
                        api_key=invocation.api_key(),
                        model=invocation.model_profile.model,
                        idempotency_key=invocation.idempotency_key,
                        request_timeout_seconds=(
                            invocation.model_profile.request_timeout_seconds
                        ),
                    )
                )
            elif action == 'run_leaf_repair':
                values = LeafRepairInput.model_validate(invocation.input)
                result = asyncio.run(
                    propose_leaf_repair(
                        asset_dir=invocation.workspace.path(),
                        page_numbers=values.page_numbers,
                        schema_path=invocation.artifact_path('schema'),
                        api_base=invocation.model_profile.base_url,
                        api_key=invocation.api_key(),
                        model=invocation.model_profile.model,
                        idempotency_key=invocation.idempotency_key,
                        field_names=values.field_names,
                        request_timeout_seconds=(
                            invocation.model_profile.request_timeout_seconds
                        ),
                    )
                )
            else:
                raise ValueError(f'unknown action: {action!r}')
    except (FileNotFoundError, ValidationError, ValueError) as error:
        return [
            error_frame(
                session_id,
                code='invalid_request',
                message=str(error),
            )
        ]

    return [
        session_frame(session_id, 'stream', result),
        session_frame(session_id, 'end', {}),
    ]
