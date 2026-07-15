"""Daemon request handling for the document extraction plugin."""

from __future__ import annotations

from typing import Any, Callable

from marie_plugins.runtime import SessionFrame, error_frame, session_frame

from .dispatch import extract_document
from .registry import capability_snapshot


def _extract_input(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    parameters = payload.get('tool_parameters')
    if isinstance(parameters, dict) and parameters:
        return parameters
    return payload


def dispatch_request(
    request: dict[str, Any],
    *,
    extractor: Callable[..., dict[str, Any]] = extract_document,
    capabilities: Callable[[], dict[str, Any]] = capability_snapshot,
) -> list[SessionFrame]:
    """Convert one daemon request into session events."""
    session_id = str(request.get('session_id') or '')
    payload = request.get('data')
    params = _extract_input(payload)
    action = (
        payload.get('action', 'extract') if isinstance(payload, dict) else 'extract'
    )

    try:
        if action == 'capabilities':
            result = capabilities()
        elif action == 'extract':
            path = params.get('path')
            output_dir = params.get('output_dir')
            if not isinstance(path, str) or not path:
                raise ValueError('path is required')
            if not isinstance(output_dir, str) or not output_dir:
                raise ValueError('output_dir is required')
            result = extractor(
                path=path,
                format_hint=params.get('format'),
                mime_type=params.get('mime_type'),
                intent=params.get('intent', 'semantic'),
                output_dir=output_dir,
                provider=params.get('provider'),
                fallback=params.get('fallback', True),
                provider_options=params.get('provider_options'),
                output_format=params.get('output_format', 'markdown'),
            )
        else:
            raise ValueError(f'unknown action: {action!r}')
    except (FileNotFoundError, PermissionError, ValueError) as error:
        return [
            error_frame(
                session_id,
                code='invalid_request',
                message=str(error),
            )
        ]
    except Exception:
        return [
            error_frame(
                session_id,
                code='internal_error',
                message='document extraction provider failed',
            )
        ]

    return [
        session_frame(session_id, 'stream', result),
        session_frame(session_id, 'end', {}),
    ]
