"""Datasource dispatch over the existing session protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from marie.extension.datasource import (
    DatasourceDefinition,
    DatasourceInputError,
    DatasourceOutputError,
    DatasourceRequest,
    DatasourceResult,
    invoke_datasource,
)

from .session import SessionFrame, error_frame, session_frame


def dispatch_datasource(
    request: dict[str, Any],
    definition: DatasourceDefinition,
    fetch: Callable[[DatasourceRequest], DatasourceResult | dict[str, Any]],
) -> Iterator[SessionFrame]:
    """Invoke a plugin-bound handler; callers cannot select implementation code."""
    session_id = request['session_id']
    try:
        result = invoke_datasource(definition, request.get('data'), fetch)
    except DatasourceInputError:
        yield error_frame(
            session_id,
            code='invalid_datasource_request',
            message='Invalid datasource request',
        )
        return
    except DatasourceOutputError:
        yield error_frame(
            session_id,
            code='invalid_datasource_output',
            message='Invalid datasource output',
        )
        return
    yield session_frame(session_id, 'stream', result.model_dump())
    yield session_frame(session_id, 'end', {})
