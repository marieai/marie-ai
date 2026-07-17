from __future__ import annotations

from typing import Any, Optional, Protocol

from marie.engine.completion_contract import CompletionCallParams


class ExecutionAdapter(Protocol):
    async def execute(
        self,
        call: CompletionCallParams,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Any: ...
