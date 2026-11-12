from __future__ import annotations

from collections.abc import Callable

from marie.kb.item_state import (
    bind_upstream_input,
    item_indexing_run,
    resolve_indexing_parameters,
)
from marie.scheduler.models import WorkInfo
from marie.storage.database.postgres_pool import (
    AsyncPostgresConnection,
    AsyncPostgresConnectionPool,
)


class KbSubmissionHooks:
    """KB-owned preparation and transactional workflow admission."""

    def __init__(self, pool: Callable[[], AsyncPostgresConnectionPool]) -> None:
        self._pool = pool

    async def prepare(self, work_info: WorkInfo) -> None:
        """Resolve registered indexing input before its workflow is planned."""
        metadata = work_info.data.get('metadata', {})
        if metadata.get('planner', work_info.name) not in (
            'kb_indexing',
            'kb_index_content',
        ) or not (metadata.get('item_id') or metadata.get('ref_type') == 'kb_item'):
            return
        scope, _ = item_indexing_run(metadata)
        item = await self._pool().fetchval(
            '''SELECT to_jsonb(item) FROM marie_scheduler.kb_item_state item
               WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND item_id=$4''',
            *scope.values(),
        )
        work_info.data['metadata'] = resolve_indexing_parameters(item, metadata)

    async def persist(
        self, connection: AsyncPostgresConnection, work_info: WorkInfo
    ) -> None:
        """Bind waiting KB items in the workflow's admission transaction."""
        metadata = work_info.data.get('metadata', {})
        await bind_upstream_input(
            connection,
            work_info.id,
            metadata,
            metadata.get('planner') or work_info.name,
        )
