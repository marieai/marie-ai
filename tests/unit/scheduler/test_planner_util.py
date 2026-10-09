from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import marie.query_planner.mapper as mapper_module
import marie.scheduler.planner_util as planner_util
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    Query,
    QueryPlan,
)
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


def test_query_plan_work_items_preserves_submission_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extract_dir = tmp_path / 'extract'
    (extract_dir / 'base').mkdir(parents=True)
    (extract_dir / 'base' / 'mapper.yml').write_text(
        'model_executors: {}\nserverless_executor: default\n'
    )
    monkeypatch.setattr(mapper_module, '__default_extract_dir__', str(extract_dir))
    monkeypatch.setattr(planner_util, '__default_extract_dir__', str(extract_dir))
    monkeypatch.chdir(tmp_path)
    plan = QueryPlan(
        nodes=[
            Query(task_id='root', query_str='start', definition=NoopQueryDefinition()),
            Query(
                task_id='child',
                query_str='extract',
                dependencies=['root'],
                definition=ExecutorEndpointQueryDefinition(
                    endpoint='extract://default'
                ),
            ),
        ]
    )
    monkeypatch.setattr(planner_util, 'query_planner', lambda _: plan)
    now = datetime.now(timezone.utc)
    submitted = WorkInfo(
        id='submission',
        name='test-input-planner',
        data={
            'metadata': {'on': 'extract://direct', 'custom': ['preserved']},
        },
        state=WorkState.CREATED,
        retry_limit=3,
        retry_delay=2,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=60,
        keep_until=now + timedelta(days=1),
        dependencies=['submission-parent'],
    )
    original = submitted.model_dump()

    _, jobs = planner_util.query_plan_work_items(submitted)

    assert submitted.model_dump() == original
    assert [job.id for job in jobs] == ['root', 'child']
    assert jobs[0].dependencies == []
    assert jobs[1].dependencies == ['root']
    assert jobs[1].data['metadata']['on'] == 'extract://direct'
    jobs[1].data['metadata']['custom'].append('changed')
    assert submitted.model_dump() == original
