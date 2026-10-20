import re
from contextlib import contextmanager
from datetime import datetime, timezone

from tools.stress.scheduler_db_stresser import CorpusGenerator, StressConfig, build_dag


def test_generated_corpus_populates_submission_queue() -> None:
    copies = []

    class Copy:
        def __init__(self, statement):
            self.columns = [
                column.strip()
                for column in re.search(r"\((.*?)\)", statement, re.S)[1].split(",")
            ]
            self.rows = []

        def write_row(self, row):
            self.rows.append(dict(zip(self.columns, row, strict=True)))

    class Connection:
        @contextmanager
        def cursor(self):
            yield self

        @contextmanager
        def copy(self, statement):
            copy = Copy(statement)
            copies.append(copy)
            yield copy

    config = StressConfig.from_mapping(
        dict(run_id="queue-identity", target_dag_count=1, queue_name="kb_indexing")
    )
    dag = build_dag(config, 0, datetime.now(timezone.utc))
    CorpusGenerator(Connection())._copy_chunk(config, [dag])
    assert copies[0].rows[0].get("submission_name") == "kb_indexing"
    assert all(row["name"] == "kb_indexing" for row in copies[1].rows)
