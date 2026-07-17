"""Marie AI persistence adapter for reusable LLM scheduler configuration."""

import re
from typing import Any, Optional

from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin

DEFAULT_SCHEDULER_CONFIG_SCHEMA = "marie_scheduler"
DEFAULT_FABRIC_CONFIG_TABLE = "llm_queue_fabric_config"
DEFAULT_POOL_TABLE = "llm_queue_pool"
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresSchedulerConfigRepository(PostgresqlMixin):
    """Load engine scheduler configuration from Marie's PostgreSQL store."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        logger: Optional[MarieLogger] = None,
    ) -> None:
        super().__init__()
        self.logger = logger or MarieLogger(self.__class__.__name__)
        self.config_schema = _sql_identifier(
            config.get("schema") or DEFAULT_SCHEDULER_CONFIG_SCHEMA,
            label="schema",
        )
        self._setup_storage(config, connection_only=True)

    def load_scheduler_config(self, fabric_group_id: str) -> dict[str, Any]:
        fabric_config_table = f"{self.config_schema}.{DEFAULT_FABRIC_CONFIG_TABLE}"
        pool_table = f"{self.config_schema}.{DEFAULT_POOL_TABLE}"
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT policy, total_concurrent_dispatch
                FROM {fabric_config_table}
                WHERE fabric_group_id = %s
                  AND enabled = true
                """,
                (fabric_group_id,),
            )
            fabric_config = cursor.fetchone()
            if fabric_config is None:
                raise ValueError(
                    f"LLM queue Runtime Fabric group {fabric_group_id!r} is not configured"
                )

            cursor.execute(
                f"""
                SELECT
                    pool_id,
                    display_name,
                    endpoint_url,
                    quantum,
                    min_concurrent,
                    max_concurrent,
                    max_burst_per_visit,
                    enabled
                FROM {pool_table}
                WHERE fabric_group_id = %s
                ORDER BY sort_order ASC, pool_id ASC
                """,
                (fabric_group_id,),
            )
            lanes = [
                {
                    "pool_id": row[0],
                    "display_name": row[1],
                    "endpoint_url": row[2],
                    "quantum": row[3],
                    "min_concurrent": row[4],
                    "max_concurrent": row[5],
                    "max_burst_per_visit": row[6],
                    "enabled": row[7],
                }
                for row in cursor.fetchall()
            ]
            conn.commit()
            return {
                "policy": fabric_config[0],
                "total_concurrent_dispatch": fabric_config[1],
                "lanes": lanes,
            }
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)


def _sql_identifier(value: Any, *, label: str) -> str:
    identifier = str(value).strip()
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid LLM queue scheduler {label}: {value!r}")
    return identifier
