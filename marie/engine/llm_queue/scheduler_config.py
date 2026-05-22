from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from marie.engine.llm_queue.config import DEFAULT_LLM_QUEUE_POOL_ID
from marie.engine.llm_queue.scheduler import DrrLaneConfig
from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie.utils.types import to_bool

DEFAULT_SCHEDULER_CONFIG_SCHEMA = "marie_scheduler"
DEFAULT_FABRIC_CONFIG_TABLE = "llm_queue_fabric_config"
DEFAULT_POOL_TABLE = "llm_queue_pool"
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LlmQueueSchedulerConfig:
    policy: str = "fifo"
    total_concurrent_dispatch: int = 0
    lanes: tuple[DrrLaneConfig, ...] = ()

    @property
    def is_drr(self) -> bool:
        return self.policy == "drr"


class SchedulerConfigSource(Protocol):
    def load(self) -> LlmQueueSchedulerConfig: ...


class SchedulerConfigRepository(Protocol):
    def load_scheduler_config(self, fabric_group_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StaticSchedulerConfigSource:
    config: LlmQueueSchedulerConfig

    def load(self) -> LlmQueueSchedulerConfig:
        return self.config


@dataclass(frozen=True)
class DatabaseSchedulerConfigSource:
    repository: SchedulerConfigRepository
    fabric_group_id: str
    default_total_concurrent_dispatch: int = 0

    def load(self) -> LlmQueueSchedulerConfig:
        data = self.repository.load_scheduler_config(self.fabric_group_id)
        return scheduler_config_from_mapping(
            data,
            default_total_concurrent_dispatch=self.default_total_concurrent_dispatch,
        )


class PostgresSchedulerConfigRepository(PostgresqlMixin):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        logger: Optional[MarieLogger] = None,
    ) -> None:
        super().__init__()
        self.logger = logger or MarieLogger(PostgresSchedulerConfigRepository.__name__)
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


def scheduler_config_from_mapping(
    data: dict[str, Any],
    *,
    default_total_concurrent_dispatch: int = 0,
) -> LlmQueueSchedulerConfig:
    policy = (
        str(data.get("policy") or data.get("scheduler_policy") or "fifo")
        .strip()
        .lower()
    )
    if policy not in {"fifo", "drr"}:
        raise ValueError(f"Unsupported LLM queue scheduler policy: {policy!r}")
    total_concurrent_dispatch = int(
        data.get("total_concurrent_dispatch") or default_total_concurrent_dispatch
    )
    lane_items = data.get("lanes") or ()
    if not isinstance(lane_items, (list, tuple)):
        raise ValueError("LLM queue scheduler config lanes must be a list")

    lanes = tuple(_lane_from_item(item) for item in lane_items)
    config = LlmQueueSchedulerConfig(
        policy=policy,
        total_concurrent_dispatch=total_concurrent_dispatch,
        lanes=lanes,
    )
    return ensure_default_pool(config)


def ensure_default_pool(
    config: LlmQueueSchedulerConfig,
    *,
    default_pool_id: str = DEFAULT_LLM_QUEUE_POOL_ID,
) -> LlmQueueSchedulerConfig:
    if not config.is_drr:
        return config
    if any(lane.pool_id == default_pool_id and lane.enabled for lane in config.lanes):
        return config
    return LlmQueueSchedulerConfig(
        policy=config.policy,
        total_concurrent_dispatch=config.total_concurrent_dispatch,
        lanes=(
            *config.lanes,
            DrrLaneConfig(
                pool_id=default_pool_id,
                display_name="Default",
                quantum=1,
            ),
        ),
    )


def _lane_from_item(item: Any) -> DrrLaneConfig:
    if isinstance(item, str):
        return DrrLaneConfig(pool_id=item)
    if not isinstance(item, dict):
        raise ValueError("LLM queue scheduler lane entries must be strings or objects")
    return DrrLaneConfig(
        pool_id=str(item["pool_id"]),
        quantum=int(item.get("quantum", 1)),
        min_concurrent=int(item.get("min_concurrent", 0)),
        max_concurrent=_optional_int(item.get("max_concurrent")),
        max_burst_per_visit=_optional_int(item.get("max_burst_per_visit")),
        display_name=_optional_str(item.get("display_name")),
        endpoint_url=_optional_str(item.get("endpoint_url")),
        enabled=to_bool(item.get("enabled"), True),
    )


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _sql_identifier(value: Any, *, label: str) -> str:
    identifier = str(value).strip()
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid LLM queue scheduler {label}: {value!r}")
    return identifier
