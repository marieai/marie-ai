from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from marie.engine.llm_queue.config import DEFAULT_LLM_QUEUE_POOL_ID
from marie.engine.llm_queue.scheduler import DrrLaneConfig


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
        enabled=_to_bool(item.get("enabled"), True),
    )


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
