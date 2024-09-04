from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

from marie_server.scheduler.state import WorkState


# https://cs.android.com/androidx/platform/frameworks/support/+/androidx-main:work/work-runtime/src/main/java/androidx/work/WorkInfo.kt
class WorkInfo(BaseModel):
    id: str = Field(default_factory=lambda: uuid7str())
    name: str
    priority: int
    data: Dict[str, Any]
    state: WorkState
    retry_limit: int
    retry_delay: int
    retry_backoff: bool
    start_after: datetime
    expire_in_seconds: int
    keep_until: datetime


class ExistingWorkPolicy(Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"
    FAIL = "FAIL"
