from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

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


class JobSubmissionModel(BaseModel):
    action_type: str
    command: str
    action: str
    name: str
    metadata: Dict[str, Any]


class ExistingWorkPolicy(Enum):
    ALLOW_ALL = "allow_all"  # allow all submissions
    REJECT_ALL = "reject_all"  # reject all submissions
    REPLACE = "replace"  # replace existing submission that is not in terminal state
    ALLOW_DUPLICATE = "allow_duplicate"  # allow duplicate submissions
    REJECT_DUPLICATE = "reject_duplicate"  # reject duplicate submissions


class BackoffPolicyType(Enum):
    EXPONENTIAL_BACKOFF = "EXPONENTIAL_BACKOFF"
    FIXED_BACKOFF = "FIXED_BACKOFF"

    @staticmethod
    def create(value: str):
        if value.upper() not in BackoffPolicyType.__members__:
            raise ValueError(f"Invalid value {value} for BackoffPolicyType")
        return BackoffPolicyType[value.upper()]


class RetryPolicy:
    DEFAULT_USER_RETRY_LIMIT = 2
    DEFAULT_USER_RETRY_DELAY = 2
    DEFAULT_TIMEOUT_RETRY_LIMIT = 3
    DEFAULT_RETRY_BACKOFF = True

    def __init__(self, retry_limit: int, retry_delay: int, retry_backoff: bool):
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

    @classmethod
    def default_retry_policy(cls):
        return cls(
            retry_limit=cls.DEFAULT_USER_RETRY_LIMIT,
            retry_delay=cls.DEFAULT_USER_RETRY_DELAY,
            retry_backoff=cls.DEFAULT_RETRY_BACKOFF,
        )

    DEFAULT_RETRY_POLICY = default_retry_policy()
