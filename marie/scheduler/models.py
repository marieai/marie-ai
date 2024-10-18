from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

from marie.scheduler.state import WorkState


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
    policy: Optional[str] = None


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

    @staticmethod
    def create(value: str, default_policy: "ExistingWorkPolicy" = REJECT_DUPLICATE):
        if not value or value.upper() == "DEFAULT":
            return (
                ExistingWorkPolicy[default_policy.upper()]
                if isinstance(default_policy, str)
                else default_policy
            )

        try:
            return ExistingWorkPolicy[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid value {value} for ExistingWorkPolicy")


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


DEFAULT_RETRY_POLICY = RetryPolicy(
    retry_limit=RetryPolicy.DEFAULT_USER_RETRY_LIMIT,
    retry_delay=RetryPolicy.DEFAULT_USER_RETRY_DELAY,
    retry_backoff=RetryPolicy.DEFAULT_RETRY_BACKOFF,
)
