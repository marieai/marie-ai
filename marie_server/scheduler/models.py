from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str


class JobWorkItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid7str())
    name: str
    priority: int
    data: Dict[str, Any]
    retryLimit: int
    retryDelay: int
    retryBackoff: bool
    startAfter: datetime
    expireInSeconds: int
    singletonKey: str
    keepUntil: datetime
    onComplete: bool


class ExistingWorkPolicy(Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"
    FAIL = "FAIL"
