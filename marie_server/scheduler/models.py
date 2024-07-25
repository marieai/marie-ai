from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str


class ScheduledJob(BaseModel):
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
