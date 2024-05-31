from enum import Enum


class States(Enum):
    CREATED = "created"
    RETRY = "retry"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"
