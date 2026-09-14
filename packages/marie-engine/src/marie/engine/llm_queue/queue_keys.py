from __future__ import annotations

import re
from dataclasses import dataclass

from marie.engine.completion_contract import COMPLETION_QUEUE_CONTRACT_VERSION


def queue_namespace() -> str:
    return f"llm:{COMPLETION_QUEUE_CONTRACT_VERSION}"


def request_queue_key(pool_id: str) -> str:
    return f"list:{queue_namespace()}:requests:{pool_id}"


def reply_queue_key(producer_id: str) -> str:
    return f"list:{queue_namespace()}:replies:{producer_id}"


def producer_alive_key(producer_id: str) -> str:
    return f"key:{queue_namespace()}:producer:{producer_id}:alive"


def validate_identifier(value: str) -> str:
    """Accept bounded key components without separators or cluster hash tags."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError(
            'Identifiers must contain 1-128 letters, digits, underscores or hyphens'
        )
    return value


@dataclass(frozen=True, slots=True)
class QueueKeys:
    fabric_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.fabric_id)
        object.__setattr__(self, 'fabric_id', self.fabric_id.lower())

    @property
    def prefix(self) -> str:
        return f'llm:v3:{{fabric:{self.fabric_id}}}:'

    @property
    def owner(self) -> str:
        return self.prefix + 'owner'

    def request(self, attempt_id: str) -> str:
        return self.prefix + 'request:' + validate_identifier(attempt_id)

    def ready(self, pool_id: str) -> str:
        return self.prefix + 'ready:' + validate_identifier(pool_id)

    def alive(self, producer_id: str) -> str:
        return self.prefix + 'producer:' + validate_identifier(producer_id) + ':alive'

    def members(self, producer_id: str) -> str:
        return (
            self.prefix + 'producer:' + validate_identifier(producer_id) + ':requests'
        )

    def route(self, pool_id: str) -> str:
        return self.prefix + 'route:' + validate_identifier(pool_id)

    def endpoint(self, endpoint_id: str) -> str:
        return self.prefix + 'endpoint:' + validate_identifier(endpoint_id)
