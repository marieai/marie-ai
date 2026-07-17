from __future__ import annotations

from marie.engine.completion_contract import COMPLETION_QUEUE_CONTRACT_VERSION


def queue_namespace() -> str:
    return f"llm:{COMPLETION_QUEUE_CONTRACT_VERSION}"


def request_queue_key(pool_id: str) -> str:
    return f"list:{queue_namespace()}:requests:{pool_id}"


def reply_queue_key(producer_id: str) -> str:
    return f"list:{queue_namespace()}:replies:{producer_id}"


def producer_alive_key(producer_id: str) -> str:
    return f"key:{queue_namespace()}:producer:{producer_id}:alive"
