from __future__ import annotations


def request_queue_key(pool_id: str) -> str:
    return f"list:llm:requests:{pool_id}"


def reply_queue_key(producer_id: str) -> str:
    return f"list:llm:replies:{producer_id}"


def producer_alive_key(producer_id: str) -> str:
    return f"key:llm:producer:{producer_id}:alive"
