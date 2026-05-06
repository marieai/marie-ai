from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from marie.engine.completion_contract import CompletionReplyEnvelope
from marie.engine.llm_queue.queue_io import ListQueueClient


@dataclass
class ReplyWaiter:
    request_id: str
    reply: Optional[CompletionReplyEnvelope] = None


class ProducerSession:
    def __init__(
        self,
        *,
        queue_client: ListQueueClient,
        producer_id: str,
        alive_value: str,
        producer_ttl_seconds: int,
        refresh_interval_seconds: float,
        reply_pop_timeout_seconds: float,
        logger,
    ):
        self.queue_client = queue_client
        self.producer_id = producer_id
        self.alive_value = alive_value
        self.producer_ttl_seconds = producer_ttl_seconds
        self.refresh_interval_seconds = refresh_interval_seconds
        self.reply_pop_timeout_seconds = reply_pop_timeout_seconds
        self.logger = logger

        self._condition = threading.Condition()
        self._waiters: Dict[str, ReplyWaiter] = {}
        self._started = False
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reply_thread: Optional[threading.Thread] = None
        self._alive_thread: Optional[threading.Thread] = None

    @property
    def condition(self) -> threading.Condition:
        return self._condition

    def ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            self._stop_event.clear()
            self._reply_thread = threading.Thread(
                target=self._reply_loop,
                name=f"llm-queue-replies-{self.producer_id}",
                daemon=True,
            )
            self._alive_thread = threading.Thread(
                target=self._alive_loop,
                name=f"llm-queue-alive-{self.producer_id}",
                daemon=True,
            )
            self._reply_thread.start()
            self._alive_thread.start()
            self._started = True

    def register_waiter(self, request_id: str) -> ReplyWaiter:
        self.ensure_started()
        with self._condition:
            waiter = ReplyWaiter(request_id=request_id)
            self._waiters[request_id] = waiter
            return waiter

    def remove_waiter(self, request_id: str) -> None:
        with self._condition:
            self._waiters.pop(request_id, None)

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.queue_client.clear_producer_alive(self.producer_id)
        except Exception:
            pass
        with self._condition:
            self._condition.notify_all()
        if self._reply_thread is not None:
            self._reply_thread.join(timeout=self.reply_pop_timeout_seconds + 0.5)
        if self._alive_thread is not None:
            self._alive_thread.join(timeout=self.refresh_interval_seconds + 0.5)

    def _alive_loop(self) -> None:
        while not self._stop_event.is_set():
            self.queue_client.set_producer_alive(
                self.producer_id,
                self.alive_value,
                self.producer_ttl_seconds,
            )
            self._stop_event.wait(self.refresh_interval_seconds)

    def _reply_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                reply = self.queue_client.pop_reply(
                    self.producer_id,
                    timeout=self.reply_pop_timeout_seconds,
                )
            except Exception as exc:
                self.logger.error("Dropping malformed queue reply: %r", exc)
                continue
            if reply is None:
                continue
            with self._condition:
                waiter = self._waiters.get(reply.request_id)
                if waiter is None:
                    self.logger.info(
                        "Dropping late or unknown queue reply for request %s",
                        reply.request_id,
                    )
                    continue
                waiter.reply = reply
                self._condition.notify_all()
