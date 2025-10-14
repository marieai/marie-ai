from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple


class SseBroker:
    """
    In-process SSE broker with:
      - topics by api_key
      - per-topic replay ring buffer
      - bounded per-subscriber queues (backpressure: drop-oldest)
      - periodic heartbeats
    """

    def __init__(
        self,
        *,
        replay_size: int = 200,
        subscriber_q_max: int = 1024,
        heartbeat_interval_s: float = 15.0,
    ):
        self._replay_size = replay_size
        self._subscriber_q_max = subscriber_q_max
        self._hb_s = heartbeat_interval_s

        # Per-topic state: topic -> (next_id, deque[(id, event, data_json)])
        self._replay_topics: Dict[str, Tuple[int, Deque[Tuple[int, str, str]]]] = (
            defaultdict(lambda: (1, deque(maxlen=self._replay_size)))
        )
        self._subs_topics: Dict[str, set[asyncio.Queue[str]]] = defaultdict(set)

        # Global state (ALL): (next_id, deque[(id, topic, event, data_json)])
        self._replay_all: Tuple[int, Deque[Tuple[int, str, str, str]]] = (
            1,
            deque(maxlen=self._replay_size),
        )
        # Queue -> bool(named=True/False)
        self._subs_all: Dict[asyncio.Queue[str], bool] = {}

        self._lock = asyncio.Lock()

    @staticmethod
    def _sse_frame(
        *,
        event: Optional[str],
        data: str,
        id_: Optional[int] = None,
        retry_ms: Optional[int] = None,
    ) -> str:
        lines: List[str] = []
        if retry_ms is not None:
            lines.append(f"retry: {retry_ms}")
        if id_ is not None:
            lines.append(f"id: {id_}")
        if event:
            lines.append(f"event: {event}")
        for line in data.splitlines() or [""]:
            lines.append(f"data: {line}")
        lines.append("")  # blank terminator
        print('SSE Frame :', lines)
        return "\n".join(lines)

    async def publish(self, *, topic: str, event: str, payload: Any) -> None:
        print('Publishing SSE', topic, event, payload)
        data_json = json.dumps(payload, default=str)
        async with self._lock:
            # Per-topic
            next_id, dq = self._replay_topics[topic]
            msg_id = next_id
            self._replay_topics[topic] = (next_id + 1, dq)
            dq.append((msg_id, event, data_json))

            frame_topic = self._sse_frame(event=event, data=data_json, id_=msg_id)
            dead: List[asyncio.Queue[str]] = []
            for q in self._subs_topics[topic]:
                while q.qsize() >= self._subscriber_q_max:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                try:
                    q.put_nowait(frame_topic)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subs_topics[topic].discard(q)

            # Global (ALL)
            next_all, dq_all = self._replay_all
            all_id = next_all
            self._replay_all = (next_all + 1, dq_all)
            dq_all.append((all_id, topic, event, data_json))

            dead_all: List[asyncio.Queue[str]] = []
            for q, named in list(self._subs_all.items()):
                while q.qsize() >= self._subscriber_q_max:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if named:
                    frame = self._sse_frame(event=event, data=data_json, id_=all_id)
                else:
                    # default message event with type + topic included
                    frame = self._sse_frame(
                        event=None,
                        data=json.dumps(
                            {
                                "type": event,
                                "topic": topic,
                                "payload": json.loads(data_json),
                            }
                        ),
                        id_=all_id,
                    )
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    dead_all.append(q)
            for q in dead_all:
                self._subs_all.pop(q, None)

    async def subscribe(
        self, *, topic: str, last_event_id: Optional[int] = None, retry_ms: int = 20000
    ) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._subscriber_q_max)
        stop = asyncio.Event()

        async with self._lock:
            self._subs_topics[topic].add(q)
            await q.put(self._sse_frame(event=None, data="", retry_ms=retry_ms))
            if last_event_id is not None:
                _, dq = self._replay_topics[topic]
                for mid, ev, dj in dq:
                    if mid > last_event_id:
                        await q.put(self._sse_frame(event=ev, data=dj, id_=mid))

        async def heartbeat():
            while not stop.is_set():
                await asyncio.sleep(self._hb_s)
                await q.put(f": keepalive {int(time.time())}\n\n")

        hb_task = asyncio.create_task(heartbeat())
        try:
            while not stop.is_set():
                yield await q.get()
        finally:
            stop.set()
            hb_task.cancel()
            async with self._lock:
                self._subs_topics[topic].discard(q)

    async def subscribe_all(
        self,
        *,
        last_event_id: Optional[int] = None,
        named: bool = True,  # True => keep SSE event names; False => single default 'message'
        retry_ms: int = 20000,
    ) -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._subscriber_q_max)
        stop = asyncio.Event()
        print('SSE SUBSCRIBE ALL', last_event_id, named, retry_ms)

        async with self._lock:
            self._subs_all[q] = named
            await q.put(self._sse_frame(event=None, data="", retry_ms=retry_ms))
            # replay from global buffer
            next_all, dq_all = self._replay_all
            for mid, topic, ev, dj in dq_all:
                print('sse all debug ', topic, ev, dj)
                if last_event_id is not None and mid <= last_event_id:
                    continue
                if named:
                    await q.put(self._sse_frame(event=ev, data=dj, id_=mid))
                else:
                    await q.put(
                        self._sse_frame(
                            event=None,
                            data=json.dumps(
                                {"type": ev, "topic": topic, "payload": json.loads(dj)}
                            ),
                            id_=mid,
                        )
                    )

        async def heartbeat():
            while not stop.is_set():
                await asyncio.sleep(self._hb_s)
                await q.put(f": keepalive {int(time.time())}\n\n")

        hb_task = asyncio.create_task(heartbeat())
        try:
            while not stop.is_set():
                yield await q.get()
        finally:
            stop.set()
            hb_task.cancel()
            async with self._lock:
                self._subs_all.pop(q, None)

    async def _heartbeat_task(self, q: asyncio.Queue[str], stop: asyncio.Event):
        while not stop.is_set():
            await asyncio.sleep(self._hb_s)
            await q.put(f": keepalive {int(time.time())}\n\n")
