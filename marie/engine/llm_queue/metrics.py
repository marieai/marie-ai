from __future__ import annotations

from typing import Any

from opentelemetry import metrics as otel_metrics

from marie.engine.llm_queue.registry import dispatch_runtime_snapshot

try:  # pragma: no cover - import surface varies by OTel version
    from opentelemetry.metrics import Observation
except ImportError:  # pragma: no cover
    Observation = None  # type: ignore[assignment]


class DispatchMetrics:
    def __init__(self) -> None:
        self._meter = otel_metrics.get_meter("marie.engine.llm_queue.dispatcher")
        self._batches_counter = self._meter.create_counter(
            name="marie_llm_dispatch_batches",
            description="Number of LLM dispatch batches executed",
        )
        self._items_counter = self._meter.create_counter(
            name="marie_llm_dispatch_items",
            description="Number of LLM dispatch items executed",
        )
        self._batch_duration = self._meter.create_histogram(
            name="marie_llm_dispatch_batch_seconds",
            description="Time spent executing a dispatch batch",
            unit="s",
        )
        self._batch_size = self._meter.create_histogram(
            name="marie_llm_dispatch_batch_size",
            description="Number of completion calls included in a dispatch batch",
            unit="{items}",
        )
        self._request_duration = self._meter.create_histogram(
            name="marie_llm_dispatch_request_seconds",
            description="Time spent executing an individual queued completion call",
            unit="s",
        )
        self._execution_failures = self._meter.create_counter(
            name="marie_llm_dispatch_execution_failures",
            description="Number of queued completion execution failures",
        )
        self._dropped_requests = self._meter.create_counter(
            name="marie_llm_dispatch_dropped_requests",
            description="Number of queued requests dropped before execution",
        )
        self._dropped_replies = self._meter.create_counter(
            name="marie_llm_dispatch_dropped_replies",
            description="Number of queued replies dropped before delivery",
        )

        if Observation is not None:
            self._meter.create_observable_gauge(
                name="marie_llm_dispatch_registered_dispatchers",
                callbacks=[self._observe_registered_dispatchers],
                description="Number of registered LLM dispatch runtimes in this process",
                unit="{dispatchers}",
            )
            self._meter.create_observable_gauge(
                name="marie_llm_dispatch_running_dispatchers",
                callbacks=[self._observe_running_dispatchers],
                description="Number of running LLM dispatch runtimes in this process",
                unit="{dispatchers}",
            )
            self._meter.create_observable_gauge(
                name="marie_llm_dispatch_request_queue_depth",
                callbacks=[self._observe_queue_depth],
                description="Current queued request depth per LLM dispatch runtime",
                unit="{requests}",
            )

    def record_batch(
        self,
        *,
        pool_id: str,
        dispatcher_id: str,
        batch_size: int,
        duration_seconds: float,
    ) -> None:
        attrs = self._attrs(pool_id=pool_id, dispatcher_id=dispatcher_id)
        self._batches_counter.add(1, attributes=attrs)
        self._items_counter.add(batch_size, attributes=attrs)
        self._batch_size.record(batch_size, attributes=attrs)
        self._batch_duration.record(duration_seconds, attributes=attrs)

    def record_request_execution(
        self,
        *,
        pool_id: str,
        dispatcher_id: str,
        duration_seconds: float,
        ok: bool,
    ) -> None:
        attrs = self._attrs(
            pool_id=pool_id,
            dispatcher_id=dispatcher_id,
            status="ok" if ok else "error",
        )
        self._request_duration.record(duration_seconds, attributes=attrs)
        if not ok:
            self._execution_failures.add(1, attributes=attrs)

    def record_request_drop(
        self,
        *,
        pool_id: str,
        dispatcher_id: str,
        reason: str,
    ) -> None:
        self._dropped_requests.add(
            1,
            attributes=self._attrs(
                pool_id=pool_id,
                dispatcher_id=dispatcher_id,
                reason=reason,
            ),
        )

    def record_reply_drop(
        self,
        *,
        pool_id: str,
        dispatcher_id: str,
        reason: str,
    ) -> None:
        self._dropped_replies.add(
            1,
            attributes=self._attrs(
                pool_id=pool_id,
                dispatcher_id=dispatcher_id,
                reason=reason,
            ),
        )

    @staticmethod
    def _attrs(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def _observe_registered_dispatchers(self, _options: Any):
        snapshot = dispatch_runtime_snapshot()
        return [
            Observation(
                snapshot.get("registered_dispatchers", 0),
                {"contract_version": str(snapshot.get("contract_version", "unknown"))},
            )
        ]

    def _observe_running_dispatchers(self, _options: Any):
        snapshot = dispatch_runtime_snapshot()
        return [
            Observation(
                snapshot.get("running_dispatchers", 0),
                {"contract_version": str(snapshot.get("contract_version", "unknown"))},
            )
        ]

    def _observe_queue_depth(self, _options: Any):
        snapshot = dispatch_runtime_snapshot()
        observations = []
        contract_version = str(snapshot.get("contract_version", "unknown"))
        for dispatcher in snapshot.get("dispatchers", []):
            depth = dispatcher.get("request_queue_depth")
            if depth is None:
                continue
            observations.append(
                Observation(
                    depth,
                    {
                        "contract_version": contract_version,
                        "pool_id": str(dispatcher.get("pool_id", "")),
                        "dispatcher_id": str(dispatcher.get("dispatcher_id", "")),
                    },
                )
            )
        return observations


dispatch_metrics = DispatchMetrics()
