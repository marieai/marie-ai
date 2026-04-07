import time
from typing import TYPE_CHECKING, Optional

from marie.proto import jina_pb2

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.metrics import Meter

    from marie.types_core.request import Request


class MonitoringRequestMixin:
    """
    Mixin for the request handling monitoring using OpenTelemetry.

    :param meter: optional OpenTelemetry Meter for metrics
    :param runtime_name: optional runtime_name that will be registered during monitoring
    """

    def __init__(
        self,
        meter: Optional['Meter'] = None,
        runtime_name: Optional[str] = None,
    ):
        self._meter_request_init_time = {} if meter else None

        if meter:
            self._receiving_request_histogram = meter.create_histogram(
                name='marie_receiving_request_seconds',
                description='Time spent processing request',
            )

            self._pending_requests_up_down_counter = meter.create_up_down_counter(
                name='marie_number_of_pending_requests',
                description='Number of pending requests',
            )

            self._failed_requests_counter = meter.create_counter(
                name='marie_failed_requests',
                description='Number of failed requests',
            )

            self._successful_requests_counter = meter.create_counter(
                name='marie_successful_requests',
                description='Number of successful requests',
            )

            self._request_size_histogram = meter.create_histogram(
                name='marie_received_request_bytes',
                description='The size in bytes of the request returned to the client',
            )

            self._sent_response_bytes_histogram = meter.create_histogram(
                name='marie_sent_response_bytes',
                description='The size in bytes of the request returned to the client',
            )
        else:
            self._receiving_request_histogram = None
            self._pending_requests_up_down_counter = None
            self._failed_requests_counter = None
            self._successful_requests_counter = None
            self._request_size_histogram = None
            self._sent_response_bytes_histogram = None
        self._metric_labels = {'runtime_name': runtime_name}

    def _update_start_request_metrics(self, request: 'Request'):
        if self._request_size_histogram:
            self._request_size_histogram.record(
                request.nbytes, attributes=self._metric_labels
            )

        if self._receiving_request_histogram:
            self._meter_request_init_time[request.request_id] = time.time()

        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                1, attributes=self._metric_labels
            )

    def _update_end_successful_requests_metrics(self, result: 'Request'):
        if self._receiving_request_histogram:
            init_time = self._meter_request_init_time.pop(
                result.request_id
            )  # need to pop otherwise it stays in memory forever
            self._receiving_request_histogram.record(
                time.time() - init_time, attributes=self._metric_labels
            )

        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                -1, attributes=self._metric_labels
            )

        if self._successful_requests_counter:
            self._successful_requests_counter.add(1, attributes=self._metric_labels)

        if self._sent_response_bytes_histogram:
            self._sent_response_bytes_histogram.record(
                result.nbytes, attributes=self._metric_labels
            )

    def _update_end_failed_requests_metrics(self):
        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                -1, attributes=self._metric_labels
            )

        if self._failed_requests_counter:
            self._failed_requests_counter.add(1, attributes=self._metric_labels)

    def _update_end_request_metrics(self, result: 'Request'):
        if result.status.code != jina_pb2.StatusProto.ERROR:
            self._update_end_successful_requests_metrics(result)
        else:
            self._update_end_failed_requests_metrics()
