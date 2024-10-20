import time
from typing import TYPE_CHECKING, Optional, Union

from marie.importer import ImportExtensions
from marie.proto import jina_pb2

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.metrics import Meter
    from prometheus_client import CollectorRegistry

    from marie.types_core.request import Request


class MonitoringMixin:
    """The Monitoring Mixin for pods"""

    def _setup_monitoring(self, monitoring: bool, port_monitoring: Union[int, str]):
        """
        Wait for the monitoring server to start
        :param monitoring: flag indicating whether monitoring has to be activated
        :param port_monitoring: port where to expose the monitoring
        """

        if monitoring:
            from prometheus_client import CollectorRegistry

            self.metrics_registry = CollectorRegistry()
        else:
            self.metrics_registry = None

        if monitoring:
            from prometheus_client import start_http_server

            try:
                start_http_server(int(port_monitoring), registry=self.metrics_registry)
            except OSError:
                self.logger.error(
                    f'Could not start the monitoring server on port {port_monitoring}. '
                    f'Please check that the port is not already in use.'
                )
                raise


class MonitoringRequestMixin:
    """
    Mixin for the request handling monitoring

    :param metrics_registry: optional metrics registry for prometheus used if we need to expose metrics from the executor or from the data request handler
    :param runtime_name: optional runtime_name that will be registered during monitoring
    """

    def __init__(
        self,
        metrics_registry: Optional['CollectorRegistry'] = None,
        meter: Optional['Meter'] = None,
        runtime_name: Optional[str] = None,
    ):

        self._request_init_time = {} if metrics_registry else None
        self._meter_request_init_time = {} if meter else None

        if metrics_registry:
            with ImportExtensions(
                required=True,
                help_text='You need to install the `prometheus_client` to use the monitoring functionality of marie',
            ):
                from prometheus_client import Counter, Gauge, Summary

                from marie.serve.monitoring import _SummaryDeprecated

            self._receiving_request_metrics = Summary(
                'receiving_request_seconds',
                'Time spent processing successful request',
                registry=metrics_registry,
                namespace='marie',
                labelnames=('runtime_name',),
            ).labels(runtime_name)

            self._pending_requests_metrics = Gauge(
                'number_of_pending_requests',
                'Number of pending requests',
                registry=metrics_registry,
                namespace='marie',
                labelnames=('runtime_name',),
            ).labels(runtime_name)

            self._failed_requests_metrics = Counter(
                'failed_requests',
                'Number of failed requests',
                registry=metrics_registry,
                namespace='marie',
                labelnames=('runtime_name',),
            ).labels(runtime_name)

            self._successful_requests_metrics = Counter(
                'successful_requests',
                'Number of successful requests',
                registry=metrics_registry,
                namespace='marie',
                labelnames=('runtime_name',),
            ).labels(runtime_name)

            self._request_size_metrics = _SummaryDeprecated(
                old_name='request_size_bytes',
                name='received_request_bytes',
                documentation='The size in bytes of the request returned to the client',
                namespace='marie',
                labelnames=('runtime_name',),
                registry=metrics_registry,
            ).labels(runtime_name)

            self._sent_response_bytes = Summary(
                'sent_response_bytes',
                'The size in bytes of the request returned to the client',
                namespace='marie',
                labelnames=('runtime_name',),
                registry=metrics_registry,
            ).labels(runtime_name)

        else:
            self._receiving_request_metrics = None
            self._pending_requests_metrics = None
            self._failed_requests_metrics = None
            self._successful_requests_metrics = None
            self._request_size_metrics = None
            self._sent_response_bytes = None

        if meter:
            self._receiving_request_histogram = meter.create_histogram(
                name='marie_receiving_request_seconds',
                description='Time spent processing successful request',
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
        if self._request_size_metrics:
            self._request_size_metrics.observe(request.nbytes)
        if self._request_size_histogram:
            self._request_size_histogram.record(
                request.nbytes, attributes=self._metric_labels
            )

        if self._receiving_request_metrics:
            self._request_init_time[request.request_id] = time.time()
        if self._receiving_request_histogram:
            self._meter_request_init_time[request.request_id] = time.time()

        if self._pending_requests_metrics:
            self._pending_requests_metrics.inc()
        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                1, attributes=self._metric_labels
            )

    def _update_end_successful_requests_metrics(self, result: 'Request'):
        if (
            self._receiving_request_metrics
        ):  # this one should only be observed when the metrics is succesful
            init_time = self._request_init_time.pop(
                result.request_id
            )  # need to pop otherwise it stays in memory forever
            self._receiving_request_metrics.observe(time.time() - init_time)
        if (
            self._receiving_request_histogram
        ):  # this one should only be observed when the metrics is succesful
            init_time = self._meter_request_init_time.pop(
                result.request_id
            )  # need to pop otherwise it stays in memory forever
            self._receiving_request_histogram.record(
                time.time() - init_time, attributes=self._metric_labels
            )

        if self._pending_requests_metrics:
            self._pending_requests_metrics.dec()
        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                -1, attributes=self._metric_labels
            )

        if self._successful_requests_metrics:
            self._successful_requests_metrics.inc()
        if self._successful_requests_counter:
            self._successful_requests_counter.add(1, attributes=self._metric_labels)

        if self._sent_response_bytes:
            self._sent_response_bytes.observe(result.nbytes)
        if self._sent_response_bytes_histogram:
            self._sent_response_bytes_histogram.record(
                result.nbytes, attributes=self._metric_labels
            )

    def _update_end_failed_requests_metrics(self):
        if self._pending_requests_metrics:
            self._pending_requests_metrics.dec()
        if self._pending_requests_up_down_counter:
            self._pending_requests_up_down_counter.add(
                -1, attributes=self._metric_labels
            )

        if self._failed_requests_metrics:
            self._failed_requests_metrics.inc()
        if self._failed_requests_counter:
            self._failed_requests_counter.add(1, attributes=self._metric_labels)

    def _update_end_request_metrics(self, result: 'Request'):
        if result.status.code != jina_pb2.StatusProto.ERROR:
            self._update_end_successful_requests_metrics(result)
        else:
            self._update_end_failed_requests_metrics()
