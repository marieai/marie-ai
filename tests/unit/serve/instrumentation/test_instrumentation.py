import json
import time

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from marie.serve.instrumentation import MetricsTimer


@pytest.fixture
def metrics_setup():
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    meter = meter_provider.get_meter('test')
    yield metric_reader, meter
    if hasattr(meter_provider, 'force_flush'):
        meter_provider.force_flush()
    if hasattr(meter_provider, 'shutdown'):
        meter_provider.shutdown()


def test_timer_context(metrics_setup):
    def _do_something():
        time.sleep(0.1)

    metric_reader, meter = metrics_setup
    histogram = meter.create_histogram(
        name='time_taken', description='measure something'
    )

    with MetricsTimer(histogram=histogram):
        _do_something()

    # OpenTelemetry samples
    histogram_metric = json.loads(
        metric_reader.get_metrics_data()
        .resource_metrics[0]
        .scope_metrics[0]
        .metrics[0]
        .to_json()
    )
    assert 'time_taken' == histogram_metric['name']
    assert 1 == histogram_metric['data']['data_points'][0]['count']


def test_timer_decorator(metrics_setup):
    metric_reader, meter = metrics_setup
    histogram = meter.create_histogram(
        name='time_taken_decorator', description='measure something'
    )

    @MetricsTimer(histogram)
    def _sleep():
        time.sleep(0.1)

    _sleep()

    # OpenTelemetry samples
    histogram_metric = json.loads(
        metric_reader.get_metrics_data()
        .resource_metrics[0]
        .scope_metrics[0]
        .metrics[0]
        .to_json()
    )
    assert 'time_taken_decorator' == histogram_metric['name']
    assert 1 == histogram_metric['data']['data_points'][0]['count']
    assert {} == histogram_metric['data']['data_points'][0]['attributes']

    labels = {
        'cat': 'meow',
        'dog': 'woof',
    }

    @MetricsTimer(histogram, labels)
    def _sleep_with_labels():
        time.sleep(0.1)

    _sleep_with_labels()

    # OpenTelemetry samples - check that we have 2 data points now
    histogram_metric = json.loads(
        metric_reader.get_metrics_data()
        .resource_metrics[0]
        .scope_metrics[0]
        .metrics[0]
        .to_json()
    )
    assert 'time_taken_decorator' == histogram_metric['name']
    # Find the data point with labels
    data_points = histogram_metric['data']['data_points']
    assert len(data_points) == 2
    labeled_point = next(dp for dp in data_points if dp['attributes'] == labels)
    assert 1 == labeled_point['count']
