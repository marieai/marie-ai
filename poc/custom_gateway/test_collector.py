from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Quick test - try to connect to collector
try:
    exporter = OTLPMetricExporter(endpoint='localhost:4317', insecure=True)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter('test')
    histogram = meter.create_histogram('marie_test_histogram', unit='s')
    histogram.record(0.5, {'test': 'true'})
    print('Metric recorded')
    import time

    time.sleep(2)
    provider.force_flush()
    print('Flushed metrics to localhost:4317')
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f'Error: {e}')
