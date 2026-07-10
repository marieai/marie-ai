import pytest

from marie.sensors.exceptions import SensorRegistryError
from marie.sensors.registry import SensorRegistry, register_all_sensors, register_sensor
from marie.sensors.types import SensorType


@pytest.fixture(autouse=True)
def _ensure_builtin_sensors_registered():
    # Production sensors register via module-level decorators; guarantee they
    # are loaded regardless of test import/collection order.
    register_all_sensors()


class _FakeBase:
    def __init__(self, sensor_data):
        self.sensor_data = sensor_data


def test_subtype_dispatch_exact_match():
    registry = SensorRegistry.get_instance()

    @register_sensor(SensorType.DATA_SINK, subtype="alpha")
    class AlphaSensor(_FakeBase):
        pass

    @register_sensor(SensorType.DATA_SINK, subtype="beta")
    class BetaSensor(_FakeBase):
        pass

    assert registry.get_evaluator(SensorType.DATA_SINK, subtype="alpha") is AlphaSensor
    assert registry.get_evaluator(SensorType.DATA_SINK, subtype="beta") is BetaSensor


def test_subtype_fallback_and_missing():
    registry = SensorRegistry.get_instance()

    # Production registration: S3DataSinkSensor is registered as (DATA_SINK, "s3"),
    # which serves as the back-compat default for unknown DATA_SINK subtypes.
    s3_default = registry.get_evaluator(SensorType.DATA_SINK, subtype="s3")
    assert (
        registry.get_evaluator(SensorType.DATA_SINK, subtype="does-not-exist")
        is s3_default
    )

    # A type with no default registration at all must raise, subtype or not.
    # (SensorType.ASSET has no evaluator registered anywhere in the codebase.)
    with pytest.raises(SensorRegistryError):
        registry.get_evaluator(SensorType.ASSET, subtype="does-not-exist")
    with pytest.raises(SensorRegistryError):
        registry.get_evaluator(SensorType.ASSET)

    # A type with a default (subtype=None) registration resolves any unknown
    # subtype to that default — this is intentional: SCHEDULE, WEBHOOK, etc.
    # don't use subtypes, so any subtype value falls through to (type, None).
    assert (
        registry.get_evaluator(SensorType.SCHEDULE, subtype="does-not-exist")
        is registry.get_evaluator(SensorType.SCHEDULE)
    )


def test_non_data_sink_unchanged():
    registry = SensorRegistry.get_instance()
    # schedule sensor resolution must work exactly as before (no subtype)
    assert registry.get_evaluator(SensorType.SCHEDULE) is not None
