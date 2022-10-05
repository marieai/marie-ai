from marie.healthchecks.health_indicator import HealthIndicator


class HealthStatus:
    """Health status of a health check"""

    def __init__(self, indicator, msg, attributes):
        self._indicator = indicator
        self._message = msg
        self._attributes = attributes

    @classmethod
    def from_status(cls, status):
        cls(status.indicator, status.msg, status.attributes)

    @property
    def message(self):
        return self._message

    @property
    def indicator(self):
        return self._indicator

    @property
    def attributes(self):
        return self._attributes

    @staticmethod
    def ok():
        return HealthStatus(HealthIndicator.HEALTHY, "OK")

    @staticmethod
    def failure():
        return HealthStatus(HealthIndicator.FAILURE, "Failure")

    @staticmethod
    def failure(failure):
        return HealthStatus(HealthIndicator.FAILURE, "Failure :" + failure)

    @staticmethod
    def critical():
        return HealthStatus(HealthIndicator.CRITICAL, "Critical")
