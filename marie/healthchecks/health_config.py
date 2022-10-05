from typing import Dict, List, Union


class HealthConfig:
    """
    Base health check configuration that
    Times are in  ISO-8601 representation
    """

    def __init__(self):
        # Unique ID
        self._service_id = None
        # User friendly check name
        self._name = None
        # Service name that we are monitoring or null/empty for node level monitoring
        self._service = None
        # Time To Live for the results before they are invalidated, 0 = NO TTL, default to 10 seconds
        self._ttl = "PT10S"
        # Timeout time for the check before failing, default to 2 seconds
        self._timeout = "PT2S"
        # Interval between checks, Default to 0 = NO interval checks
        self.interval = "PT0S"
        # Additional tags associated with this command
        self._tags = None

    @property
    def interval(self):
        return self.interval

    @interval.setter
    def interval(self, interval):
        self.interval = interval

    @property
    def ttl(self):
        return self._ttl

    @ttl.setter
    def ttl(self, ttl):
        self._ttl = ttl

    @property
    def service_id(self):
        return self._service_id

    @service_id.setter
    def service_id(self, _service_id):
        self._service_id = _service_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        self._timeout = timeout

    @property
    def service(self):
        return self._service

    @service.setter
    def service(self, service):
        self._service = service

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags: [Union[Dict, List[Dict]]]):
        self._tags = tags

    def __str__(self):
        return (
            "HealthConfig{"
            + "service_id='"
            + self.service_id
            + "'"
            + ", name='"
            + self._name
            + "'"
            + ", service='"
            + self._service
            + "'"
            + ", ttl='"
            + self._ttl
            + "'"
            + ", timeout='"
            + self._timeout
            + "'"
            + ", interval='"
            + self.interval
            + "'"
            + ", tags="
            + self._tags
            + "}"
        )
