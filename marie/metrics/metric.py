"""
Metric class for model server
"""

import socket
import time
from builtins import str
from collections import OrderedDict

from metrics.unit import Units

MetricUnit = Units()


class Metric(object):
    """
    Class for generating metrics and printing it to stdout of the worker
    """

    def __init__(
        self, name, value, unit, dimensions, request_id=None, metric_method=None
    ):
        """
        Constructor for Metric class

           Metric class will spawn a thread and report collected metrics to stdout of worker

        Parameters
        ----------
        name: str
            Name of metric
        value : int, float
           Can be integer or float
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimensions: list
            list of dimension objects
        request_id: str
            req_id of metric
        metric_method: str
           useful for defining different operations, optional

        """
        self.name = name
        self.unit = unit
        if unit in list(MetricUnit.units.keys()):
            self.unit = MetricUnit.units[unit]
        self.metric_method = metric_method
        self.value = value
        self.dimensions = dimensions
        self.request_id = request_id

    def update(self, value):
        """
        Update function for Metric class

        Parameters
        ----------
        value : int, float
            metric to be updated
        """

        if self.metric_method == 'counter':
            self.value += value
        else:
            self.value = value

    def __str__(self):
        dims = ",".join([str(d) for d in self.dimensions])
        if self.request_id:
            return "{}.{}:{}|#{}|#hostname:{},{},{}".format(
                self.name,
                self.unit,
                self.value,
                dims,
                socket.gethostname(),
                int(time.time()),
                self.request_id,
            )

        return "{}.{}:{}|#{}|#hostname:{},{}".format(
            self.name,
            self.unit,
            self.value,
            dims,
            socket.gethostname(),
            int(time.time()),
        )

    def to_dict(self):
        """
        return an Ordered Dictionary
        """
        return OrderedDict(
            {
                'MetricName': self.name,
                'Value': self.value,
                'Unit': self.unit,
                'Dimensions': self.dimensions,
                'Timestamp': int(time.time()),
                'HostName': socket.gethostname(),
                'RequestId': self.request_id,
            }
        )
