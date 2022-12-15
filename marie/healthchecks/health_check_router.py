import json
import os
import re
import shutil
import subprocess

import psutil
from flask import jsonify, url_for
from flask_restful import Resource, reqparse, request

from marie.api import extract_payload
from marie.logging.logger import MarieLogger
from marie.utils.network import get_ip_address

logger = MarieLogger("")


def check_disk_usage(disk):
    """Returns True if free disk is less than 20%"""
    disk_usage = shutil.disk_usage(disk)
    free_disk_percent = disk_usage.free / disk_usage.total * 100
    return free_disk_percent < 20


def check_cpu_usage():
    """Returns True if cpu usage is greater than 80%"""
    cpu_percent = int(psutil.cpu_percent(1))
    return cpu_percent > 80


def check_host():
    """Returns True if localhost not in 127.0.0.1"""
    process = subprocess.run(["host", "127.0.0.1"], capture_output=True)
    result = process.stdout.decode().split()
    regex = re.search(r"localhost", result[-1])
    return regex is None


def check_ram():
    """Returns True if free RAM is less than 500 MB."""
    ram_usage = dict(psutil.virtual_memory()._asdict())
    free_ram = float("{:.2f}".format(ram_usage["free"] / 1024 / 1024))
    return free_ram < 500


def check_gpu_usage():
    """Verifies that there's enough unused GPU"""
    # nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu
    return True


class HealthCheckRouter:
    """
    Health check router that will be used to check status of the services
    """

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        if app is None:
            raise RuntimeError("Expected app arguments is null")
        self.app = app
        prefix = "health"
        app.add_url_rule(
            rule=f"/{prefix}",
            endpoint="_health_index",
            view_func=self.index,
            methods=["GET"],
        )

        app.add_url_rule(
            rule=f"/{prefix}/status",
            endpoint="_health_status",
            view_func=self.version,
            methods=["GET"],
        )

    def list_routes(self):

        output = []
        app = self.app
        for rule in app.url_map.iter_rules():

            options = {}
            for arg in rule.arguments:
                options[arg] = "[{0}]".format(arg)

            methods = ",".join(rule.methods)
            url = url_for(rule.endpoint, **options)
            line = "{:50s} {:20s} {}".format(rule.endpoint, methods, url)
            output.append(line)

        for line in sorted(output):
            print(line)

        return output

    def version(self):
        """Get application status"""
        import os

        if True:
            raise TypeError("This is a fake error")

        build = {}
        if os.path.exists(".build"):
            with open(".build", "r") as fp:
                build = json.load(fp)
        host = get_ip_address()
        routes = self.list_routes()

        return (
            jsonify(
                {
                    "name": "marie-icr",
                    "host": host,
                    "component": [
                        {"name": "craft", "version": "1.0.0"},
                        {"name": "craft-benchmark", "version": "1.0.0"},
                    ],
                    "build": build,
                    "routes": routes,
                }
            ),
            200,
        )

    def index(self):
        """Get application health checks"""

        return (
            jsonify(
                {
                    "status": "Healthy",
                    "totalDuration": "00:00:00.0023926",
                    "entries": [
                        {"name": "check-01", "id": "health.id.0"},
                        {"name": "check-02", "id": "health.id.1"},
                    ],
                }
            ),
            200,
        )
