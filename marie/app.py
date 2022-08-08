from __future__ import absolute_import

import sys
import os
import traceback

import torch
import yaml
from werkzeug.exceptions import HTTPException

import conf
from api import api
from flask import Flask, url_for
from functools import wraps
from flask import Flask, redirect, jsonify
from arg_parser import ArgParser

import marie.api.IcrAPIRoutes as IcrAPIRoutes
import marie.api.WorkflowRoutes as WorkflowRoutes
from marie.api.icr_router import ICRRouter
from marie.api.ner_router import NERRouter
from marie.api.route_handler import RouteHandler
from marie.api.sample_route import SampleRouter

from marie.common.volume_handler import VolumeHandler
from marie.executor import NerExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger
from marie.utils.network import find_open_port
from marie.version import __version__
from marie.common.file_io import PathManager
from marie import __cache_dir__

# from marie.logger import setup_logger
from marie.utils.utils import ensure_exists, FileSystem

# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable
logger = default_logger


def create_app(config_data):
    logger.info(f"Starting app in {conf.APP_ENV} environment")
    ensure_exists(f"/tmp/marie")
    # Register VFS handlers
    base_dir = FileSystem.get_share_directory()
    PathManager.register_handler(VolumeHandler(volume_base_dir=base_dir))

    app = Flask(__name__)
    app.config.from_object(conf)
    app.config["APPLICATION_ROOT"] = "/api"

    @app.errorhandler(Exception)
    def error_500(exception):
        """
        Override exception handler to return JSON.
        """
        code = 500
        name = str(type(exception).__name__)
        description = str(exception)

        if isinstance(exception, HTTPException):
            code = exception.code
            name = exception.name
            description = exception.description

        # we have critical status and not able to recover
        # let the monitoring service know so we can unregister the service
        ipc_send_status(online_status=False)

        return (
            jsonify(
                {"error": {"code": code, "name": name, "description": description,}}
            ),
            code,
            {"Content-Type": "application/json"},
        )

    api.init_app(app)

    def __executor_conf(config, executor_name):
        if "executors" not in config_data:
            return {}

        for executor in config_data["executors"]:
            if "uses" in executor:
                if executor["uses"] == executor_name:
                    return executor
        return {}

    @app.route("/")
    def index():
        return {"version": __version__}, 200

    with app.app_context():
        # RouteHandler.register_route(ICRRouter(app))
        RouteHandler.register_route(
            NERRouter(
                app, **__executor_conf(config_data, NerExtractionExecutor.__name__)
            )
        )

    return app


def list_routes(app):
    output = []
    for rule in app.url_map.iter_rules():

        options = {}
        for arg in rule.arguments:
            options[arg] = "[{0}]".format(arg)

        methods = ",".join(rule.methods)
        url = url_for(rule.endpoint, **options)
        line = "{:50s} {:20s} {}".format(rule.endpoint, methods, url)
        output.append(line)

    for line in sorted(output):
        logger.info(line)

    return output


def ipc_send_status(online_status: bool):
    """
    Send IPC status message indicating status of the service
    """
    from multiprocessing.connection import Client

    conn = None
    address = ("localhost", 6500)
    try:
        conn = Client(address, authkey=b"redfox")
        conn.send({"command": "status", "online": online_status})
        conn.send("CLOSE")
    except ConnectionRefusedError:
        logger.warning(f"Unable to connect to monitoring client : {address}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    # import sys
    # for p in sys.path:
    #     print(p)

    # export PYTHONPATH = "$PWD"
    pypath = os.environ["PYTHONPATH"]

    # os.environ["MARIE_DEFAULT_SHARE_PATH"] = "/opt/shares/medrxprovdata"

    opt = ArgParser.extract_args()
    # Load config
    with open(opt.config, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        logger.info(f"Config read successfully : {opt.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing 🦊-Marie (X004): %s", __version__)
    logger.info("[PID]%d [UID]%d", os.getpid(), os.getuid())
    logger.info("Python runtime: %s", sys.version.replace("\n", ""))
    logger.info("Environment : %s", conf.APP_ENV)
    logger.info("Torch version : %s", torch.__version__)
    logger.info("Using device: %s", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        logger.info("Device : %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory Allocated: %d GB",
            round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1),
        )
        logger.info(
            "GPU Memory Cached: %d GB",
            round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1),
        )

    # Setting use_reloader to false prevents application from initializing twice
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FLASK_DEBUG"] = "1"

    # by default cache is located in '~/.cache' here we will map it under the runtime cache directory
    os.environ["TORCH_HOME"] = str(__cache_dir__)

    # os.environ["MARIE_PORT"] = "-1"

    logger.info(f"Environment variables")
    for k, v in os.environ.items():
        logger.info(f"env : {k}={v}")

    server_port = os.getenv("MARIE_PORT", 5000)
    if server_port == "-1":
        server_port = find_open_port()
    else:
        server_port = int(server_port)

    with open("port.dat", "w", encoding="utf-8") as fsrc:
        fsrc.write(f"{server_port}")
        logger.info(f"server_port = {server_port}")

    service = create_app(config_data)
    service.run(host="0.0.0.0", port=server_port, debug=False, use_reloader=False)
