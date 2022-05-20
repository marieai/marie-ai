from __future__ import absolute_import

import sys
import os
import traceback

import torch

import conf
from api import api
from flask import Flask

from arg_parser import ArgParser

import marie.api.IcrAPIRoutes as IcrAPIRoutes
import marie.api.WorkflowRoutes as WorkflowRoutes
from marie.api.route_handler import RouteHandler
from marie.api.sample_route import SampleRouter

from marie.common.volume_handler import VolumeHandler
from marie.version import __version__
from marie.common.file_io import PathManager
from marie.logger import setup_logger
from marie.utils.utils import ensure_exists, FileSystem

# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable
logger = setup_logger(__file__)


def create_app():
    logger.info(f"Starting app in {conf.APP_ENV} environment")
    ensure_exists(f"/tmp/marie")
    # Register VFS handlers
    base_dir = FileSystem.get_share_directory()
    PathManager.register_handler(VolumeHandler(volume_base_dir=base_dir))
    # PathManager.register_handler(VolumeHandler(volume_base_dir="/home/gbugaj/datasets/medprov/"))

    app = Flask(__name__)
    app.config.from_object(conf)
    api.init_app(app)

    @app.route("/")
    def index():
        return {"version": __version__}

    with app.app_context():
        # app.register_blueprint(IcrAPIRoutes.blueprint)
        # app.register_blueprint(WorkflowRoutes.blueprint)
        RouteHandler.register_route(SampleRouter(app))

    return app


if __name__ == "__main__":

    args = ArgParser.server_parser()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Marie-AI : %s", __version__)
    logger.info("[PID]%d [UID]%d", os.getpid(), os.getuid())
    logger.info("Python runtime: %s", sys.version.replace("\n", ""))
    logger.info("Environment : %s", conf.APP_ENV)
    logger.info("Torch version : %s", torch.__version__)
    logger.info("Using device: %s", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        logger.info("Device : %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory Allocated: %d GB", round(torch.cuda.memory_allocated(0) / 1024**3, 1))
        logger.info("GPU Memory Cached: %d GB", round(torch.cuda.memory_reserved(0) / 1024**3, 1))

    # Setting use_reloader to false prevents application from initializing twice
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FLASK_DEBUG"] = "1"
    logger.info("Starting 🦊-marie ")

    service = create_app()
    service.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
