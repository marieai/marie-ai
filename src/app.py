from __future__ import absolute_import

import os
import traceback

import api.IcrAPIRoutes as IcrAPIRoutes
import api.WorkflowRoutes as WorkflowRoutes
import conf
from api import api
from flask import Flask

from common.file_io import PathManager, VolumeHandler
from logger import create_info_logger
from utils.utils import ensure_exists, FileSystem

# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable

log = create_info_logger("app", "marie.log")
# traceback.print_stack()
# print(repr(traceback.format_stack()))
# print(repr(traceback.extract_stack()))


def create_app():
    log.info(f'Starting app in {conf.APP_ENV} environment')
    ensure_exists(f'/tmp/marie')

    # Register VFS handlers
    base_dir = FileSystem.get_share_directory()
    PathManager.register_handler(VolumeHandler(volume_base_dir=base_dir))
    log.info(f'*** vfs base_dir :  {base_dir}')

    app = Flask(__name__)
    app.config.from_object(conf)
    api.init_app(app)

    @app.route("/")
    def index():
        return {
            "version": "1.0.2"
        }

    with app.app_context():
        # Import parts of our application
        # Register Blueprints
        app.register_blueprint(IcrAPIRoutes.blueprint)
        app.register_blueprint(WorkflowRoutes.blueprint)

    return app


if __name__ == "__main__":
    log.info('Initializing system')
    log.info(f'***config PATH {conf}')
    log.info(f'***config PATH {conf.APP_ENV}')
    # Setting use_reloader to false prevents application from initializing twice
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FLASK_DEBUG"] = "1"

    service = create_app()
    service.run(host='0.0.0.0', port=5100, debug=True, use_reloader=True)
    # service.run(host='0.0.0.0', port=5100, debug=True, use_reloader=False)
