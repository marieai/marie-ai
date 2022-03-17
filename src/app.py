from __future__ import absolute_import

import os
import traceback

import api.IcrAPIRoutes as IcrAPIRoutes
import conf
from api import api
from flask import Flask
from logger import create_info_logger
from utils.utils import ensure_exists


# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable


log = create_info_logger("app", "marie.log")
# traceback.print_stack()
# print(repr(traceback.format_stack()))
# print(repr(traceback.extract_stack()))


def create_app():
    log.info(f'Starting app in {conf.APP_ENV} environment')
    ensure_exists(f'/tmp/marie')

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

    return app


if __name__ == "__main__":
    log.info('Initializing system')

    print(f'***config PATH {conf}')
    print(f'***config PATH {conf.APP_ENV}')

    # Setting use_reloader to false prevents application from initializing twice
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FLASK_DEBUG"] = "1"

    service = create_app()
    service.run(host='0.0.0.0', port=5100, debug=True, use_reloader=True)
    # service.run(host='0.0.0.0', port=5100, debug=True, use_reloader=False)
