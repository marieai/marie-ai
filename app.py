from os import environ

from api import api
from flask import Flask
# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable

import config
import api.IcrAPIRoutes as IcrAPIRoutes
from logger import create_info_logger
from utils.utils import ensure_exists
import traceback

log = create_info_logger("app", "marie.log")

def create_app():
    log.info(f'Starting app in {config.APP_ENV} environment')
    ensure_exists(f'/tmp/marie')

    app = Flask(__name__)
    app.config.from_object('config')
    api.init_app(app)

    @app.route("/")
    def index():
       return {
           "version": "1.0.1"
       }

    with app.app_context():
        # Import parts of our application
        # Register Blueprints
        app.register_blueprint(IcrAPIRoutes.blueprint)

    return app

if __name__ == "__main__":
    log.info('Initializing system')
    app = create_app()
    app.run(host='0.0.0.0', debug=True)
   # app.run(threaded=True, port=environ.get('PORT'), debug=True)
