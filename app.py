from __future__ import absolute_import
from os import environ

import conf
print(f'***config pathx {conf}')
print(f'***config path {conf.APP_ENV}')

from api import api
from flask import Flask
# from api.IcrAPIRoutes import IcrAPIRoutes # TypeError: 'module' object is not callable

import api.IcrAPIRoutes as IcrAPIRoutes
from logger import create_info_logger
from utils.utils import ensure_exists
import traceback

#from . import config
#config.API_PREFIX)

log = create_info_logger("app", "marie.log")

def create_app():
    log.info(f'Starting app in {conf.APP_ENV} environment')
    ensure_exists(f'/tmp/marie')

    app = Flask(__name__)
    app.config.from_object(conf)
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
