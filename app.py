import logging
import sys
from logging import handlers
from logging.handlers import RotatingFileHandler
from os import environ

from api import api
from flask import Flask

import config

LOGFILE = 'icr.log'
log = logging.getLogger('marie')
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = handlers.RotatingFileHandler(LOGFILE, maxBytes=(1048576 * 5), backupCount=7)
fh.setFormatter(format)
log.addHandler(fh)


def create_app():
   log.info(f'Starting app in {config.APP_ENV} environment')

   app = Flask(__name__)
   app.config.from_object('config')
   api.init_app(app)

   @app.route("/")
   def home():
      return "ICR Index"

   return app

if __name__ == "__main__":
   log.info('Initializing system')
   app = create_app()
   app.run(host='0.0.0.0', debug=True)
   # app.run(threaded=True, port=environ.get('PORT'), debug=True)
