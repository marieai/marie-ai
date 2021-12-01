import logging
import sys
from logging import handlers
from logging.handlers import RotatingFileHandler

# https://www.loggly.com/ultimate-guide/python-logging-basics/


def create_info_logger(logname: str, logfile: str):
    """
        Create info logger
    """
    log = logging.getLogger(logname)
    log.setLevel(logging.INFO)
    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = handlers.RotatingFileHandler(
        logfile, maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    return log
