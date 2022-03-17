import logging
import sys
from logging import handlers
from logging.handlers import RotatingFileHandler


# https://www.loggly.com/ultimate-guide/python-logging-basics/

loggers = {}

def create_info_logger(logname: str, logfile: str):
    """
        Return a logger with the specified name, creating it if necessary.
    """

    if logname in loggers:
        return loggers[logname]

    log = logging.getLogger(logname)
    log.setLevel(logging.DEBUG)
    
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    format = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = handlers.RotatingFileHandler(logfile, maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    loggers[logname] = log
    return log
