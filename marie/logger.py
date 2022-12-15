import logging
import sys
from logging import handlers

# https://www.loggly.com/ultimate-guide/python-logging-basics/

loggers = {}


def setup_logger(name: str, logfile: str = "marie.log"):
    """
    Return a logger with the specified name, creating it if necessary.
    """
    if name in loggers:
        raise RuntimeError(f'Logger is already setup : {name}')

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False  # prevent double logging triggered from Flask

    # format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # format = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
    format = logging.Formatter(
        '%(asctime)s, %(levelname)-8s [%(filename)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s'
    )

    # console log
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    # file log
    fh = handlers.RotatingFileHandler(logfile, maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    loggers[name] = log
    return log
