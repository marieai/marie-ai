import logging

from marie.logging.mdc import MDC


class MDCContextFilter(logging.Filter):
    """This filter adds the MDC to the log record."""

    def filter(self, record):
        ctx_dict = MDC.get_all()
        if ctx_dict:
            record.__dict__.update(ctx_dict)
        return True
