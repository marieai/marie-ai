import logging

from marie.logging.mdc import MDC


class MDCContextFilter(logging.Filter):
    """This filter adds the MDC to the log record."""

    def __init__(self, name: str = '', **kwargs):
        super(MDCContextFilter, self).__init__(name)
        self.context = kwargs

    def filter(self, record):
        ctx_dict = MDC.get_all()
        if not ctx_dict:
            ctx_dict = {}

        for key, value in self.context.items():
            if key not in ctx_dict:
                ctx_dict[key] = value
        record.__dict__.update(ctx_dict)
        return True
