import contextvars

ctx_mdc = contextvars.ContextVar("mdc", default={})


class MDC(object):
    """
    The Mapped Diagnostic Context, or MDC in short, is an instrument for distinguishing interleaved log output from different sources.
    """

    @staticmethod
    def put(key: str, value: object):
        """Put a context value as identified with the key parameter into the current thread's context map."""
        d = ctx_mdc.get()
        d[key] = value
        ctx_mdc.set(d)

    @staticmethod
    def get(key: str):
        """Get the context identified by the key parameter."""
        d = ctx_mdc.get()
        if key in d:
            return d[key]
        else:
            raise KeyError(f"Key {key} not found in MDC")

    @staticmethod
    def remove(key: str):
        """Remove the context identified by the key parameter."""
        d = ctx_mdc.get()
        if key in d:
            del d[key]
        ctx_mdc.set(d)

    @staticmethod
    def clear():
        """Remove all values from the MDC."""
        d = ctx_mdc.get()
        d.clear()
        ctx_mdc.set(d)

    @staticmethod
    def get_all() -> dict:
        """
        Get all the context values as a dictionary.
        :return: A dictionary containing all the context values.
        """
        ctx = contextvars.copy_context()
        return ctx.get(ctx_mdc)
