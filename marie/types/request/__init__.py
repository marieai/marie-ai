import traceback
from typing import Optional

from marie.serve.executors import BaseExecutor
from marie.types.mixin import ProtoTypeMixin


class StatusProto:
    ERROR = 500


class Request(ProtoTypeMixin):

    def __getattr__(self, name: str):
        return getattr(self.proto, name)

    def add_exception(self, ex: Optional["Exception"] = None, executor: "BaseExecutor" = None) -> None:
        """Add exception to the last route in the envelope
        :param ex: Exception to be added
        :param executor: Executor related to the exception
        """
        d = self.header.status
        d.code = StatusProto.ERROR
        d.description = repr(ex)

        if executor:
            d.exception.executor = executor.__class__.__name__
        d.exception.name = ex.__class__.__name__
        d.exception.args.extend([str(v) for v in ex.args])
        d.exception.stacks.extend(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
