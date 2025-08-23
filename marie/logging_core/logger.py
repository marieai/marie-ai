import atexit
import copy
import logging
import logging.handlers
import os
import platform
import sys
from collections import OrderedDict
from typing import Iterable, Mapping, NamedTuple, Optional

from rich.logging import LogRender as _LogRender
from rich.logging import RichHandler as _RichHandler
from typing_extensions import Final

from marie import check
from marie.constants import __resources_path__, __uptime__, __windows__
from marie.enums import LogVerbosity
from marie.jaml import JAML
from marie.logging_core import formatter
from marie.logging_core.filters import EnsureFieldsFilter
from marie.logging_core.log_bus import GLOBAL_LOG_BUS
from marie.logging_core.mdc_filter import MDCContextFilter
from marie.utils.json import to_json

# TODO : Implement MDC like context for logging
# https://stackoverflow.com/questions/6618513/python-logging-with-context


def _to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    if val is None:
        return default
    return bool(val)


BACKFILL_TAG_LENGTH = 8

PYTHON_LOGGING_LEVELS_MAPPING: Final[Mapping[str, int]] = OrderedDict(
    {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10}
)

PYTHON_LOGGING_LEVELS_ALIASES: Final[Mapping[str, str]] = OrderedDict(
    {"FATAL": "CRITICAL", "WARN": "WARNING"}
)

PYTHON_LOGGING_LEVELS_NAMES = frozenset(
    [
        level_name.lower()
        for level_name in sorted(
            list(PYTHON_LOGGING_LEVELS_MAPPING.keys())
            + list(PYTHON_LOGGING_LEVELS_ALIASES.keys())
        )
    ]
)


class _MyLogRender(_LogRender):
    """Override the original rich log record for more compact layout."""

    def __call__(
        self,
        console,
        renderables,
        log_time=None,
        time_format=None,
        level=None,
        path=None,
        line_no=None,
        link_path=None,
    ):
        from rich.containers import Renderables
        from rich.table import Table
        from rich.text import Text

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_level:
            output.add_column(style="log.level", width=5)

        output.add_column(ratio=1, style="log.message", overflow="ellipsis")

        if self.show_time:
            output.add_column(style="log.path")
        row = []

        if self.show_level:
            row.append(level)

        row.append(Renderables(renderables))

        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        output.add_row(*row)
        return output


class RichHandler(_RichHandler):
    """Override the original rich handler for more compact layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_render = _MyLogRender(
            show_time=self._log_render.show_time,
            show_level=self._log_render.show_level,
            show_path=self._log_render.show_path,
            time_format=self._log_render.time_format,
            omit_repeated_times=self._log_render.omit_repeated_times,
            level_width=None,
        )


class SysLogHandlerWrapper(logging.handlers.SysLogHandler):
    """
    Override the priority_map :class:`SysLogHandler`.

    .. warning::
        This messages at DEBUG and INFO are therefore not stored by ASL, (ASL = Apple System Log)
        which in turn means they can't be printed by syslog after the fact. You can confirm it via :command:`syslog` or
        :command:`tail -f /var/log/system.log`.
    """

    priority_map = {
        "DEBUG": "debug",
        "INFO": "info",
        "WARNING": "warning",
        "ERROR": "error",
        "CRITICAL": "critical",
    }


class MarieLogger:
    """
    Build a logger for a context.

    :param context: The context identifier of the class, module or method.
    :param log_config: The configuration file for the logger.
    :return:: an executor object.
    """

    supported = {"FileHandler", "StreamHandler", "SysLogHandler", "RichHandler"}

    def __init__(
        self,
        context: str,
        name: Optional[str] = None,
        log_config: Optional[str] = None,
        quiet: bool = False,
        **kwargs,
    ):

        log_config = os.getenv(
            "MARIE_LOG_CONFIG",
            log_config or "default",
        )

        if quiet or os.getenv("MARIE_LOG_CONFIG", None) == "QUIET":
            log_config = "quiet"

        if not name:
            name = os.getenv("MARIE_DEPLOYMENT_NAME", context)

        # # Remove all handlers associated with the root logger object.
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)

        mdx_context_vars = {"request_id": ""}

        self.logger = logging.getLogger(context)

        # import inspect
        # mod = inspect.getmodule(inspect.currentframe().f_back)
        # logger_name = mod.__name__ if mod and mod.__name__ else context
        # self.logger = logging.getLogger(logger_name)

        self.logger.propagate = False

        # persist context for restore on unpickle
        self._context = context
        self.logger.addFilter(MDCContextFilter(context, **mdx_context_vars))

        # context_vars = {"name": name, "uptime": __uptime__, "context": context}
        context_vars = {"app": name, "uptime": __uptime__, "context": context}

        # Build handlers and optionally enable queue mode
        self.add_handlers(log_config, **context_vars, **kwargs)

        # convenience bindings
        self.debug = self.logger.debug
        self.warning = self.logger.warning
        self.critical = self.logger.critical
        self.error = self.logger.error
        self.info = self.logger.info
        self.exception = self.logger.exception

        self._is_closed = False
        self.debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

    def __copy__(self):
        # Shallow copy: loggers are singletons; return self to avoid copying locks/threads.
        return self

    def __deepcopy__(self, memo):
        # Deepcopy: short-circuit and reuse the same instance.
        memo[id(self)] = self
        return self

    def __getstate__(self):
        # Serialize only minimal, safe state. Handlers/threads/queues are not serialized.
        return {
            "_logger_name": getattr(self.logger, "name", None),
            "_level": getattr(self.logger, "level", None),
            "_is_closed": getattr(self, "_is_closed", False),
            "_context": getattr(self, "_context", None),  # <— add
        }

    def __setstate__(self, state):
        name = state.get("_logger_name") or self.__class__.__name__
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        level = state.get("_level")
        if level is not None:
            self.logger.setLevel(level)

        # restore MDC and persisted context
        ctx = state.get("_context") or name
        self._context = ctx
        self.logger.addFilter(MDCContextFilter(ctx, request_id=""))

        # reattach to global log bus so records emit again
        try:
            GLOBAL_LOG_BUS.attach_logger(self.logger)
        except Exception:
            pass  # be tolerant during deserialization

        # rebind methods
        self.debug = self.logger.debug
        self.warning = self.logger.warning
        self.critical = self.logger.critical
        self.error = self.logger.error
        self.info = self.logger.info
        self.exception = self.logger.exception

        self._is_closed = state.get("_is_closed", False)
        self.debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

    def __reduce_ex__XXXX(self, protocol):
        # Use default reduce that leverages __getstate__/__setstate__
        return (self.__class__, (self.logger.name,), self.__getstate__())

    def success(self, *args):
        """
        Provides an API to print success messages

        :param args: the args to be forwarded to the log
        """
        self.logger.log(LogVerbosity.SUCCESS, *args)

    @property
    def handlers(self):
        """
        Get the handlers of the logger.

        :return:: Handlers of logger.
        """
        return self.logger.handlers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all handlers attached to this logger."""
        if self._is_closed:
            return

        # Close handlers attached to this logger (QueueHandler + any direct handlers)
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)

        self._is_closed = True

    def add_handlers(self, config_path: Optional[str] = None, **kwargs):
        """
        Add handlers from config file.

        :param config_path: Path of config file.
        :param kwargs: Extra parameters.
        """

        self.logger.handlers = []

        if not os.path.exists(config_path):
            old_config_path = config_path
            if "logging." in config_path and ".yml" in config_path:
                config_path = os.path.join(__resources_path__, config_path)
            else:
                config_path = os.path.join(
                    __resources_path__, f"logging.{config_path}.yml"
                )
            if not os.path.exists(config_path):
                config_path = old_config_path

        with open(config_path, encoding="utf-8") as fp:
            config = JAML.load(fp)

        target_handlers: list[logging.Handler] = []

        for h in config["handlers"]:
            cfg = config["configs"].get(h, None)
            fmt = getattr(formatter, cfg.get("formatter", "Formatter"))

            if h not in self.supported or not cfg:
                raise ValueError(
                    f"can not find configs for {h}, maybe it is not supported"
                )

            handler = None
            if h == "StreamHandler":
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(fmt(cfg["format"].format_map(kwargs)))
            elif h == "RichHandler":
                kwargs_handler = copy.deepcopy(cfg)
                kwargs_handler.pop("format")

                handler = RichHandler(**kwargs_handler)
                handler.setFormatter(fmt(cfg["format"].format_map(kwargs)))

            elif h == "SysLogHandler" and not __windows__:
                if cfg["host"] and cfg["port"]:
                    handler = SysLogHandlerWrapper(address=(cfg["host"], cfg["port"]))
                else:
                    # a UNIX socket is used
                    if platform.system() == "Darwin":
                        handler = SysLogHandlerWrapper(address="/var/run/syslog")
                    else:
                        handler = SysLogHandlerWrapper(address="/dev/log")
                if handler:
                    handler.ident = cfg.get("ident", "")
                    handler.setFormatter(fmt(cfg["format"].format_map(kwargs)))

                try:
                    handler._connect_unixsocket(handler.address)
                except OSError:
                    handler = None
                    pass
            elif h == "FileHandler":
                filename = cfg["output"].format_map(kwargs)
                if __windows__:
                    # colons are not allowed in filenames
                    filename = filename.replace(":", ".")

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                handler = logging.FileHandler(filename, delay=True)
                handler.setFormatter(fmt(cfg["format"].format_map(kwargs)))

            if handler:
                target_handlers.append(handler)

        verbose_level = LogVerbosity.from_string(config["level"])
        if "MARIE_LOG_LEVEL" in os.environ:
            verbose_level = LogVerbosity.from_string(os.environ["MARIE_LOG_LEVEL"])
        self.logger.setLevel(verbose_level.value)

        #  sensible defaults
        app_default = kwargs.get("app", "DEFAULT")  # or the deployment name you inject
        ensure_defaults = EnsureFieldsFilter(
            {
                "context": "",  # MDC context (e.g., JobSupervisor); empty if absent
                "request_id": "-",  # show a dash instead of raising
                "app": app_default,  # so %(app)s is always available if used in YAML
                "name": app_default,  # ensure {name} is always present for old configs (docker)
            }
        )

        for h in target_handlers:
            h.addFilter(ensure_defaults)

        # Decide queue mode , default is True
        # If use_queue is set to False, then it will not use the queue handler
        # This was contributing to the memory growth
        use_queue_param = kwargs.pop("use_queue", None)
        use_queue_env = _to_bool(os.getenv("MARIE_LOG_USE_QUEUE", "1"), default=True)
        use_queue = _to_bool(use_queue_param, default=use_queue_env)

        # Detach and close existing handlers from this logger only (do not touch global sinks)
        for h in list(self.logger.handlers):
            # Keep only non-QueueHandler handlers if not using queue
            try:
                h.close()
            except Exception:
                pass
            self.logger.removeHandler(h)

        if use_queue:
            GLOBAL_LOG_BUS.set_sinks(target_handlers)
            GLOBAL_LOG_BUS.attach_logger(self.logger)
        else:
            for h in target_handlers:
                self.logger.addHandler(h)


_shutdown_handler_called = False


def _flush_all_handlers(handlers: Iterable[logging.Handler]) -> None:
    for h in list(handlers):
        try:
            h.flush()
        except Exception:
            pass


def _flush_every_logger() -> None:
    # flush handlers of all known loggers (incl. root)
    root = logging.getLogger()
    _flush_all_handlers(root.handlers)
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger):
            _flush_all_handlers(lg.handlers)


_shutdown_handler_called = False


def _shutdown_handler():
    global _shutdown_handler_called
    if _shutdown_handler_called:
        return
    _shutdown_handler_called = True

    # 1) Stop the bus FIRST (drains queue and flushes sinks in caller)
    try:
        GLOBAL_LOG_BUS.stop(timeout=1.0)  # short join; final-drain happens inside
    except Exception:
        pass

    # 2) Flush any remaining direct handlers BEFORE logging.shutdown()
    try:
        _flush_every_logger()
    except Exception:
        pass

    # 3) Finalize stdlib logging exactly once
    try:
        logging.shutdown()
    except Exception:
        pass


atexit.register(_shutdown_handler)


class JsonFileHandler(logging.Handler):
    def __init__(self, json_path: str):
        super(JsonFileHandler, self).__init__()
        self.json_path = check.str_param(json_path, "json_path")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_dict = copy.copy(record.__dict__)

            # This horrific monstrosity is to maintain backwards compatability
            # with the old behavior of the JsonFileHandler, which the clarify
            # project has a dependency on. It relied on the dagster-defined
            # properties smashing all the properties of the LogRecord object
            # and uploads all of those properties to a redshift table for
            # in order to do analytics on the log

            if "dagster_meta" in log_dict:
                dagster_meta_dict = log_dict["dagster_meta"]
                del log_dict["dagster_meta"]
            else:
                dagster_meta_dict = {}

            log_dict.update(dagster_meta_dict)

            with open(self.json_path, "a", encoding="utf8") as ff:
                text_line = to_json(log_dict)
                ff.write(text_line + "\n")
        # Need to catch Exception here, so disabling lint
        except Exception as e:
            logging.critical("[%s] Error during logging!", self.__class__.__name__)
            logging.exception(str(e))


class StructuredLoggerMessage(
    NamedTuple(
        "_StructuredLoggerMessage",
        [
            ("name", str),
            ("message", str),
            ("level", int),
            ("meta", Mapping[object, object]),
            ("record", logging.LogRecord),
        ],
    )
):
    def __new__(
        cls,
        name: str,
        message: str,
        level: int,
        meta: Mapping[object, object],
        record: logging.LogRecord,
    ):
        return super(StructuredLoggerMessage, cls).__new__(
            cls,
            check.str_param(name, "name"),
            check.str_param(message, "message"),
            coerce_valid_log_level(level),
            check.mapping_param(meta, "meta"),
            check.inst_param(record, "record", logging.LogRecord),
        )


class JsonEventLoggerHandler(logging.Handler):
    def __init__(self, json_path: str, construct_event_record):
        super(JsonEventLoggerHandler, self).__init__()
        self.json_path = check.str_param(json_path, "json_path")
        self.construct_event_record = construct_event_record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event_record = self.construct_event_record(record)
            with open(self.json_path, "a", encoding="utf8") as ff:
                text_line = to_json(event_record.to_dict())
                ff.write(text_line + "\n")

        # Need to catch Exception here, so disabling lint
        except Exception as e:
            logging.critical("[%s] Error during logging!", self.__class__.__name__)
            logging.exception(str(e))


class StructuredLoggerHandler(logging.Handler):
    def __init__(self, callback):
        super(StructuredLoggerHandler, self).__init__()
        self.callback = check.is_callable(callback, "callback")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(
                StructuredLoggerMessage(
                    name=record.name,
                    message=record.msg,
                    level=record.levelno,
                    meta=record.dagster_meta,  # type: ignore
                    record=record,
                )
            )
        # Need to catch Exception here, so disabling lint
        except Exception as e:
            logging.critical("[%s] Error during logging!", self.__class__.__name__)
            logging.exception(str(e))


def coerce_valid_log_level(level):
    return level


def construct_single_handler_logger(name, level, handler):
    check.str_param(name, "name")
    check.inst_param(handler, "handler", logging.Handler)

    level = coerce_valid_log_level(level)

    # @logger
    def single_handler_logger(_init_context):
        klass = logging.getLoggerClass()
        logger_ = klass(name, level=level)
        logger_.addHandler(handler)
        handler.setLevel(level)
        return logger_

    return single_handler_logger
