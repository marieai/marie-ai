import inspect
import os
import platform
import sys
import traceback
from typing import Any, Dict, Optional

from rich.traceback import install

from marie import Deployment, Flow, __version__
from marie.conf.helper import load_yaml
from marie.constants import (
    __cache_path__,
    __config_dir__,
    __marie_home__,
    __model_path__,
)
from marie.importer import ImportExtensions
from marie.logging_core.mdc import MDC
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import Toast
from marie.messaging.events import EngineEventData, MarieEvent
from marie.messaging.publisher import event_builder
from marie.utils.device import gpu_device_count
from marie.utils.server_runtime import setup_auth, setup_storage, setup_toast_events
from marie.utils.types import strtobool

DEFAULT_TERM_COLUMNS = 120


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    from dotenv import load_dotenv

    logger.info(f"Loading env file from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, verbose=True)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exceptions
    :param exc_type:
    :param exc_value:
    :param exc_traceback:
    """
    logger.error("exc_type", exc_type)
    logger.error("exc_value", exc_value)
    logger.error("exc_traceback", exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


def columns():
    """Returns the number of columns available for displaying the output."""
    if "COLUMNS" in os.environ:
        return int(os.environ["COLUMNS"])

    if not sys.stdout.isatty():
        return DEFAULT_TERM_COLUMNS

    try:
        tput_columns = os.popen("tput cols", "r").read().rstrip()
        os.environ["COLUMNS"] = str(int(tput_columns))

        return int(tput_columns)
    except:
        return DEFAULT_TERM_COLUMNS


def main(
    yml_config: str,
    env: Optional[Dict[str, str]] = None,
    env_file: Optional[str] = None,
):
    """
    Main entry point for the Marie server
    :param yml_config:
    :param env:
    :param env_file:
    """
    try:
        # setup debugpy for remote debugging
        if strtobool(os.environ.get("MARIE_DEBUG", False)):
            debugpy_port = int(
                (
                    os.environ.get("MARIE_DEBUG_PORT")
                    if os.environ.get("MARIE_DEBUG_PORT")
                    else 5678
                )
            )

            logger.info(
                f"Setting up debugpy for remote debugging on port {debugpy_port}"
            )

            with ImportExtensions(
                required=True,
                help_text=f'debugpy is needed to enable remote debugging. Please install it with "pip install debugpy"',
            ):
                import debugpy

                # Required see https://github.com/microsoft/debugpy/issues/262
                # debugpy.configure({"python": "python", "subProcess": True})
                host = os.environ.get("MARIE_DEBUG_HOST", "0.0.0.0")
                debugpy.listen((host, debugpy_port))

                # Pause the program until a remote debugger is attached
                if os.environ.get("MARIE_DEBUG_WAIT_FOR_CLIENT", True):
                    logger.info(
                        f"Waiting for the debugging client to connect on port {debugpy_port}"
                    )
                    debugpy.wait_for_client()
    except Exception as e:
        logger.error(f"Error setting up debugpy : {e}")

    __main__(yml_config, env, env_file)


def patch_libs():
    """Patch the libraries"""
    logger.warning("Patching libraries if needed")


def __main__(
    yml_config: str,
    env: Optional[Dict[str, str]] = None,
    env_file: Optional[str] = None,
):
    """Main entry point for the Marie server
    :param yml_config:
    :param env:
    :param env_file:
    """
    # install handler for exceptions
    sys.excepthook = handle_exception
    install(show_locals=True)

    MDC.put("request_id", "main")
    logger.info(f"Starting marie server : {__version__}")

    if "NO_VERSION_CHECK" not in os.environ:
        from marie_server.helper import is_latest_version

        is_latest_version(github_repo="marie-ai")

    torch_home = os.path.join(__model_path__, "cache", "torch")
    os.environ["TORCH_HOME"] = torch_home

    PYTHONPATH = os.environ.get("PYTHONPATH", "")

    logger.info(f"Debugging information:")
    logger.info(f"__model_path__ = {__model_path__}")
    logger.info(f"__config_dir__ = {__config_dir__}")
    logger.info(f"__marie_home__ = {__marie_home__}")
    logger.info(f"__cache_path__ = {__cache_path__}")
    logger.info(f"yml_config = {yml_config}")
    logger.info(f"env = {env}")
    logger.info(f"CONTEXT.gpu_device_count = {gpu_device_count()}")

    if not env_file:
        env_file = os.path.join(__config_dir__, ".env")
    load_env_file(dotenv_path=env_file)

    context = {
        "gpu_device_count": gpu_device_count(),
    }

    jemallocpath = "/usr/lib/%s-linux-gnu/.so.2" % (platform.machine(),)

    if os.path.isfile(jemallocpath):
        logger.info("Found %s, will use" % (jemallocpath,))
        os.environ["LD_PRELOAD"] = jemallocpath
    else:
        logger.warning("Could not find %s, will not use" % (jemallocpath,))

    # put env variables into context
    if env:
        for k, v in env.items():
            context[k] = v
            os.environ[k] = v

    # dump environment variables
    if "DUMP_ENV" in os.environ:
        for k, v in os.environ.items():
            print(f"{k} = {v}")

    patch_libs()

    # Load the config file and set up the toast events
    config = load_yaml(yml_config, substitute=True, context=context)
    prefetch = config.get("prefetch", 1)

    # flow or deployment
    if True:
        f = Flow.load_config(
            config,
            extra_search_paths=[
                os.path.dirname(inspect.getfile(inspect.currentframe()))
            ],
            substitute=True,
            context=context,
            include_gateway=False,
            noblock_on_start=False,
            prefetch=prefetch,
            external=True,
        ).config_gateway(prefetch=prefetch)

    if False:
        f = Deployment.load_config(
            config,
            extra_search_paths=[
                os.path.dirname(inspect.getfile(inspect.currentframe()))
            ],
            substitute=True,
            context=context,
            include_gateway=False,
            noblock_on_start=False,
            prefetch=prefetch,
            statefull=False,
            external=False,
        )

    # marie.helper.extend_rest_interface = partial(extend_rest_interface, f, prefetch)

    filter_endpoint()
    setup_server(config)
    if False:
        api_key = "main"
        job_id = "main"
        event = "server_start"
        job_tag = "main"
        status = "start"
        timestamp = None
        payload = {}

        Toast.notify_sync(
            "server_start",
            event_builder(api_key, job_id, event, job_tag, status, timestamp, payload),
        )

        if False:
            MarieEvent.engine_event(
                f"Starting server with config {yml_config} and env {env}",
                EngineEventData(
                    metadata={
                        "config": yml_config,
                        "env": env,
                        "context": context,
                        "prefetch": prefetch,
                    }
                ),
            )

    with f:
        f.block()


def setup_server(config: Dict[str, Any]) -> None:
    """
    Set up the server
    :param config: the config
    """
    setup_toast_events(config.get("toast", {}))
    setup_storage(config.get("storage", {}))
    setup_auth(config.get("auth", {}))


def filter_endpoint() -> None:
    """
    Filter out dry_run endpoint from uvicorn logs
    :return:
    """
    import logging

    # filter out dry_run and health/status  endpoint from uvicorn logs

    class _EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return (
                record.getMessage().find("GET /dry_run") == -1
                and record.getMessage().find("GET /health/status") == -1
            )

    logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-i":
            _input = sys.stdin.read()
        else:
            _input = sys.argv[1]
    else:
        _input = os.path.join(__config_dir__, "service", "marie.yml")

    main(_input)
