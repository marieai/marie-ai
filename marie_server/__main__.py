import asyncio
import inspect
import os
import sys
import traceback
from typing import Dict, Any, Optional
import torch
from marie import Flow
from marie import __version__
import marie.helper
from marie.logging.predefined import default_logger as logger

from marie.conf.helper import load_yaml
from marie.constants import (
    __model_path__,
    __config_dir__,
    __marie_home__,
    __cache_path__,
)
from marie.messaging import (
    Toast,
    NativeToastHandler,
    RabbitMQToastHandler,
    PsqlToastHandler,
)
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.device import gpu_device_count
from rich.traceback import install

from marie_server.rest_extension import extend_rest_interface

torch.set_float32_matmul_precision("high")


def setup_toast_events(toast_config: Dict[str, Any]):
    native_config = toast_config["native"]
    psql_config = toast_config["psql"]
    rabbitmq_config = toast_config["rabbitmq"]

    Toast.register(NativeToastHandler("/tmp/events.json"), native=True)

    if psql_config is not None:
        Toast.register(PsqlToastHandler(psql_config), native=False)

    if rabbitmq_config is not None:
        Toast.register(RabbitMQToastHandler(rabbitmq_config), native=False)


def setup_storage(storage_config: Dict[str, Any]):
    """Setup the storage handler"""

    if "s3" in storage_config:
        logger.info("Setting up storage handler for S3")
        handler = S3StorageHandler(config=storage_config["s3"], prefix="S3_")
        StorageManager.register_handler(handler=handler)
        StorageManager.ensure_connection("s3://")
        StorageManager.mkdir("s3://marie")


def setup_scheduler(scheduler_config: Dict[str, Any]):
    """Set up the job scheduler"""
    if "psql" in scheduler_config:
        # check if the scheduler is enabled
        if scheduler_config["psql"]["enabled"]:
            from marie_server.scheduler import PostgreSQLJobScheduler

            scheduler = PostgreSQLJobScheduler(config=scheduler_config["psql"])
            scheduler.start_schedule()
    else:
        logger.warning("No scheduler config found")


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


def main(
    yml_config: str,
    env: Optional[Dict[str, str]] = None,
    env_file: Optional[str] = None,
):
    __main__(yml_config, env, env_file)


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
    logger.info(f"Starting marie server : {__version__}")

    if "NO_VERSION_CHECK" not in os.environ:
        from marie_server.helper import is_latest_version

        is_latest_version(github_repo="marie-ai")

    if False:
        import shutil

        # os.environ['COLUMNS'] = "211"
        # os.environ['LINES'] = "50"

        print(f"shutil.which('python') = {shutil.which('python')}")
        print(shutil.get_terminal_size())

        import logging
        import shutil
        from rich.logging import RichHandler
        from rich.console import Console

        logging.basicConfig(
            level=logging.DEBUG, handlers=[RichHandler(enable_link_path=True)]
        )
        logging.error('test')
        logging.warning('test')
        logging.info('test')
        logging.debug('test')
        Console().print(shutil.get_terminal_size())

    PYTHONPATH = os.environ.get("PYTHONPATH", "")

    logger.info(f"Debugging information:")
    logger.info(f"__model_path__ = {__model_path__}")
    logger.info(f"__config_dir__ = {__config_dir__}")
    logger.info(f"__marie_home__ = {__marie_home__}")
    logger.info(f"__cache_path__ = {__cache_path__}")
    logger.info(f"yml_config = {yml_config}")
    logger.info(f"env = {env}")
    logger.info(f"CONTEXT.gpu_device_count = {gpu_device_count()}")

    # load env file
    if not env_file:
        env_file = os.path.join(__config_dir__, ".env")

    load_env_file(dotenv_path=env_file)

    context = {
        "gpu_device_count": gpu_device_count(),
    }

    # put env variables into context
    if env:
        for k, v in env.items():
            context[k] = v
            os.environ[k] = v

    # dump environment variables
    if "DUMP_ENV" in os.environ:
        for k, v in os.environ.items():
            print(f"{k} = {v}")

    # Load the config file and setup the toast events
    config = load_yaml(yml_config, substitute=True, context=context)

    f = Flow.load_config(
        config,
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
        substitute=True,
        context=context,
        include_gateway=False,
        # noblock_on_start=True,
    )

    marie.helper.extend_rest_interface = extend_rest_interface
    filter_endpoint()
    setup_server(config)

    with f:
        f.block()


def setup_server(config: Dict[str, Any]) -> None:
    setup_toast_events(config.get("toast", {}))
    setup_storage(config.get("storage", {}))
    setup_scheduler(config.get("scheduler", {}))


def filter_endpoint():
    """
    Filter out dry_run endpoint from uvicorn logs
    :return:
    """
    import logging

    # filter out dry_run endpoint from uvicorn logs

    class _EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("GET /dry_run") == -1

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
