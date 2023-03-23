import inspect
import os
import sys
import traceback
from typing import Dict, Any, Optional
import torch

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
    AmazonMQToastHandler,
    PsqlToastHandler,
)
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.device import gpu_device_count
from rich.traceback import install

from marie_server.rest_extension import extend_rest_interface

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def setup_toast_events(toast_config: Dict[str, Any]):
    native_config = toast_config["native"]
    psql_config = toast_config["psql"]
    amazon_config = toast_config["amazon-mq"]

    Toast.register(NativeToastHandler("/tmp/events.json"), native=True)
    Toast.register(PsqlToastHandler(psql_config), native=False)
    Toast.register(AmazonMQToastHandler(amazon_config), native=False)


def setup_storage(storage_config: Dict[str, Any]):
    """Setup the storage handler"""
    if "s3" in storage_config:
        handler = S3StorageHandler(config=storage_config["s3"], prefix="S3_")

        # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
        StorageManager.register_handler(handler=handler)
        StorageManager.ensure_connection("s3://")


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=dotenv_path, verbose=True)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("exc_type", exc_type)
    logger.error("exc_value", exc_value)
    logger.error("exc_traceback", exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


def main(yml_config: str, env: Dict[str, str], env_file: str):
    # install handler for exceptions
    sys.excepthook = handle_exception
    install(show_locals=True)
    logger.info(f"Starting marie server : {__version__}")

    if "NO_VERSION_CHECK" not in os.environ:
        from marie_server.helper import is_latest_version

        is_latest_version(github_repo="marie-ai")

    from marie import Flow

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
    for k, v in os.environ.items():
        print(f"{k} = {v}")

    # Load the config file and setup the toast events
    config = load_yaml(yml_config, substitute=True, context=context)

    setup_toast_events(config.get("toast", {}))
    setup_storage(config.get("storage", {}))

    marie.helper.extend_rest_interface = extend_rest_interface

    f = Flow.load_config(
        config,
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
        substitute=True,
        context=context,
    )

    with f:
        f.block()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-i":
            _input = sys.stdin.read()
        else:
            _input = sys.argv[1]
    else:
        _input = os.path.join(__config_dir__, "service", "marie.yml")

    main(_input)
