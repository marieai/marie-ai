from datetime import datetime
from typing import Any, Dict

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.logging_core.logger import MarieLogger
from marie.serve.executors import __dry_run_endpoint__
from marie.utils.server_runtime import setup_storage, setup_toast_events

try:
    import pynvml

    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False


class MarieExecutor(Executor):
    """Base executor class for Marie AI framework providing core functionality.

    This executor handles basic setup, configuration management and provides
    default endpoints for execution and dry runs. It supports toast notifications,
    storage configuration and GPU monitoring capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("MarieExecutor initialized")
        self.logger.info(f"Kwargs : {kwargs}")

        config = {
            "toast": kwargs.get("toast", {}),
            "storage": kwargs.get("storage", {}),
        }
        self.setup_executor(config)

    def setup_executor(self, config: Dict[str, Any]) -> None:
        """Configure and initialize the executor with provided settings.

        Args:
            config: Configuration dictionary containing toast notifications and storage settings
        """
        setup_toast_events(config.get("toast", {}))
        setup_storage(config.get("storage", {}))

    # @requests(on=__default_endpoint__)
    async def default_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any] = None,
        *args,
        **kwargs,
    ) -> DocList[TextDoc]:
        """Process documents through the default execution endpoint.

        Args:
            docs: List of text documents to process
            parameters: Optional execution parameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed document list
        """
        self.logger.debug("Default endpoint called")
        return docs

    @requests(on=__dry_run_endpoint__)
    async def dry_run_func(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any] = None,
        *args,
        **kwargs,
    ) -> None:
        """Execute a dry run to verify system configuration and GPU availability.

        Args:
            docs: List of text documents (not processed in dry run)
            parameters: Optional execution parameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Raises:
            RuntimeError: If any errors occur during the dry run
        """
        self.logger.info(
            f"Starting dry run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        has_error = False
        message = ""
        if _pynvml_exist:
            try:
                import pynvml
                from pynvml import NVMLError, nvmlSystemGetDriverVersion

                pynvml.nvmlInit()
                self.logger.debug("Driver Version:", nvmlSystemGetDriverVersion())
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.logger.debug(
                        "Device", i, ":", pynvml.nvmlDeviceGetName(handle)
                    )
                    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    remaining = info.free
                    self.logger.debug("Remaining memory:", remaining)
            except Exception as error:
                self.logger.error(f"Error in DryRun: {error}")
                has_error = True
                message = str(error)

        if has_error:
            raise RuntimeError(f"Error in DryRun : {message}")
