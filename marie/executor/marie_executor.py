from datetime import datetime

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.constants import __default_endpoint__
from marie.logging_core.logger import MarieLogger
from marie.serve.executors import __dry_run_endpoint__

try:
    import pynvml

    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False


class MarieExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__)

    # @requests(on=__default_endpoint__)
    async def default_endpoint(
        self,
        docs: DocList[TextDoc],
        parameters=None,
        *args,
        **kwargs,
    ):
        """
        Default endpoint to be executed in asynchronous mode
        :param docs:
        :param parameters:
        :param args:
        :param kwargs:
        :return:
        """
        print("Default endpoint called")
        return docs

    @requests(on=__dry_run_endpoint__)
    async def dry_run_func(
        self,
        docs: DocList[TextDoc],
        parameters=None,
        *args,
        **kwargs,
    ):
        """
         DryRun function to be executed in asynchronous mode
        :param docs:
        :param parameters:
        :param args:
        :param kwargs:
        """
        print(
            f"DryRun(custom) func called : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
