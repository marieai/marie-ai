import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import psutil
import torch
from onnxruntime import ExecutionMode
from torch import tensor

logger = logging.getLogger(__name__)

# TODO: Try a better workaround to lazy import tensorrt package.
tensorrt_imported = False
if not tensorrt_imported:
    try:
        import tensorrt  # Unused but required by TensorrtExecutionProvider

        tensorrt_imported = True
    except:
        # We silently omit the import failure here to avoid overwhelming warning messages in case of multi-gpu.
        tensorrt_imported = False


def onnx_get_dynamic_axes(input_keys: List[str]):
    dynamic_axes = {}
    for k in input_keys:
        if "token_ids" in k or "segment_ids" in k:
            dynamic_axes[k] = {0: "batch_size", 1: "seq_length"}
        elif (
            "valid_length" in k
            or k.startswith("numerical")
            or k.startswith("timm_image")
        ):
            dynamic_axes[k] = {0: "batch_size"}

    return dynamic_axes


def get_provider_name(provider_config: Union[str, tuple]) -> str:
    if isinstance(provider_config, tuple):
        provider_name = provider_config[0]
    else:
        assert isinstance(
            provider_config, str
        ), "input provider config is expected to be either str or tuple"
        provider_name = provider_config
    return provider_name


class OnnxModule(object):
    """
    OnnxModule is as a replacement of torch.nn.Module for running forward pass with onnxruntime.

    The module can be generated with MultiModalPredictor.export_tensorrt(),
    so that we can predict with TensorRT by simply replacing predictor._model with OnnxModule.

    ref : https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu
    ref : https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/icess22.pdf
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Union[dict, List[str]]] = None,
        use_io_binding: bool = True,
    ):
        """
        Parameters
        ----------
        onnx_path : str
            The file path of the onnx model that need to be executed in onnxruntime.
        providers : dict or str, default=None
            A list of execution providers for model prediction in onnxruntime.
        """
        import onnx
        import onnxruntime as ort

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"failed to located onnx file at {onnx_path}")

        logger.info(
            f"Loading ONNX[io_binding = {use_io_binding}] file from path {onnx_path}..."
        )
        onnx_model = onnx.load(onnx_path)

        # ONNX model
        sess_options = ort.SessionOptions()
        sess_options.add_session_config_entry("session.load_model_format", "ONNX")
        sess_options.execution_mode = ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        sess_options.log_verbosity_level = 1
        sess_options.enable_profiling = False
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # https://github.com/microsoft/onnxruntime/issues/15002
        if providers == None:
            dirname = os.path.dirname(os.path.abspath(onnx_path))
            cache_path = os.path.join(dirname, "model_trt")
            providers = [
                # (
                #     "TensorrtExecutionProvider",
                #     {
                #         "device_id": 0,
                #         # "trt_max_workspace_size": 2147483648,
                #         "trt_fp16_enable": True,
                #         "trt_engine_cache_path": cache_path,
                #         "trt_engine_cache_enable": True,
                #     },
                # ),
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",  # EXHAUSTIVE
                        "cudnn_conv_use_max_workspace": True,  # Reduces inference time by ~25%
                        "do_copy_in_default_stream": True,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "enable_cuda_graph": False,
                        "cudnn_conv1d_pad_to_nc1d": True,
                        # "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
                    },
                    # "CUDAExecutionProvider",
                    # {
                    #     "device_id": 0,
                    #     "arena_extend_strategy": "kNextPowerOfTwo",
                    #     # "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    #     "cudnn_conv_algo_search": "EXHAUSTIVE",
                    #     "do_copy_in_default_stream": True,
                    # },
                ),
                ("CPUExecutionProvider", {}),
            ]

        if (
            len(providers) == 1
            and get_provider_name(providers[0]) == "TensorrtExecutionProvider"
        ):
            if not tensorrt_imported:
                raise ImportError(
                    "tensorrt package is not installed. The package can be install via `pip install tensorrt`."
                )

        self.session = ort.InferenceSession(
            onnx_model.SerializeToString(), sess_options, providers=providers
        )

        if (
            get_provider_name(providers[0]) == "TensorrtExecutionProvider"
            and tensorrt_imported
        ):
            assert "TensorrtExecutionProvider" in self.session.get_providers(), (
                f"unexpected TensorRT compilation failure: TensorrtExecutionProvider not in providers ({self.session.get_providers()}). "
                "Make sure onnxruntime package gets lazy imported everywhere."
            )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [i.name for i in inputs]
        self.output_names = [i.name for i in outputs]
        self.io_binding = self.session.io_binding() if use_io_binding else None

    def __call__(self, *args):
        """
        Make the module callable like torch.nn.Module, while runnning forward pass with onnxruntime.

        https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession.run
        https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/bert_perf_test.py#L143

        Parameters
        ----------
        args : list of torch.Tensor
            A list of torch.Tensor that are inputs of the model.

        Returns
        -------
        onnx_outputs : list of torch.Tensor
            A list of torch.Tensor that are outputs of the model.
        """
        import torch

        session = self.session
        if self.io_binding is not None:
            io_binding = self.io_binding

            for i, k in enumerate(self.input_names):
                io_binding.bind_cpu_input(k, args[i])

            for i, k in enumerate(self.output_names):
                io_binding.bind_output(k)

            session.run_with_iobinding(io_binding)
            onnx_outputs = io_binding.copy_outputs_to_cpu()
            onnx_outputs = [torch.from_numpy(out) for out in onnx_outputs]

            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

            return onnx_outputs
        else:

            input_dict = {k: args[i] for i, k in enumerate(self.input_names)}
            onnx_outputs = session.run(self.output_names, input_dict)
            onnx_outputs = onnx_outputs[:3]
            onnx_outputs = [torch.from_numpy(out) for out in onnx_outputs]

            return onnx_outputs

    def to(self, *args):
        """A dummy function that act as torch.nn.Module.to() function"""

        class DummyModel:
            def eval():
                pass

        return DummyModel


def create_input_output_tensors(inputs, outputs, device):
    input_tensors = {
        name: torch.from_numpy(array).to(device) for name, array in inputs.items()
    }
    output_tensors = {
        name: torch.from_numpy(array).to(device) for name, array in outputs.items()
    }
    return input_tensors, output_tensors
