from __future__ import print_function

import os
import timeit
from typing import Optional

import numpy as np
import onnx as onnx
import onnxruntime as onnxruntime
import psutil
import torch
from PIL import Image
from onnxruntime import ExecutionMode
from onnxruntime.quantization import quantize_dynamic, QuantType

from marie.models.pix2pix.models import create_model
from marie.models.pix2pix.options.test_options import TestOptions
from marie.utils.onnx import OnnxModule

# TODO: Try a better workaround to lazy import tensorrt package.
tensorrt_imported = False
if not tensorrt_imported:
    try:
        import tensorrt  # Unused but required by TensorrtExecutionProvider

        tensorrt_imported = True
    except:
        # We silently omit the import failure here to avoid overwhelming warning messages in case of multi-gpu.
        tensorrt_imported = False

print(tensorrt_imported)

# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# Docs
# https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
# https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/
# http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/gyexamples/plot_benchmark_graph_opt.html
# https://medium.com/axinc-ai/using-the-onnx-official-optimizer-27d1c7da3531
# https://github.com/microsoft/OLive
# https://nietras.com/2021/01/25/onnxruntime/


def run_onnx_inference_as_module(model_path: str, image_path: str) -> None:
    model_path = os.path.expanduser(model_path)
    # create onnx module for evaluation
    model = OnnxModule(model_path, providers=None)
    print(model)

    # Input > N x C x W x H
    # read image and add batch dimension
    img = Image.open(image_path).convert("RGB")
    # make sure image is divisible by 32
    print("Image size: ", img.size)

    ow, oh = img.size
    base = 32  # image size must be divisible by 32
    if ow % base != 0 or oh % base != 0:
        h = oh // base * base + base
        w = ow // base * base + base
        img = img.resize((w, h), Image.LANCZOS)

    print("Image size After: ", img.size)
    data = np.array(img).astype(np.float32)
    # convert from HWC to CHW
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)  # add batch dimension  N x C x W x H

    starttime = timeit.default_timer()
    # data = np.random.rand(1, 3, 512, 512).astype(np.float32)
    # Output > N x C x W x H
    outputs = model(data)
    print("Batch size is :", outputs[0].shape[0])
    print("Inference time is :", timeit.default_timer() - starttime)
    batch_output = outputs[0][0]  # get the first batch
    # tensor to numpy
    image_numpy = batch_output.detach().cpu().numpy()
    # convert from CHW to HWC and scale from [-1, 1] to [0, 255]
    image_numpy = (
        (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    )  # post-processing: transpose and scaling

    # convert to pillow image
    img = Image.fromarray(image_numpy.astype("uint8")).convert("RGB")
    img.save(f"/tmp/onnx_test.png")
    print("Eval time is :", timeit.default_timer() - starttime)


def run_onnx_inference(model_path, input_data):
    model_path = os.path.expanduser(model_path)

    # ONNX model
    sess_options = onnxruntime.SessionOptions()
    sess_options.add_session_config_entry("session.load_model_format", "ONNX")
    sess_options.execution_mode = ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    sess_options.log_verbosity_level = 1
    print("Available providers: ", onnxruntime.get_available_providers())

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # enable model serialization  and using TensorRT for inference will results in an error
    # FAIL : Unable to serialize model as it contains compiled nodes. Please disable any execution providers which generate compiled nodes
    if False:
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = (
            "/tmp/optimized_model.onnx"  # ~ 5% faster while using this option
        )

    # https://onnxruntime.ai/docs/performance/tune-performance.html#tips-for-tuning-performance
    onnx_path = model_path
    dirname = os.path.dirname(os.path.abspath(onnx_path))
    cache_path = os.path.join(dirname, "model_trt")

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": 2147483648,
                "trt_fp16_enable": True,
                "trt_engine_cache_path": cache_path,
                "trt_engine_cache_enable": True,
            },
        ),
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",  # EXHAUSTIVE
                "cudnn_conv_use_max_workspace": True,  # Reduces inference time by ~25%
                "do_copy_in_default_stream": True,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv1d_pad_to_nc1d": True,
                # "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
            },
        ),
    ]

    # ONNX model
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    # bind = SessionIOBinding(session._sess)
    print("session providers: ", session.get_providers())

    print("Running inference")
    # print all the inputs and outputs in the model
    print("The model expects names: ", session.get_inputs()[0].name)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    print("The model outputs shape: ", session.get_outputs()[0].shape)

    # Input > N x C x W x H
    # read image and add batch dimension
    img = Image.open("/home/gbugaj/sample.png").convert("RGB")
    # img = Image.open("/tmp/segment-1024x-2048.png").convert("RGB")
    # make sure image is divisible by 32
    print("Image size: ", img.size)

    ow, oh = img.size
    base = 32  # image size must be divisible by 32
    if ow % base != 0 or oh % base != 0:
        h = oh // base * base + base
        w = ow // base * base + base
        img = img.resize((w, h), Image.LANCZOS)

    print("Image size After: ", img.size)
    data = np.array(img).astype(np.float32)
    # convert from HWC to CHW
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)  # add batch dimension

    # time the inference
    # timed = timeit(lambda: onnx_model.run(None, {onnx_model.get_inputs()[0].name: data}), number=10)
    # print("Time: ", timed)

    for i in range(10):
        starttime = timeit.default_timer()
        # data = np.random.rand(1, 3, 512, 512).astype(np.float32)
        # Output > N x C x W x H
        input_dict = {session.get_inputs()[0].name: data}
        outputs = session.run(None, input_dict)
        print("Inference time is :", timeit.default_timer() - starttime)
        image_numpy = outputs[0][0]  # get the first batch
        # convert from CHW to HWC and scale from [-1, 1] to [0, 255]
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling

        # convert to pillow image
        img = Image.fromarray(image_numpy.astype("uint8")).convert("RGB")
        img.save(f"/tmp/onnx_test_{i}.png")

        print("Eval time is :", timeit.default_timer() - starttime)


def _generate_dummy_images(
    batch_size: int = 1,
    num_channels: int = 3,
    image_height: int = 256,
    image_width: int = 256,
):
    images = []
    for _ in range(batch_size):
        data = np.random.rand(image_height, image_width, num_channels) * 255
        # images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
        images.append(data.astype("uint8"))
    return images


def build_model():
    gpu_id = "0"
    args = [
        "--dataroot",
        "./data",
        "--name",
        "claim_mask",
        "--model",
        "test",
        "--netG",
        "local",
        "--direction",
        "AtoB",
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--gpu_id",
        gpu_id,
        "--norm",
        "instance",
        "--preprocess",
        "none",
        "--checkpoints_dir",
        "./model_zoo/overlay",
        "--ngf",
        "64",
        "--ndf",
        "64",
        "--no_dropout",
    ]

    opt = TestOptions().parse(args)
    # hard-code parameters for test
    opt.eval = True
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_dropout = False
    opt.display_id = -1
    opt.output_nc = 3  # Need to build model for BITONAL images only so we could output 1 chanell only
    model = create_model(opt)

    return model


def export_onnx(
    path: str,
    output_path_fp32: str,
    verbose: Optional[bool] = True,
    opset_version: Optional[int] = 17,
):
    print("Exporting model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = path
    load_path = os.path.expanduser(load_path)
    model = load_torch_model(load_path, device)

    print(model)
    inputs = dict(
        input=torch.randn(1, 3, 256, 256).to(device),
        # input =  _generate_dummy_images()
    )

    model_fp32 = output_path_fp32
    torch.onnx.export(
        model,  # model being run
        tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
        f=os.path.expanduser(
            model_fp32
        ),  # where to save the model (can be a file or file-like object)
        verbose=verbose,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        do_constant_folding=True,  # whether to execute constant folding for optimization
        export_params=True,
        opset_version=opset_version,
        dynamic_axes={
            # "input": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            # "output": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )

    if False:
        model_quant = "/tmp/latest_net_G.quant.onnx"
        quantized_model = quantize_dynamic(
            model_fp32,
            model_quant,
            weight_type=QuantType.QUInt8,
            per_channel=True,
            reduce_range=True,
        )  # chnage QInt8 to QUInt8)

        # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/run.py
        print("Calibrated and quantized model saved.")
        print("Done exporting model")


def load_torch_model(load_path: str, device: torch.device):
    model_pix = build_model()
    model = model_pix.netG
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    print("loading the model from %s" % load_path)
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model = model.eval()

    return model


def optimize(src_onnx, opt_onnx):
    # pip3 install onnxoptimizer
    import onnx
    import onnxoptimizer as onnxopt

    src_onnx = os.path.expanduser(src_onnx)
    opt_onnx = os.path.expanduser(opt_onnx)
    # load model
    model = onnx.load(src_onnx)
    # optimize
    # model = onnxopt.optimize(model, ["fuse_bn_into_conv"])
    model = onnxopt.optimize(model)

    # save optimized model
    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())


def test_trn_backend(model_path):
    import onnx
    import onnx_tensorrt.backend as backend
    import numpy as np

    model = onnx.load(model_path)
    engine = backend.prepare(model, device='CUDA:0')
    input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
    output_data = engine.run(input_data)[0]
    # print(output_data)
    print(output_data.shape)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(3)
    # import tensorrt as trt

    print("Torch version:", torch.__version__)
    print("ONNX version:", onnx.__version__)
    print("ONNX Runtime version:", onnxruntime.__version__)
    print("CUDA version:", torch.version.cuda)
    # print("TensorRT version: {}".format(trt.__version__))
    print("CUDNN version:", torch.backends.cudnn.version())
    providers = onnxruntime.get_available_providers()
    print("Available providers:", providers)

    import torch._dynamo as dynamo

    print(dynamo.list_backends(None))
    # test_trn_backend("/tmp/latest_net_G.onnx")

    if True:
        export_onnx(
            path="~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/latest_net_G.pth",
            output_path_fp32="~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/model.onnx",
        )

    if True:
        optimize(
            "~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/model.onnx",
            "~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/model.optimized.onnx",
        )

    # optimize("/tmp/latest_net_G.onnx", "/tmp/latest_net_G.opt.onnx")

    if False:
        run_onnx_inference(
            "~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/latest_net_G.optimized.onnx",
            torch.randn(1, 3, 256, 256),
        )

    if False:
        run_onnx_inference_as_module(
            "~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/latest_net_G.optimized.onnx",
            "/home/gbugaj/sample.png",
        )

    # run_onnx_inference("/tmp/latest_net_G.opt.onnx", torch.randn(1, 3, 256, 256))
    # run_onnx_inference("/tmp/latest_net_G.quant.onnx", torch.randn(1, 3, 256, 256))
