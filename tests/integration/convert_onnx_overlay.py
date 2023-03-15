import os

import numpy as np
import onnxruntime
import onnx
from onnxruntime import ExecutionMode, SessionIOBinding
from onnxruntime.quantization import quantize_dynamic, QuantType
import psutil
import torch
from PIL import Image
import timeit
import random

from marie.models.pix2pix.data.base_dataset import __make_power_2
from marie.models.pix2pix.models import create_model
from marie.models.pix2pix.options.test_options import TestOptions
from marie.models.pix2pix.util.util import tensor2im


# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# Docs
# https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
# https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/
# http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/gyexamples/plot_benchmark_graph_opt.html
# https://medium.com/axinc-ai/using-the-onnx-official-optimizer-27d1c7da3531
# https://github.com/microsoft/OLive
# https://nietras.com/2021/01/25/onnxruntime/


def run_onnx_inference(model_path, input_data):
    # ONNX model
    sess_options = onnxruntime.SessionOptions()
    sess_options.add_session_config_entry("session.load_model_format", "ONNX")
    sess_options.execution_mode = ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = (
        "/tmp/optimized_model.onnx"  # ~ 5% faster while using this option
    )

    # CPUExecutionProvider
    # TensorrtExecutionProvider
    # CUDAExecutionProvider
    # DnnlExecutionProvider
    # https://onnxruntime.ai/docs/performance/tune-performance.html#tips-for-tuning-performance
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",  # Reduces inference time by ~25%
                "cudnn_conv_use_max_workspace": True,  # Reduces inference time by ~25%
                "do_copy_in_default_stream": True,
                "arena_extend_strategy": "kSameAsRequested", #"kNextPowerOfTwo",
                "cudnn_conv1d_pad_to_nc1d": True,
                # "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
            },
        ),
    ]

    # model_path = "/tmp/optimized_model.onnx"
    # ONNX model
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    bind = SessionIOBinding(session._sess)

    print("Running inference")
    # print all the inputs and outputs in the model
    print("The model expects names: ", session.get_inputs()[0].name)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    print("The model outputs shape: ", session.get_outputs()[0].shape)

    # Input > N x C x W x H
    # read image and add batch dimension
    img = Image.open("/tmp/segment.png").convert("RGB")
    # img = Image.open("/tmp/segment-1024x-2048.png").convert("RGB")
    # make sure image is divisible by 32
    print("Image size: ", img.size)
    # img = __make_power_2(img, base=4, method=Image.LANCZOS)

    ow, oh = img.size
    base = 8  # image size must be divisible by 32
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
        outputs = session.run(None, {session.get_inputs()[0].name: data})
        print("Inference time is :", timeit.default_timer() - starttime)
        image_numpy = outputs[0][0]  # get the first batch
        # convert from CHW to HWC and scale from [-1, 1] to [0, 255]
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling

        # convert to pillow image
        # img = Image.fromarray(image_numpy.astype("uint8")).convert("RGB")
        # img.save(f"/tmp/onnx_test_{i}.png")

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


def export():
    print("Exporting model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = "~/dev/marieai/marie-ai/model_zoo/overlay/claim_mask/latest_net_G.pth"
    load_path = os.path.expanduser(load_path)
    model = load_torch_model(load_path, device)

    print(model)
    inputs = dict(
        input=torch.randn(1, 3, 256, 256).to(device),
        # input =  _generate_dummy_images()
    )

    model_fp32 = "/tmp/latest_net_G.onnx"
    torch.onnx.export(
        model,  # model being run
        tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
        f=os.path.expanduser(
            model_fp32
        ),  # where to save the model (can be a file or file-like object)
        verbose=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        do_constant_folding=True,  # whether to execute constant folding for optimization
        export_params=True,
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            "output": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        },
    )

    model_quant = "/tmp/latest_net_G.quant.onnx"
    quantized_model = quantize_dynamic(
        model_fp32, model_quant, weight_type=QuantType.QUInt8
    )  # chnage QInt8 to QUInt8)
    print("Done exporting model")
    print(quantized_model)


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

    # load model
    model = onnx.load(src_onnx)

    # optimize
    model = onnxopt.optimize(model, ["fuse_bn_into_conv"])

    # save optimized model
    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(16)
    # export()
    # optimize("/tmp/latest_net_G.onnx", "/tmp/latest_net_G.opt.onnx")
    # run_onnx_inference("/tmp/latest_net_G.onnx", torch.randn(1, 3, 256, 256))

    # run_onnx_inference("/tmp/latest_net_G.opt.onnx", torch.randn(1, 3, 256, 256))
    run_onnx_inference("/tmp/latest_net_G.op2.onnx", torch.randn(1, 3, 256, 256))

    # run_onnx_inference("/tmp/latest_net_G.quant.onnx", torch.randn(1, 3, 256, 256))
