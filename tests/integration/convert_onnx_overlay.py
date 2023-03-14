import os
from timeit import timeit

import numpy as np
import onnxruntime
import psutil
import torch

from marie.models.pix2pix.models import create_model
from marie.models.pix2pix.options.test_options import TestOptions

# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# Docs
# https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
# https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/


def run_onnx_inference(model_path, input_data):
    # ONNX model
    sess_options = onnxruntime.SessionOptions()
    sess_options.add_session_config_entry('session.load_model_format', 'ONNX')
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    providers = [
        (
            'CUDAExecutionProvider',
            {
                'device_id': 0,
                'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
            },
        ),
        # 'CPUExecutionProvider',
    ]

    # ONNX model
    onnx_model = onnxruntime.InferenceSession(
        model_path,
        sess_options,
        providers
        # providers=[
        #     # 'TensorrtExecutionProvider',
        #     'CUDAExecutionProvider',
        #     # 'CPUExecutionProvider',
        # ],
    )

    print("Running inference")
    # print all the inputs and outputs in the model
    print("The model expects names: ", onnx_model.get_inputs()[0].name)
    print("The model expects input shape: ", onnx_model.get_inputs()[0].shape)
    print("The model outputs shape: ", onnx_model.get_outputs()[0].shape)

    data = np.random.rand(1, 3, 256, 256).astype(np.float32)
    ort_outputs = onnx_model.run(None, {onnx_model.get_inputs()[0].name: data})

    print(ort_outputs[0].shape)


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

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        f=os.path.expanduser("/tmp/latest_net_G.onnx"),
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
    )

    # https://github.com/huggingface/transformers/pull/17953
    # export
    # torch.onnx.export(
    #     model,
    #     tuple(input_dict.values()),
    #     f="/tmp/torch-model.onnx",
    #     input_names=['input_ids', 'attention_mask'],
    #     output_names=['logits'],
    #     dynamic_axes={
    #         'input_ids': {0: 'batch_size', 1: 'sequence'},
    #         'attention_mask': {0: 'batch_size', 1: 'sequence'},
    #         'logits': {0: 'batch_size', 1: 'sequence'},
    #     },
    #     do_constant_folding=True,
    #     opset_version=13,
    # )


def load_torch_model(load_path: str, device: torch.device):
    model_pix = build_model()
    model = model_pix.netG
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(16)
    # export()

    # run_onnx_inference("/tmp/latest_net_G.onnx", _generate_dummy_images()[0])
    run_onnx_inference("/tmp/latest_net_G.onnx", torch.randn(1, 3, 256, 256))
