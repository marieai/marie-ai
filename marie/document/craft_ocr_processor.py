import os
import sys
import typing
from typing import Callable

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from marie.constants import __model_path__
from marie.document.ocr_processor import OcrProcessor
from marie.lang import Object
from marie.logging.logger import MarieLogger
from marie.logging.profile import TimeContext
from marie.models.icr.dataset import AlignCollate
from marie.models.icr.memory_dataset import MemoryDataset
from marie.models.icr.model import Model
from marie.models.icr.utils import AttnLabelConverter, CTCLabelConverter
from marie.models.utils import torch_gc

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class CraftOcrProcessor(OcrProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/icr",
        models_dir: str = os.path.join(__model_path__, "icr"),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(work_dir, cuda, **kwargs)
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.logger.info("CRAFT ICR processor [cuda={}]".format(cuda))

        saved_model = os.path.join(
            models_dir,
            "TPS-ResNet-BiLSTM-Attn-case-sensitive-ft",
            "best_accuracy.pth",
        )

        if cuda and not torch.cuda.is_available():
            raise Exception("CUDA specified but no cuda devices found ")

        self.device = "cuda" if cuda else "cpu"

        if True:
            opt = Object()
            opt.Transformation = "TPS"
            opt.FeatureExtraction = "ResNet"
            opt.SequenceModeling = "BiLSTM"
            opt.Prediction = "Attn"
            opt.saved_model = saved_model
            opt.sensitive = True
            opt.imgH = 32
            opt.imgW = 100
            opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
            opt.rgb = False
            opt.num_fiducial = 20
            opt.input_channel = 1
            opt.output_channel = 512
            opt.hidden_size = 256
            opt.batch_max_length = 48
            opt.batch_size = 2  # FIXME: setting batch size to 1 will cause "TypeError: forward() missing 2 required positional arguments: 'input' and 'text'"
            opt.PAD = True
            opt.workers = 4
            opt.num_gpu = -1
            opt.image_folder = "./"

        if False:
            opt = Object()
            opt.Transformation = "TPS"
            opt.FeatureExtraction = "ResNet"
            opt.SequenceModeling = "BiLSTM"
            opt.Prediction = "Attn"
            opt.saved_model = "./models/icr/TPS-ResNet-BiLSTM-Attn/best_accuracy.pth"
            # opt.saved_model = './models/icr/TPS-ResNet-BiLSTM-Attn-case-sensitive/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'
            opt.saved_model = (
                "./models/icr/TPS-ResNet-BiLSTM-Attn/TPS-ResNet-BiLSTM-Attn.pth"
            )
            opt.sensitive = False
            opt.imgH = 32
            opt.imgW = 100
            opt.character = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            # opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
            opt.num_fiducial = 20
            opt.input_channel = 1
            opt.output_channel = 512
            opt.hidden_size = 256
            opt.batch_max_length = 25
            opt.batch_size = 2  # Fixme: setting batch size to 1 will cause "TypeError: forward() missing 2 required positional arguments: 'input' and 'text'"
            opt.PAD = True
            opt.rgb = False
            opt.workers = 4
            opt.num_gpu = -1
            opt.image_folder = "./"

        self.opt = opt
        self.converter, self.model = self.__load()

    def is_available(self) -> bool:
        return self.model is not None

    def __load(self):
        """model configuration"""
        opt = self.opt

        if "CTC" in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print(
            "model input parameters",
            opt.imgH,
            opt.imgW,
            opt.num_fiducial,
            opt.input_channel,
            opt.output_channel,
            opt.hidden_size,
            opt.num_class,
            opt.batch_max_length,
            opt.Transformation,
            opt.FeatureExtraction,
            opt.SequenceModeling,
            opt.Prediction,
        )

        # Somehow the model in being still loaded on GPU
        # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html

        # GPU only
        model = model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=None).to(self.device)
        model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))
        model = self.optimize_model(model)

        if False:

            class WrappedModel(torch.nn.Module):
                def __init__(self, module):
                    super(WrappedModel, self).__init__()
                    self.module = module  # that I actually define.

                def forward(self, x):
                    return self.module(x)

                    # CPU

            model = WrappedModel(model)
            model = model.to(self.device)
            state_dict = torch.load(opt.saved_model, map_location=self.device)
            model.load_state_dict(state_dict)

        return converter, model

    def optimize_model(self, model: nn.Module) -> Callable | nn.Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model [craft]", logger=self.logger):
                import torch._dynamo as dynamo

                # ['aot_eager', 'aot_eager_decomp_partition', 'aot_torchxla_trace_once', 'aot_torchxla_trivial', 'aot_ts', 'aot_ts_nvfuser', 'cudagraphs', 'dynamo_accuracy_minifier_backend', 'dynamo_minifier_backend', 'eager', 'inductor', 'ipex', 'nvprims_aten', 'nvprims_nvfuser', 'onnxrt', 'torchxla_trace_once', 'torchxla_trivial', 'ts', 'tvm']
                torch._dynamo.config.verbose = False
                torch._dynamo.config.suppress_errors = True
                # torch.backends.cudnn.benchmark = True
                # https://dev-discuss.pytorch.org/t/torchinductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874
                model = torch.compile(
                    model, backend="inductor", mode="default", fullgraph=False
                )
                return model
        except Exception as err:
            self.logger.warning(f"Model compile not supported: {err}")
            return model

    def recognize_from_fragments(
        self, images, **kwargs
    ) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """
        try:
            # debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr',key,'debug'))
            # output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr',key,'output'))

            opt = self.opt
            model = self.model
            converter = self.converter
            opt.batch_size = 64  #

            # setup data
            AlignCollate_data = AlignCollate(
                imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
            )
            eval_data = MemoryDataset(images=images, opt=opt)

            eval_loader = torch.utils.data.DataLoader(
                eval_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_data,
                pin_memory=True,
            )

            results = []
            # predict
            model.eval()
            with torch.no_grad():
                for image_tensors, image_labels in eval_loader:

                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(self.device)

                    # For max length prediction
                    length_for_pred = torch.IntTensor(
                        [opt.batch_max_length] * batch_size
                    ).to(self.device)
                    text_for_pred = (
                        torch.LongTensor(batch_size, opt.batch_max_length + 1)
                        .fill_(0)
                        .to(self.device)
                    )

                    if "CTC" in opt.Prediction:
                        preds = model(image, text_for_pred)
                        # Select max probabilty (greedy decoding) then decode index to character
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        _, preds_index = preds.max(2)
                        # preds_index = preds_index.view(-1)
                        preds_str = converter.decode(preds_index, preds_size)

                    else:
                        preds = model(image, text_for_pred, is_train=False)
                        # select max probabilty (greedy decoding) then decode index to character
                        _, preds_index = preds.max(2)
                        preds_str = converter.decode(preds_index, length_for_pred)

                    dashed_line = "-" * 120
                    head = f'{"key":25s}\t{"predicted_labels":32s}\tconfidence score'

                    print(f"{dashed_line}\n{head}\n{dashed_line}")

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    try:
                        for img_name, pred, pred_max_prob in zip(
                            image_labels, preds_str, preds_max_prob
                        ):
                            if "Attn" in opt.Prediction:
                                pred_EOS = pred.find("[s]")
                                pred = pred[
                                    :pred_EOS
                                ]  # prune after "end of sentence" token ([s])
                                pred_max_prob = pred_max_prob[:pred_EOS]

                            # calculate confidence score (= multiply of pred_max_prob)
                            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                            # get value from the TensorFloat
                            confidence = confidence_score.item()
                            text = pred.upper() if pred is not None else ""
                            results.append(
                                {"confidence": confidence, "text": text, "id": img_name}
                            )
                            print(
                                f"{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}"
                            )
                    except Exception as ex:
                        self.logger.error(f"Error processing image: {ex}")
                        results.append({"confidence": 0, "text": "", "id": img_name})
        except Exception as ex:
            raise ex

        torch_gc()
        return results
