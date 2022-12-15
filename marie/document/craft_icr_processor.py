import os
import sys
import typing

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from marie.document.icr_processor import IcrProcessor
from marie.lang import Object
from marie.models.icr.dataset import AlignCollate, RawDataset
from marie.models.icr.memory_dataset import MemoryDataset
from marie.models.icr.model import Model
from marie.models.icr.utils import AttnLabelConverter, CTCLabelConverter

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CraftIcrProcessor(IcrProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/icr",
        models_dir: str = "./model_zoo/icr",
        cuda: bool = True,
    ) -> None:
        super().__init__(work_dir, cuda)
        print("CRAFT ICR processor [cuda={}]".format(cuda))

        saved_model = os.path.join(
            models_dir, "TPS-ResNet-BiLSTM-Attn-case-sensitive-ft", "best_accuracy.pth"
        )

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

        cudnn.benchmark = True
        cudnn.deterministic = True

    def __load(self):
        """model configuration"""
        opt = self.opt

        if "CTC" in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        print("Evaluating on device : %s" % (device))
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
        model = torch.nn.DataParallel(model, device_ids=None).to(device)
        # load model
        print("loading pretrained model from %s" % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        if False:

            class WrappedModel(torch.nn.Module):
                def __init__(self, module):
                    super(WrappedModel, self).__init__()
                    self.module = module  # that I actually define.

                def forward(self, x):
                    return self.module(x)

                    # CPU

            model = WrappedModel(model)
            model = model.to(device)
            state_dict = torch.load(opt.saved_model, map_location=device)
            model.load_state_dict(state_dict)

        return converter, model

    def recognize_from_fragments(
        self, images, **kwargs
    ) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        print("ICR processing : recognize_from_boxes via boxes")
        try:
            # debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr',key,'debug'))
            # output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr',key,'output'))

            opt = self.opt
            model = self.model
            converter = self.converter
            opt.batch_size = 192  #

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
                    print(f"OCR : {image_labels}")
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(device)

                    # For max length prediction
                    length_for_pred = torch.IntTensor(
                        [opt.batch_max_length] * batch_size
                    ).to(device)
                    text_for_pred = (
                        torch.LongTensor(batch_size, opt.batch_max_length + 1)
                        .fill_(0)
                        .to(device)
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

                    log = open(f"./log_eval_result.txt", "a")
                    dashed_line = "-" * 120
                    head = f'{"key":25s}\t{"predicted_labels":32s}\tconfidence score'

                    print(f"{dashed_line}\n{head}\n{dashed_line}")
                    log.write(f"{dashed_line}\n{head}\n{dashed_line}\n")

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)

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
                        txt = pred
                        results.append(
                            {"confidence": confidence, "text": txt, "id": img_name}
                        )

                        print(f"{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}")
                        log.write(
                            f"{img_name:25s}\t{pred:32s}\t{confidence_score:0.4f}\n"
                        )
                    log.close()

        except Exception as ex:
            print(ex)
            raise ex
        return results
