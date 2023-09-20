import os
import sys
import typing

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from marie.document.ocr_processor import OcrProcessor
from marie.lang import Object
from marie.models.icr.dataset import AlignCollate, RawDataset
from marie.models.icr.memory_dataset import MemoryDataset
from marie.models.icr.model import Model
from marie.models.icr.utils import AttnLabelConverter, CTCLabelConverter


class LevenshteinOcrProcessor(OcrProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/icr",
        models_dir: str = "./model_zoo/icr",
        cuda: bool = True,
    ) -> None:
        super().__init__(work_dir, cuda)
        print("Tesseract OCR processor [cuda={}]".format(cuda))

        saved_model = os.path.join(
            models_dir, "TPS-ResNet-BiLSTM-Attn-case-sensitive-ft", "best_accuracy.pth"
        )

        self.converter, self.model = self.__load()

    def __load(self):
        """model configuration"""
        opt = self.opt

        print("Evaluating on device")
        model = Model(opt)

        return None, model

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
            results = []

        except Exception as ex:
            print(ex)
            raise ex
        return results
