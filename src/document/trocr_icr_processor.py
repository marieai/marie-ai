import os
import sys
import typing

import models.unilm.trocr
# import models.unilm.trocr.task
from models.unilm.trocr.task import SROIETextRecognitionTask

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from numpyencoder import NumpyEncoder

from document.icr_processor import IcrProcessor
from models.icr.memory_dataset import MemoryDataset

import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms

from timer import Timer

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


class Object(object):
    pass


def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path], arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose(
        [transforms.Resize((384, 384), interpolation=3), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    generator = task.build_generator(model, cfg.generation, extra_gen_cls_kwargs={"lm_model": None, "lm_weight": None})

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(pil_img, img_transform):
    im = pil_img.convert('RGB').resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        'net_input': {"imgs": im},
    }

    return sample


@Timer(text="Text in {:.4f} seconds")
def get_text(cfg, task, generator, model, sample, bpe):
    decoder_output = task.inference_step(generator, model, sample, prefix_tokens=None, constraints=None)
    decoder_output = decoder_output[0][0]  # top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )

    print(hypo_str)
    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str


class TrOcrIcrProcessor(IcrProcessor):
    def __init__(self, work_dir: str = "/tmp/icr", cuda: bool = False) -> None:
        super().__init__(work_dir, cuda)
        print("TROCR ICR processor [cuda={}]".format(cuda))
        model_path = "/home/gbugaj/devio/3rdparty/unilm/models/trocr-large-printed.pt"
        beam = 5
        self.model, self.cfg, self.task, self.generator, self.bpe, self.img_transform, self.device = init(
            model_path, beam
        )

        opt = Object()
        opt.rgb = False
        self.opt = opt

    def recognize_from_fragments(self, images, **kwargs) -> typing.List[typing.Dict[str, any]]:
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
            eval_data = MemoryDataset(images=images, opt=opt)
            results = []

            for img, img_name in eval_data:
                sample = preprocess(img, self.img_transform)
                text = get_text(self.cfg, self.task, self.generator, self.model, sample, self.bpe)
                print(f"format : {text}  >> {img_name}")
                confidence = 0
                results.append({"confidence": confidence, "text": text, "id": img_name})

        except Exception as ex:
            raise ex
        return results
