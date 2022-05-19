import logging
import os
import sys
import time
import typing
from random import random

import marie.models.unilm.trocr

# import models.unilm.trocr.task
from marie.lang import Object
from marie.models.unilm.trocr.task import SROIETextRecognitionTask

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from numpyencoder import NumpyEncoder

from marie.document.icr_processor import IcrProcessor
from marie.models.icr.memory_dataset import MemoryDataset

import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms

from marie.timer import Timer

import multiprocessing as mp
import functools
import concurrent.futures
import multiprocessing.pool
import random
import threading

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

device = "cuda" if torch.cuda.is_available() else "cpu"


@Timer(text="Preprocess in {:.4f} seconds")
def initXXX(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path], arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False}
    )

    # device = torch.device('cpu')
    model[0].to(device)

    img_transform = transforms.Compose(
        [transforms.Resize((384, 384), interpolation=3), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    generator = task.build_generator(model, cfg.generation, extra_gen_cls_kwargs={"lm_model": None, "lm_weight": None})

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def init(model_path, beam=5):
    model, cfg, inference_task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path], arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": True}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose(
        [transforms.Resize((384, 384), interpolation=3), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    generator = inference_task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={"lm_model": None, "lm_weight": None}
    )

    bpe = inference_task.build_bpe(cfg.bpe)

    return model, cfg, inference_task, generator, bpe, img_transform, device


@Timer(text="Preprocess in {:.4f} seconds")
def preprocess(pil_img, img_transform):
    # im = pil_img.convert("RGB").resize((384, 384))
    im = pil_img.convert("RGB").resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        "net_input": {"imgs": im},
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

    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str


def work(img, name, img_transform, cfg, task, generator, model, bpe):
    start = time.time()
    sample = preprocess(img, img_transform)
    text = get_text(cfg, task, generator, model, sample, bpe)
    confidence = 0
    delta = time.time() - start
    return {"confidence": confidence, "text": text, "id": name, "dt": delta}


def work_process(img, name):
    start = time.time()
    model_path = "./model_zoo/trocr/trocr-large-printed.pt"
    beam = 5
    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)
    print("Model load elapsed PP: %s" % (time.time() - start))

    # sample = preprocess(img, img_transform)
    # text = get_text(cfg, task, generator, model, sample, bpe)
    text = "A"
    confidence = 0
    delta = time.time() - start
    result = {"confidence": confidence, "text": text, "id": name, "dt": delta}
    print(result)
    return result


class TrOcrIcrProcessor(IcrProcessor):
    def __init__(self, work_dir: str = "/tmp/icr", models_dir: str = "./model_zoo/trocr", cuda: bool = True) -> None:
        super().__init__(work_dir, cuda)
        model_path = os.path.join(models_dir, "trocr-large-printed.pt")
        print(f"TROCR ICR processor [cuda={cuda}] : {model_path}")

        if not os.path.exists(model_path):
            raise Exception(f"File not found : {model_path}")

        start = time.time()

        beam = 5
        self.model, self.cfg, self.task, self.generator, self.bpe, self.img_transform, self.device = init(
            model_path, beam
        )
        print("Model load elapsed: %s" % (time.time() - start))

        opt = Object()
        opt.rgb = False
        self.opt = opt

    def recognize_from_fragments_process(self, images, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        print("ICR processing : recognize_from_boxes via boxes")

        try:
            logger = logging.getLogger(__name__)
            logger.info("Preprocessing ...")
            # logger.info("Loaded {} icr from {}".format(len(ret), words))
            # debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr',key,'debug'))
            # output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr',key,'output'))

            opt = self.opt
            eval_data = MemoryDataset(images=images, opt=opt)
            results = []
            args = [(img, name) for img, name in eval_data]

            start = time.time()
            from multiprocessing import Pool

            print("Time elapsed: %s" % (time.time() - start))
            max_workers = mp.cpu_count() // 4

            print("\nPool Executor:")
            pool = Pool(processes=4)
            # pool.imap_unordered()
            pool_results = pool.starmap(work_process, args)
            print(pool_results)
            pool.close()
            pool.join()

            print("Time elapsed[submitted]: %s" % (time.time() - start))
            for r in pool_results:
                print("Time elapsed[result]: %s  , %s" % (time.time() - start, r))
                # results.append(result)
            print("Time elapsed[all]: %s" % (time.time() - start))
        except Exception as ex:
            raise ex
        return results

    def recognize_from_fragments_threaded(self, images, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        print("ICR processing : recognize_from_boxes via boxes")

        try:

            logger = logging.getLogger(__name__)
            logger.info("Preprocessing ...")
            # logger.info("Loaded {} icr from {}".format(len(ret), words))
            # debug_dir =  ensure_exists(os.path.join(self.work_dir,id,'icr',key,'debug'))
            # output_dir = ensure_exists(os.path.join(self.work_dir,id,'icr',key,'output'))

            opt = self.opt
            eval_data = MemoryDataset(images=images, opt=opt)
            results = []
            args = [(img, name, self.img_transform) for img, name in eval_data]

            start = time.time()
            print("Time elapsed: %s" % (time.time() - start))

            print("\nThreadPoolExecutor:")
            f = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count() // 4) as executor:
                for img, name in eval_data:
                    f.append(
                        executor.submit(
                            work,
                            img,
                            name,
                            self.img_transform,
                            self.cfg,
                            self.task,
                            self.generator,
                            self.model,
                            self.bpe,
                        )
                    )

            print("Time elapsed[submitted]: %s" % (time.time() - start))

            for r in concurrent.futures.as_completed(f):
                result = r.result()
                print("Time elapsed[result]: %s  , %s" % (time.time() - start, result))
                results.append(result)
            print("Time elapsed[all]: %s" % (time.time() - start))

        except Exception as ex:
            raise ex
        return results

    def recognize_from_fragments(self, images, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments synchronously.

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
            start = time.time()
            for img, img_name in eval_data:
                sample = preprocess(img, self.img_transform)
                text = get_text(self.cfg, self.task, self.generator, self.model, sample, self.bpe)
                # text = ""
                confidence = 0
                results.append({"confidence": confidence, "text": text, "id": img_name})
            print("ICR Time elapsed: %s" % (time.time() - start))

        except Exception as ex:
            raise ex
        return results
