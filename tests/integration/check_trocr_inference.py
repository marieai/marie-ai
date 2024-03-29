import glob
import os

import fairseq
import torch
import torchvision.transforms as transforms
from fairseq import utils
from fairseq_cli import generate
from PIL import Image

import marie.models.unilm.trocr
from marie.constants import __config_dir__, __model_path__
from marie.timer import Timer


def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={
            "beam": beam,
            "task": "text_recognition",
            "data": "",
            "fp16": False,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    generator = task.build_generator(
        model,
        cfg.generation,
        extra_gen_cls_kwargs={"lm_model": None, "lm_weight": None},
    )

    bpe = task.build_bpe(cfg.bpe)

    print(task)
    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):
    im = Image.open(img_path).convert("RGB").resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        "net_input": {"imgs": im},
    }

    return sample


@Timer(text="Text in {:.4f} seconds")
def get_text(cfg, generator, model, sample, bpe):
    print(task)
    decoder_output = task.inference_step(
        generator, model, sample, prefix_tokens=None, constraints=None
    )
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


if __name__ == "__main__":

    print(__model_path__)
    model_path = os.path.join(__model_path__, "trocr/trocr-large-printed.pt")
    burst_dir = "/home/gbugaj/dev/marieai/marie-ai/assets/psm/word"

    beam = 5
    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)

    for _path in sorted(glob.glob(os.path.join(burst_dir, "*.*"))):
        sample = preprocess(_path, img_transform)
        text = get_text(cfg, generator, model, sample, bpe)
        print(f"format : {text}  >> {_path}")

    print("done")
