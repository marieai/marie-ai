import os
from functools import partial
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from docarray import DocList
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from ruamel.yaml import YAML

from marie import DocumentArray, check
from marie.constants import __model_path__
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...api.docs import MarieDoc
from ...boxes.dit.ulim_dit_box_processor import visualize_bboxes
from ...registry.model_registry import ModelRegistry
from ...utils.docs import frames_from_docs
from ...utils.utils import batchify
from .base import BaseDocumentTaxonomy
from .datamodel import TaxonomyPrediction
from .qavit.VisualT5 import VisualT5
from .verbalizers import create_chunks


def get_visual_t5(config):
    model = VisualT5.from_pretrained(config['llm'])
    model.freeze_llm_weights(config)
    # Apply LoRa
    if config["use_lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    model.initialize_vision_modules(config)
    model.freeze_vision_weights(config)
    # Create a two-param group optimizer
    model_base_params, model_translation_params = [], []
    for name, p in model.named_parameters():
        if 'translation_mlp' in name:
            model_translation_params.append(p)
        else:
            model_base_params.append(p)
    translation_mul = (
        config['translation_lr_mul'] if 'translation_lr_mul' in config else 1.0
    )

    optimizer = torch.optim.AdamW(
        [
            {'params': model_base_params},
            {
                'params': model_translation_params,
                'lr': config['base_lr'] * translation_mul,
            },
        ],
        lr=config['base_lr'],
        weight_decay=config['weight_decay'],
    )
    return model, optimizer, model.get_image_processor(), model.get_tokenizer()


def vt5_collate_fn(batch, tokenizer, processor, **kwargs):
    """
    Collate function for VisualT5
    :param batch:
    :param tokenizer:
    :param processor:
    :param kwargs:
    :return:
    """
    image_list, question_list = [], []
    instruct_list = []

    for image, question, instruct, _ in batch:
        image_list.append(image)
        question_list.append(question)
        instruct_list.append(instruct)
    # tokenize
    questions = tokenizer(
        question_list,
        return_tensors="pt",
        padding='longest',
        truncation=True,
        max_length=512,
    )
    return {
        'images': processor(images=image_list, return_tensors="pt"),
        'input_ids': questions.input_ids,
        'attention_mask': questions.attention_mask,
    }, instruct_list


class QaVitDocumentTaxonomy(BaseDocumentTaxonomy):
    """
    Transformer based model for document taxonomy prediction.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Load a document taxonomy model from ModelRepository or HuggingFace model hub.

        TODO: ADD EXAMPLE AND CODE SNIPPET

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param id2label: A dictionary mapping label ids to label names.
        :param kwargs: Additional keyword arguments passed to the model.
        """
        super().__init__(**kwargs)
        self.max_input_length = 512  # make this a parameter for different models
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document taxonomy : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.progress_bar = False

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        model_name_or_path = ModelRegistry.get(
            model_name_or_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )
        assert os.path.exists(model_name_or_path)
        self.logger.info(f"Resolved model : {model_name_or_path}")

        config = Path(model_name_or_path) / "config.yaml"
        yaml = YAML(typ='rt')
        self.config = yaml.load(open(config, 'r'))
        self.model, self.optimizer, self.image_processor, self.tokenizer = (
            get_visual_t5(self.config)
        )
        self.model.eval()
        self.model.to(self.device)

        self.collator = partial(
            vt5_collate_fn, tokenizer=self.tokenizer, processor=self.image_processor
        )

    def predict(
        self,
        documents: DocList[MarieDoc],
        metadata: List[dict],
        taxonomy_key: str = "taxonomy",
        batch_size: Optional[int] = None,
    ) -> DocumentArray:
        check.sequence_param(documents, "documents", of_type=MarieDoc)
        check.sequence_param(metadata, "metadata", of_type=Dict)
        check.str_param(taxonomy_key, "taxonomy_key")
        assert len(documents) == len(
            metadata
        ), "Documents and metadata must have the same length"

        if len(documents) == 0:
            return documents
        if batch_size is None:
            batch_size = self.batch_size

        frames = frames_from_docs(documents)
        max_token_length = (
            self.tokenizer.model_max_length - 112
        )  # 112 is a buffer that we need to leave for the prompt

        for doc, frame, meta in zip(documents, frames, metadata):
            chunks = create_chunks(meta, self.tokenizer, max_token_length)
            num_batches = len(chunks) // batch_size + (len(chunks) % batch_size > 0)
            batched_chunks = batchify(chunks, batch_size)
            if taxonomy_key in doc.tags:
                self.logger.warning(
                    f"Document {doc.id} already contains a tag with key {taxonomy_key}. Overwriting it."
                )

            annotations = []
            for idx, batched_chunk in enumerate(batched_chunks):
                self.logger.info(f"Processing batch {idx + 1}/{num_batches}")
                predictions = self.classify(batched_chunk, frame)

                for chunk, prediction in zip(batched_chunk, predictions):
                    annotation = TaxonomyPrediction(
                        line_id=int(chunk["line_id"]),
                        label=prediction[0],
                        score=prediction[1],
                    )
                    annotations.append(annotation)
                doc.tags[taxonomy_key] = annotations
        return documents

    @torch.no_grad()
    def classify(self, chunked_input: List[Dict], frame) -> List[Tuple[str, float]]:
        """
        Classify multiple input texts into their predicted labels and associated confidence scores.
        :rtype: List[Tuple[str, float]]
        """
        start = time()
        categories = ["TABLE", "SECTION", "CODES", "OTHER"]
        batch = []
        # convert to pillow image
        image = Image.fromarray(frame.astype("uint8")).convert("RGB")
        image.save("/home/greg/tmp/test-deck/0_1735852778890/converted_pil.png")

        idx = 0
        for chunk in chunked_input:
            question = f"classify: {chunk['question']}\ncontext: {chunk['context']}\noptions: {', '.join(categories)}"
            instruct = question

            question_bbox = [x for x in chunk['question_bbox']]
            bbox_xywh = [x for x in chunk['bbox']]
            bbox_xyxy = [
                bbox_xywh[0],
                bbox_xywh[1],
                bbox_xywh[0] + bbox_xywh[2],
                bbox_xywh[1] + bbox_xywh[3],
            ]
            src_image = visualize_bboxes(
                image,
                [question_bbox],
                format="xywh",
                blackout=True,
                blackout_color=(255, 255, 0, 125),
            )
            chunk_image = src_image.crop(bbox_xyxy)
            # chunk_image.save(f"/home/greg/tmp/test-deck/0_1735852778890/converted_pil_chunk_{idx}.png")
            batch.append((chunk_image, question, instruct, chunk))

        inputs, instructions_list = self.collator(batch)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        predicted_labels = []
        confidences = []

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=12, instructions_list=instructions_list
            )
            answers = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for answer, chunk in zip(answers, chunked_input):
                predicted_labels.append(answer)
                confidences.append(1.0)  # TODO: add confidence

        self.logger.debug(
            f"Classification of {len(chunked_input)} batch took {time() - start} seconds"
        )

        return list(zip(predicted_labels, confidences))
