import os
import traceback
from typing import Optional, Union, List, Callable, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from docarray import DocList
from torch.nn import Module
from tqdm import tqdm
from transformers import (
    pipeline,
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from marie.constants import __model_path__
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from ..document_classifier.base import BaseDocumentClassifier
from ...api.docs import MarieDoc
from ...helper import batch_iterator
from ...logging.profile import TimeContext
from ...registry.model_registry import ModelRegistry


def scale_bounding_box(
    box: List[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]


class BatchableMarieDoc(MarieDoc):
    words: List
    boxes: List


class TransformersDocumentClassifier(BaseDocumentClassifier):
    """
    Transformer based model for document classification using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: Optional[int] = 1,
        task: str = "text-classification-multimodal",
        labels: Optional[List[str]] = None,
        batch_size: int = 16,
        classification_field: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        id2label: Optional[dict[int, str]] = None,
        **kwargs,
    ):
        """
        Load a text classification model from ModelRepository or HuggingFace model hub.

        TODO: ADD EXAMPLE AND CODE SNIPPET

        See https://huggingface.co/models for full list of available models.
        Filter for text classification models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
        Filter for zero-shot classification models (NLI): https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to use GPU (if available).
        :param top_k: The number of top predictions to return. The default is 1. Enter None to return all the predictions. Only used for task 'text-classification'.
        :param task: 'text-classification' or 'zero-shot-classification'
        :param labels: Only used for task 'zero-shot-classification'. List of string defining class labels, e.g.,
        ["positive", "negative"] otherwise None. Given a LABEL, the sequence fed to the model is "<cls> sequence to
        classify <sep> This example is LABEL . <sep>" and the model predicts whether that sequence is a contradiction
        or an entailment.
        :param batch_size: Number of Documents to be processed at a time.
        :param classification_field: Name of Document's meta field to be used for classification. If left unset, Document.tensor is used by default.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param show_error: Whether to show errors during inference. If set to False, errors will be logged as debug messages.
        :param id2label: A dictionary mapping label ids to label names. Only used for task 'text-classification' and 'text-classification-multimodal'.
        :param kwargs: Additional keyword arguments passed to the model.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document classification : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.task = task
        self.labels = labels
        self.top_k = top_k
        self.progress_bar = False
        self.id2label = id2label
        # Keys are always strings in JSON/YAML so convert ids to int here.
        if id2label is not None:
            self.id2label = {int(key): value for key, value in self.id2label.items()}

        if labels and task == "text-classification":
            self.logger.warning(
                "Provided labels %s will be ignored for task text-classification. Set task to "
                "zero-shot-classification to use labels.",
                labels,
            )

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

        if tokenizer is None:
            tokenizer = model_name_or_path

        if task == "zero-shot-classification":
            self.model = pipeline(
                task=task,
                model=model_name_or_path,
                tokenizer=tokenizer,
                revision=model_version,
                use_auth_token=use_auth_token,
                device=resolved_devices[0],
            )
        elif task == "text-classification":
            self.model = pipeline(
                task=task,
                model=model_name_or_path,
                tokenizer=tokenizer,
                device=resolved_devices[0],
                revision=model_version,
                top_k=top_k,
                # use_auth_token=use_auth_token,
            )
        elif task == "text-classification-multimodal":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path
            )
            self.model = self.optimize_model(self.model)
            self.model = self.model.eval().to(resolved_devices[0])
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

            feature_extractor = LayoutLMv3ImageProcessor(
                apply_ocr=False, do_resize=True, resample=Image.BILINEAR
            )
            self.processor = LayoutLMv3Processor(
                feature_extractor, tokenizer=self.tokenizer
            )

    def predict(
        self,
        documents: DocList[MarieDoc],
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        if batch_size is None:
            batch_size = self.batch_size

        if len(documents) == 0:
            return documents

        if self.task == "text-classification-multimodal":
            assert (
                words is not None and boxes is not None
            ), "words and boxes must be provided for sequence classification"
            assert (
                len(documents) == len(words) == len(boxes)
            ), "documents, words and boxes must have the same length"

        # create a named tuple of (document, words, boxes) for each document

        batchable_docs = DocList(
            BatchableMarieDoc(
                tensor=doc.tensor,
                words=word,
                boxes=box,
            )
            for doc, word, box in zip(documents, words, boxes)
        )

        batches = batch_iterator(batchable_docs, batch_size)
        predictions = []
        pb = tqdm(
            total=len(documents),
            disable=not self.progress_bar,
            desc="Classifying documents",
        )

        id2label = self.id2label
        for batch in batches:
            batch_results = []
            if self.task == "zero-shot-classification":
                batch_results = self.model(
                    batch,
                    candidate_labels=self.labels,
                    truncation=True,
                )
            elif self.task == "text-classification":
                try:
                    from torch.utils.data import Dataset

                    class IterableData(Dataset):
                        def __init__(self, items: Sequence[BatchableMarieDoc]):
                            self.items = items

                        def __len__(self):
                            return len(self.items)

                        def __getitem__(self, idx):
                            doc = self.items[idx]
                            text = " ".join(doc.words)
                            return text

                    pipe_batched_results = self.model(
                        IterableData(batch), top_k=self.top_k, truncation=True
                    )
                    k_results = []

                    for results in pipe_batched_results:
                        for k in range(self.top_k):
                            result_k1 = results[k]
                            label = result_k1["label"]
                            label_id = int(label.split("_")[1])
                            if id2label is not None:
                                label = id2label[label_id]
                            score = result_k1["score"]
                            k_results.append(
                                {
                                    "label": label,
                                    "score": score,
                                }
                            )
                    batch_results.append(k_results)
                except Exception as err:
                    # print traceback
                    traceback.print_exc()

                    err_msg = f"Error while classifying documents: {err}"
                    if self.show_error:
                        self.logger.error(err_msg)
                    else:
                        self.logger.debug(err_msg)

            elif self.task == "text-classification-multimodal":
                batch_results = []
                for doc in batchable_docs:
                    batch_results.append(
                        self.predict_document_image(
                            doc.tensor,
                            words=doc.words,
                            boxes=doc.boxes,
                            top_k=self.top_k,
                        )
                    )

            predictions.extend(batch_results)
            pb.update(len(batch))
        pb.close()

        for document, prediction in zip(documents, predictions):
            if (
                self.task == "text-classification-multimodal"
                or self.task == "text-classification"
            ):
                formatted_prediction = {
                    "label": prediction[0]["label"],
                    "score": prediction[0]["score"],
                    "details": {el["label"]: el["score"] for el in prediction},
                }
            else:
                raise NotImplementedError  # TODO
            document.tags["classification"] = formatted_prediction
        return documents

    def predict_document_image(
        self,
        image: np.ndarray,
        words: List[List[str]],
        boxes: List[List[int]],
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Predicts the label of a document image

        :param image: image to predict
        :param words: words in the image
        :param boxes: bounding boxes of the words
        :param top_k: number of predictions to return
        :return: prediction dictionary with label and score
        """

        if self.id2label is None:
            id2label = self.model.config.id2label
        else:
            id2label = self.id2label

        width, height = image.shape[1], image.shape[0]
        width_scale = 1000 / width
        height_scale = 1000 / height
        boxes_normalized = [
            scale_bounding_box(box, width_scale, height_scale) for box in boxes
        ]

        encoding = self.processor(
            image,
            words,
            boxes=boxes_normalized,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # https://github.com/pytorch/pytorch/tree/main/torch/csrc/jit/codegen/onednn#example-with-bfloat16
        # Disable AMP for JIT
        # torch._C._jit_set_autocast_mode(False)
        # with torch.no_grad(), torch.cpu.amp.autocast():

        with torch.inference_mode():
            output = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                bbox=encoding["bbox"].to(self.device),
                pixel_values=encoding["pixel_values"].to(self.device),
            )
        # TODO: add top_k support

        logits = output.logits
        predicted_class = logits.argmax(-1)
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

        return [
            {
                "label": id2label[predicted_class.item()],
                "score": probabilities[predicted_class.item()],
            }
        ]

    def optimize_model(self, model: nn.Module) -> Callable | Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        if True:
            return model

        try:
            with TimeContext("Compiling model", logger=self.logger):
                import torchvision.models as models
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
