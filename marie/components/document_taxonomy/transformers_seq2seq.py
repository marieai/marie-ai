import os
from time import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from docarray import DocList
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from marie import DocumentArray, check
from marie.constants import __model_path__
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...api.docs import MarieDoc
from ...registry.model_registry import ModelRegistry
from ...utils.utils import batchify
from .base import BaseDocumentTaxonomy
from .datamodel import TaxonomyPrediction
from .verbalizers import create_chunks


class DocumentTaxonomySeq2SeqLM(BaseDocumentTaxonomy):
    """
    Transformer based model for document taxonomy prediction.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        id2label: Optional[dict[int, str]] = None,
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
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document taxonomy Seq2Seq : {model_name_or_path}")
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

        if tokenizer is None:
            tokenizer = model_name_or_path

        if id2label is None:
            raise ValueError("id2label is required for Seq2Seq models")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model = self.model.eval().to(resolved_devices[0])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_input_length = self.tokenizer.model_max_length
        self.id2label = {int(key): value for key, value in id2label.items()}

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

        for doc, meta in zip(documents, metadata):
            chunks = create_chunks(
                meta, self.tokenizer, max_token_length=self.max_input_length
            )
            num_batches = len(chunks) // batch_size + (len(chunks) % batch_size > 0)
            batched_chunks = batchify(chunks, batch_size)
            if taxonomy_key in doc.tags:
                self.logger.warning(
                    f"Document {doc.id} already contains a tag with key {taxonomy_key}. Overwriting it."
                )

            annotations = []
            for idx, batch in enumerate(batched_chunks):
                self.logger.info(f"Processing batch {idx + 1}/{num_batches}")
                texts = [chunk["prompt"] for chunk in batch]
                predictions = self.classify(texts)

                for chunk, prediction in zip(batch, predictions):
                    annotation = TaxonomyPrediction(
                        line_id=int(chunk["line_id"]),
                        label=prediction[0],
                        score=prediction[1],
                    )
                    annotations.append(annotation)
                doc.tags[taxonomy_key] = annotations
        return documents

    @torch.no_grad()
    def classify(self, texts_to_classify: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple input texts into their predicted labels and associated confidence scores.

        :param texts_to_classify: A list of input texts to be classified.
        :type texts_to_classify: List[str]
        :return: A list of tuples where each tuple contains the predicted label
                 (as a string) and its associated confidence score (as a float).
        :rtype: List[Tuple[str, float]]
        """
        start = time()
        inputs = self.tokenizer(
            texts_to_classify,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences, predicted_classes = torch.max(probs, dim=1)
        predicted_classes = predicted_classes.cpu().numpy()
        confidences = confidences.cpu().numpy()
        predicted_labels = [self.id2label[class_id] for class_id in predicted_classes]

        self.logger.debug(
            f"Classification of {len(texts_to_classify)} batch took {time() - start} seconds"
        )

        return list(zip(predicted_labels, confidences))
