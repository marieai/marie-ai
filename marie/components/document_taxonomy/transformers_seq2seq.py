import os
from collections import Counter
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from docarray import DocList
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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


def split_array(
    sequence: Sequence[Any], split_points: List[int]
) -> List[Sequence[Any]]:
    """Split a sequence into chunks at the given split points."""
    if not split_points:  # Handle the edge case when no split points are provided
        return [sequence]

    chunks = []
    start = 0

    for split_point in split_points:
        if start != split_point:  # Avoid appending empty slices
            chunk = sequence[start:split_point]
            chunks.append(chunk)
        start = split_point

    chunks.append(sequence[start:])  # Add the remaining part of the sequence
    return chunks


def pad_or_truncate_batched_components(
    batch_labels: list[list],
    batch_predictions: list[list],
    batch_confidences: list[list],
) -> Tuple[list[list], list[list], list[list]]:
    """
    Align `batch_predictions` and `batch_confidences` to match the size of `batch_labels`
    for each sub-batch in the batched input.

    Args:
        batch_labels: Nested list of ground truth labels (used as reference size per sub-batch).
        batch_predictions: Nested list of predicted labels across batches.
        batch_confidences: Nested list of confidence values across batches.

    Returns:
        Nested tuple containing aligned (batch_labels, batch_predictions, batch_confidences):
          - batch_predictions and batch_confidences are truncated or padded per sub-batch
            to match the size of batch_labels.
    """
    aligned_labels, aligned_predictions, aligned_confidences = [], [], []

    # Iterate through each sub-batch
    for labels, predictions, confidences in zip(
        batch_labels, batch_predictions, batch_confidences
    ):
        target_size = len(labels)
        if len(predictions) > target_size:
            predictions = predictions[:target_size]
        else:
            predictions.extend(
                ["UNKNOWN"] * (target_size - len(predictions))
            )  # Pad with "UNKNOWN"

        if len(confidences) > target_size:
            confidences = confidences[:target_size]
        else:
            confidences.extend([0.0] * (target_size - len(confidences)))  # Pad with 0.0

        aligned_labels.append(labels)
        aligned_predictions.append(predictions)
        aligned_confidences.append(confidences)

    return aligned_labels, aligned_predictions, aligned_confidences


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
        k_completions: int = 5,
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
        :param k_completions: Number of completions to generate for each input.
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

        # check.sequence_param(documents, "documents", of_type=MarieDoc)
        # check.sequence_param(metadata, "metadata", of_type=Dict)
        # check.str_param(taxonomy_key, "taxonomy_key")
        #
        assert len(documents) == len(
            metadata
        ), "Documents and metadata must have the same length"

        if len(documents) == 0:
            return documents
        if batch_size is None:
            batch_size = self.batch_size

        for doc, meta in zip(documents, metadata):
            chunks = create_chunks(
                meta,
                self.tokenizer,
                max_token_length=self.max_input_length,
                method="SPATIAL_FORMAT",
                mode="seq2seq",
            )
            num_batches = len(chunks) // batch_size + (len(chunks) % batch_size > 0)
            batched_chunks = batchify(chunks, batch_size)
            if taxonomy_key in doc.tags:
                self.logger.warning(
                    f"Document {doc.id} already contains a tag with key {taxonomy_key}. Overwriting it."
                )

            grouped_annotations = {}

            for idx, batch in enumerate(batched_chunks):
                self.logger.info(f"Processing batch {idx + 1}/{num_batches}")
                texts = [chunk["prompt"] for chunk in batch]
                expected_batch_labels = [chunk["line_ids"] for chunk in batch]

                batch_outputs = self.classify(texts)
                batch_predictions, batch_confidences = zip(
                    *batch_outputs
                )  # Unpack into two separate lists.
                batch_predictions = list(
                    batch_predictions
                )  # Make batch_predictions mutable
                # Align the batches
                _, batch_predictions, batch_confidences = (
                    pad_or_truncate_batched_components(
                        expected_batch_labels, batch_predictions, batch_confidences
                    )
                )

                for chunk, predictions, confidences in zip(
                    batch, batch_predictions, batch_confidences
                ):
                    line_ids = chunk["line_ids"]
                    print("=====" * 10)
                    print(line_ids)
                    print(predictions)
                    print(confidences)

                    for line_id, prediction, confidence in zip(
                        line_ids, predictions, confidences
                    ):
                        if line_id not in grouped_annotations:
                            grouped_annotations[line_id] = {
                                "predictions": [],
                                "scores": [],
                            }

                        grouped_annotations[line_id]["predictions"].append(prediction)
                        grouped_annotations[line_id]["scores"].append(confidence)

            final_annotations = {}
            for line_id, data in grouped_annotations.items():
                predictions = data["predictions"]
                scores = data["scores"]

                prediction_counts = Counter(predictions)
                most_frequent_prediction = prediction_counts.most_common(1)[0][
                    0
                ]  # Get the most frequent prediction
                relevant_scores = [
                    score
                    for pred, score in zip(predictions, scores)
                    if pred == most_frequent_prediction
                ]
                average_score = sum(relevant_scores) / len(relevant_scores)
                final_annotations[line_id] = {
                    "prediction": most_frequent_prediction,
                    "average_score": round(
                        average_score, 5
                    ),  # Round to 5 decimal places if needed
                }

            annotations = []
            for line_id, data in final_annotations.items():
                prediction = data["prediction"]
                average_score = data["average_score"]
                taxonomy_prediction = TaxonomyPrediction(
                    line_id=line_id,
                    label=prediction,
                    score=average_score,
                )
                annotations.append(taxonomy_prediction)
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
        model = self.model
        tokenizer = self.tokenizer

        inputs = tokenizer(
            texts_to_classify,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        max_new_tokens = 64  # Maximum number of tokens to generate
        input_ids = inputs["input_ids"]

        outputs = model.generate(
            input_ids,
            attention_mask=inputs["attention_mask"],
            num_beams=4,  # 1 = greedy search, > 1 = beam search
            do_sample=False,
            length_penalty=1.0,  # Encourage long predictions (set >1.0 to favor length)
            max_new_tokens=max_new_tokens,  # Increase if predictions are too short
            return_dict_in_generate=True,
            output_scores=True,
        )

        predicted_labels = tokenizer.batch_decode(
            outputs.sequences,  # The actual generated token IDs
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # split predicted_labels into tokens and compute scores, this can fail for models that don't return scores
        if not hasattr(model, "compute_transition_scores"):
            raise NotImplementedError(
                "The model does not support compute_transition_scores."
            )

        predicted_labels = [label.split() for label in predicted_labels]
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
            beam_indices=outputs.beam_indices,
        )
        transition_proba = np.exp(transition_scores.cpu().numpy())
        # We only have scores for the generated tokens, so pop out the prompt tokens
        input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
        generated_ids = outputs.sequences[:, input_length:]
        stop_word_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        batched_tokens, batched_scores = [], []

        for gen_ids, trans_probs in zip(generated_ids, transition_proba):
            generated_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
            tokens, scores = [], []
            for token, proba in zip(generated_tokens, trans_probs):
                if token in stop_word_ids:
                    break
                tokens.append(token)
                scores.append(proba)

            split_loc = [i for i, token in enumerate(tokens) if token.startswith('▁')]
            batched_tokens.append(split_array(tokens, split_loc))
            batched_scores.append(split_array(scores, split_loc))

        if not (len(predicted_labels) == len(batched_tokens) == len(batched_scores)):
            raise ValueError(
                "Mismatch in lengths: predicted_labels, tokens, and scores batches."
            )
        confidences_batches = [
            [round(float(np.mean(scores)), 5) for scores in scores_batch]
            for scores_batch in batched_scores
        ]

        self.logger.debug(
            f"Classification of {len(texts_to_classify)} batch took {time() - start} seconds"
        )

        return list(zip(predicted_labels, confidences_batches))
