import os
from typing import Optional, Union, List

import torch

from marie.document_classifier.base import BaseDocumentClassifier
from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings


class TransformersDocumentClassifier(BaseDocumentClassifier):
    """
    Transformer based model for document classification using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    """

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: Optional[int] = 1,
        task: str = "text-classification",
        labels: Optional[List[str]] = None,
        batch_size: int = 1,
        classification_field: Optional[str] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """
        Load a text classification model from Transformers.

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
        :param classification_field: Name of Document's meta field to be used for classification. If left unset, Document.content is used by default.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(**kwargs)
        self.show_error = show_error  # show prediction errors
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document classification : {model_name_or_path}")

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
