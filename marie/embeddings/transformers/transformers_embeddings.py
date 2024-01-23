import os
from typing import List, Optional, Union

import torch

from marie.embeddings.base import EmbeddingsBase


class TransformersEmbeddings(EmbeddingsBase):
    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
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
        :param show_error: Whether to show errors during inference. If set to False, errors will be logged as debug messages.
        :param kwargs: Additional keyword arguments passed to the model.
        """

    def get_embeddings(self, data: str, **kwargs) -> List[float]:
        """
        Generate embedding
        :param data:
        :param kwargs:
        """
        raise NotImplementedError
