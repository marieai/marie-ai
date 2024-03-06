import os
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class OpenAIEmbeddings(EmbeddingsBase):
    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike] = "openai/clip-vit-base-patch32",
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
        super().__init__(**kwargs)

        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path

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
        if False:
            model_name_or_path = ModelRegistry.get(
                model_name_or_path,
                version=None,
                raise_exceptions_for_missing_entries=True,
                **registry_kwargs,
            )

        self.model, self.processor, self.tokenizer = self.setup_model(
            model_name_or_path, self.device
        )

    def setup_model(self, model_name_or_path, device: str = "cuda"):
        """prepare for the model"""
        model_name_or_path = "openai/clip-vit-base-patch16"
        model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)

        return model, processor, tokenizer

    def get_embeddings(
        self,
        texts: List[str],
        truncation: bool = None,
        max_length: int = None,
        image: Image.Image = None,
        boxes: List[List[int]] = None,
        **kwargs,
    ) -> EmbeddingsObject:
        with torch.inference_mode():
            try:
                embeddings = self.get_single_image_embedding(
                    self.model, self.processor, image, words=texts, boxes=boxes
                )

                result = EmbeddingsObject()
                result.embeddings = embeddings
                result.total_tokens = -1  # len(embeddings[0])

                return result
            except Exception as e:
                self.logger.error(
                    f"Error during inference: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def get_single_image_embedding(
        self, model, processor, image, words, boxes
    ) -> np.ndarray:

        with torch.inference_mode():
            # image = image.resize((224, 224))
            image = processor(text=None, images=image, return_tensors="pt")[
                "pixel_values"
            ].to(model.device)

            embedding = model.get_image_features(image)
            # convert the embeddings to numpy array
            embedding_as_np = embedding.cpu().detach().numpy()
            return embedding_as_np


def get_single_text_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np
