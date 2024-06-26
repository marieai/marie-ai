import os
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


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

        model_name_or_path = ModelRegistry.get(
            model_name_or_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        self.model, self.processor, self.tokenizer = self.setup_model(
            model_name_or_path, self.device
        )

    def setup_model(
        self, model_name_or_path="microsoft/layoutlmv3-base", device: str = "cuda"
    ):
        """prepare for the model"""
        model = AutoModel.from_pretrained(model_name_or_path).to(device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            apply_ocr=False,
            do_resize=True,
            resample=Image.BILINEAR,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            apply_ocr=False,
        )

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

        # ensure images is 224x224
        image = image.resize((224, 224))

        with torch.no_grad():
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
        encoding = processor(
            # fmt: off
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=512,
            # fmt: on
        ).to(model.device)

        with torch.inference_mode():
            model_output = model(**encoding)
            image_features = model_output.last_hidden_state
            image_features = image_features.mean(dim=1)
            image_features_as_np = image_features.cpu().detach().numpy()

            return image_features_as_np
