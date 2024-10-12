import os
from typing import List, Optional, Union

import clip
import numpy as np
import torch
from PIL import Image
from torch import nn

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.profile import TimeContext
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class OpenAIEmbeddings(EmbeddingsBase):
    def __init__(
        self,
        model_name_or_path: Union[
            str, os.PathLike
        ] = None,  # "openai/clip-vit-base-patch32",
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

        self.model_name_or_path = ModelRegistry.get(
            model_name_or_path,
            version=None,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        self.model, self.processor = self.setup_model(
            self.model_name_or_path, self.device
        )

        self.model = self.optimize_model(self.model)

    def setup_model(self, resolved_model_path: str, device: str = "cuda"):
        """prepare for the model"""
        config = ModelRegistry.config(resolved_model_path)
        if "architecture" not in config:
            raise ValueError(
                f"Model config does not contain 'architecture' key: {config}, it should contain 'architecture' key with the model architecture name."
            )
        architecture = config["architecture"]
        checkpoint = ModelRegistry.checkpoint(resolved_model_path)
        model, preprocess = clip.load(architecture, device=device, jit=False)
        checkpoint = torch.load(
            checkpoint,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, preprocess

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
                if image is None:
                    embeddings = self.get_single_text_embedding(
                        self.model, " ".join(texts)
                    )
                    result = EmbeddingsObject()
                    result.embeddings = embeddings
                    result.total_tokens = -1
                    return result

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
            # write the image tot temp file
            # image.save(f"/tmp/dim/embed/temp_{image}.png")
            src = processor(image).unsqueeze(0).to(self.device)
            embedding = model.encode_image(src)
            # convert the embeddings to numpy array
            embedding_as_np = embedding.cpu().detach().numpy()
            return embedding_as_np

    def get_single_text_embedding(self, model, text):
        with torch.inference_mode():
            inputs = clip.tokenize([text]).to(self.device)
            text_embeddings = model.encode_text(inputs)
            embedding_as_np = text_embeddings.cpu().detach().numpy()
            return embedding_as_np

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model", logger=self.logger):
                import torch._dynamo as dynamo

                torch._dynamo.config.verbose = True
                torch._dynamo.config.suppress_errors = True

                # https://dev-discuss.pytorch.org/t/torchinductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874
                model = torch.compile(
                    model, mode="max-autotune", dynamic=True, backend="cudagraphs"
                )
                return model
        except Exception as err:
            raise err
