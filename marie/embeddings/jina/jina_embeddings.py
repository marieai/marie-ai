import os
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger
from marie.logging_core.profile import TimeContext
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


def _ensure_transformers_rope_compatibility() -> None:
    """Restore the RoPE alias expected by Jina v4's bundled Qwen code."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRotaryEmbedding,
    )

    ROPE_INIT_FUNCTIONS.setdefault(
        "default", Qwen2_5_VLRotaryEmbedding.compute_default_rope_parameters
    )


class JinaEmbeddings(EmbeddingsBase):
    """Jina text embeddings with multimodal support for Jina v4 models."""

    def __init__(
        self,
        model_name_or_path: Union[
            str, os.PathLike
        ] = "jinaai/jina-embeddings-v2-base-en",
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 4,
        task: str = "retrieval",
        truncate_dim: Optional[int] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """Initialize a Jina text or Jina v4 multimodal embedding model."""

        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Embeddings Jina : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.task = task
        self.truncate_dim = truncate_dim

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
            version=model_version,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        assert os.path.exists(model_name_or_path)
        self.logger.info(f"Resolved model : {model_name_or_path}")
        _ensure_transformers_rope_compatibility()
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )  # trust_remote_code is needed to use the encode method
        self.model = self.model.to(self.device)
        self._supports_image_embeddings = hasattr(self.model, "encode_image")
        if not self._supports_image_embeddings:
            self.model = self.optimize_model(self.model)

    def get_embeddings(
        self,
        texts: List[str],
        truncation: bool = None,
        max_length: int = None,
        is_query: bool = False,
    ) -> EmbeddingsObject:
        if not texts:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        with torch.no_grad():
            try:
                if hasattr(self.model, "encode_text"):
                    embeddings = self.model.encode_text(
                        texts,
                        task=self.task,
                        max_length=max_length or 32768,
                        batch_size=self.batch_size,
                        return_numpy=True,
                        truncate_dim=self.truncate_dim,
                        prompt_name="query" if is_query else "passage",
                    )
                else:
                    embeddings = self.model.encode(
                        texts, batch_size=self.batch_size, truncation=truncation
                    )

                result = EmbeddingsObject()
                result.embeddings = np.asarray(embeddings, dtype=np.float32)
                result.total_tokens = -1  # len(embeddings[0])

                return result
            except Exception as e:
                self.logger.error(
                    f"Error during inference: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def get_image_embeddings(self, images: List[str]) -> EmbeddingsObject:
        if not images:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        if not self._supports_image_embeddings:
            raise NotImplementedError(
                "This Jina model does not provide multimodal image embeddings"
            )

        with torch.no_grad():
            try:
                embeddings = self.model.encode_image(
                    images,
                    task=self.task,
                    batch_size=self.batch_size,
                    return_numpy=True,
                    truncate_dim=self.truncate_dim,
                )
                result = EmbeddingsObject()
                result.embeddings = np.asarray(embeddings, dtype=np.float32)
                result.total_tokens = 0
                return result
            except Exception as e:
                self.logger.error(
                    f"Error during image inference: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def embed_text(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        result = self.get_embeddings(texts, is_query=is_query)
        return np.asarray(result.embeddings, dtype=np.float32)

    def embed_images(self, images: List[str]) -> np.ndarray:
        result = self.get_image_embeddings(images)
        return np.asarray(result.embeddings, dtype=np.float32)

    @property
    def supports_image_embeddings(self) -> bool:
        return self._supports_image_embeddings

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
