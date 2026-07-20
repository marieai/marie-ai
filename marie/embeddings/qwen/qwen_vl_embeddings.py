import os
from typing import List, Optional, Union

import numpy as np
import torch

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class QwenVLEmbeddings(EmbeddingsBase):
    """Qwen3-VL-Embedding for unified text + image embeddings.

    - Single-vector embeddings with Matryoshka truncation (64..2048 for the
      2B model, up to 4096 for the 8B model).
    - Query vs. document prompts for retrieval asymmetry.

    Example:
        ```python
        embeddings = QwenVLEmbeddings(truncate_dim=1024)

        text_result = embeddings.get_embeddings(["What is machine learning?"], is_query=True)
        image_result = embeddings.get_image_embeddings(["path/to/image.jpg"])
        ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike] = "Qwen/Qwen3-VL-Embedding-2B",
        model_version: Optional[str] = None,
        task: str = "retrieval",
        truncate_dim: int = 2048,
        use_gpu: bool = True,
        batch_size: int = 4,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Initialize QwenVLEmbeddings.

        Args:
            model_name_or_path: Model name from HuggingFace or local path.
            model_version: Model version tag/branch/commit.
            task: Kept for interface compatibility with the retired jina-v4
                wrapper; Qwen3-VL-Embedding uses instruction prompts instead of
                task adapters, so this only affects logging.
            truncate_dim: Output dimension (Matryoshka truncation, 64..model max).
            use_gpu: Whether to use GPU acceleration.
            batch_size: Batch size for encoding.
            use_auth_token: HuggingFace auth token for private models.
            devices: Specific devices to use.
            show_error: Whether to show detailed errors.
            torch_dtype: Torch dtype (default float16 on GPU).
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing QwenVLEmbeddings: {model_name_or_path}")

        self.task = task
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.show_error = show_error
        self._torch_dtype = torch_dtype or torch.float16

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices not supported, using first device %s",
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        # Zoo entries override when present; hub models load automatically
        # through sentence-transformers/HF otherwise.
        resolved_path = ModelRegistry.get(
            model_name_or_path,
            version=model_version,
            raise_exceptions_for_missing_entries=False,
            **registry_kwargs,
        )
        if resolved_path is None:
            resolved_path = str(model_name_or_path)

        self.logger.info(f"Resolved model path: {resolved_path}")
        self.model = self._load_model(resolved_path)

        max_dim = self.model.get_embedding_dimension()
        if self.truncate_dim > max_dim:
            self.logger.warning(
                f"truncate_dim {self.truncate_dim} exceeds model dimension {max_dim}; using {max_dim}."
            )
            self.truncate_dim = max_dim
        self.model.truncate_dim = (
            self.truncate_dim if self.truncate_dim < max_dim else None
        )

        self.logger.info(
            f"QwenVLEmbeddings initialized: dim={self.truncate_dim}, device={self.device}"
        )

    def _load_model(self, model_path: str):
        """Load the model through sentence-transformers (Qwen-maintained code)."""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading model from {model_path}")
            return SentenceTransformer(
                model_path,
                trust_remote_code=True,
                device=str(self.device),
                model_kwargs={"dtype": self._torch_dtype},
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _encode(self, inputs, prompt_name: Optional[str]) -> np.ndarray:
        embeddings = self.model.encode(
            inputs,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def get_embeddings(
        self,
        texts: List[str],
        truncation: bool = None,
        max_length: int = None,
        is_query: bool = False,
    ) -> EmbeddingsObject:
        """Generate embeddings for text content.

        Args:
            texts: List of text strings to embed.
            truncation: Unused; kept for interface compatibility.
            max_length: Unused; kept for interface compatibility.
            is_query: If True, use the query prompt; otherwise the document prompt.

        Returns:
            EmbeddingsObject containing embeddings and token count.
        """
        if not texts:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        with torch.no_grad():
            try:
                result = EmbeddingsObject()
                result.embeddings = self._encode(
                    texts, prompt_name="query" if is_query else "document"
                )
                result.total_tokens = -1
                return result
            except Exception as e:
                self.logger.error(
                    f"Error during text embedding: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def get_image_embeddings(
        self,
        images: List[str],
    ) -> EmbeddingsObject:
        """Generate embeddings for images in the same vector space as text.

        Args:
            images: List of image paths or URLs.

        Returns:
            EmbeddingsObject containing image embeddings.
        """
        if not images:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        with torch.no_grad():
            try:
                result = EmbeddingsObject()
                result.embeddings = self._encode(
                    [{"image": image} for image in images], prompt_name="document"
                )
                result.total_tokens = 0
                return result
            except Exception as e:
                self.logger.error(
                    f"Error during image embedding: {e}", exc_info=self.show_error
                )
                return EmbeddingsObject()

    def embed_text(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed text and return the raw numpy array."""
        result = self.get_embeddings(texts, is_query=is_query)
        return result.embeddings

    def embed_images(
        self,
        images: List[str],
    ) -> np.ndarray:
        """Embed images and return the raw numpy array."""
        result = self.get_image_embeddings(images)
        return result.embeddings

    @property
    def supports_image_embeddings(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.truncate_dim
