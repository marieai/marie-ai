"""jina-embeddings-v4 for unified text + image embeddings.

This module provides multimodal embedding capabilities using jina-embeddings-v4,
which embeds both text and images into the same vector space for unified retrieval.

Key features:
- Single embedding space for text AND images
- Task-specific adapters: retrieval, text-matching, code
- Query/passage prompts for retrieval optimization
- Matryoshka support: truncate to 128/256/512/1024/2048 dimensions
- 32K token context window

Reference: https://huggingface.co/jinaai/jina-embeddings-v4
"""

import os
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import nn

from marie.constants import __model_path__
from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger
from marie.logging_core.profile import TimeContext
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class JinaEmbeddingsV4(EmbeddingsBase):
    """Jina Embeddings v4 for unified text + image embeddings.

    This implementation supports the jina-embeddings-v4 model which provides
    a single embedding space for both text and images, making it ideal for
    multimodal RAG applications.

    Key differences from JinaEmbeddings (v2):
    - Multimodal: embeds text AND images in the same space
    - API: encode_text() / encode_image() instead of encode()
    - Task adapters: retrieval, text-matching, code
    - Query/passage prompts for retrieval
    - Matryoshka: can truncate to smaller dimensions

    Example:
        ```python
        embeddings = JinaEmbeddingsV4(
            model_name_or_path="jinaai/jina-embeddings-v4",
            task="retrieval",
            truncate_dim=1024,  # Use smaller dimension
        )

        # Embed text
        text_result = embeddings.get_embeddings(
            ["What is machine learning?"],
            is_query=True,  # Use query prompt
        )

        # Embed images (same embedding space!)
        image_result = embeddings.get_image_embeddings(["path/to/image.jpg"])

        # Can directly compare text and image embeddings
        similarity = cosine_similarity(text_result.embeddings[0], image_result.embeddings[0])
        ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike] = "jinaai/jina-embeddings-v4",
        model_version: Optional[str] = None,
        task: Literal["retrieval", "text-matching", "code"] = "retrieval",
        truncate_dim: int = 2048,
        use_gpu: bool = True,
        batch_size: int = 4,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Initialize JinaEmbeddingsV4.

        Args:
            model_name_or_path: Model name from HuggingFace or local path.
            model_version: Model version tag/branch/commit.
            task: Task adapter to use. Options:
                - "retrieval": For search/RAG (recommended for most cases)
                - "text-matching": For semantic similarity
                - "code": For code search
            truncate_dim: Output dimension. Matryoshka supports:
                128, 256, 512, 1024, 2048 (default).
                Smaller = faster but less accurate.
            use_gpu: Whether to use GPU acceleration.
            batch_size: Batch size for encoding.
            use_auth_token: HuggingFace auth token for private models.
            devices: Specific devices to use.
            show_error: Whether to show detailed errors.
            torch_dtype: Torch dtype (e.g., torch.float16 for memory efficiency).
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing JinaEmbeddingsV4: {model_name_or_path}")

        self.task = task
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.show_error = show_error
        self._torch_dtype = torch_dtype or torch.float16

        # Validate truncate_dim
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            self.logger.warning(
                f"truncate_dim {truncate_dim} not in standard Matryoshka dimensions {valid_dims}. "
                f"Using closest valid dimension."
            )
            self.truncate_dim = min(valid_dims, key=lambda x: abs(x - truncate_dim))

        # Device setup
        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices not supported, using first device %s",
                resolved_devices[0],
            )
        self.device = resolved_devices[0]

        # Resolve model path via registry
        registry_kwargs = {
            "__model_path__": __model_path__,
            "use_auth_token": use_auth_token,
        }

        resolved_path = ModelRegistry.get(
            model_name_or_path,
            version=model_version,
            raise_exceptions_for_missing_entries=True,
            **registry_kwargs,
        )

        self.logger.info(f"Resolved model path: {resolved_path}")

        # Load model
        self.model = self._load_model(resolved_path)
        self.logger.info(
            f"JinaEmbeddingsV4 initialized: task={task}, dim={self.truncate_dim}, device={self.device}"
        )

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the jina-embeddings-v4 model."""
        try:
            from transformers import AutoModel

            self.logger.info(f"Loading model from {model_path}")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self._torch_dtype,
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def get_embeddings(
        self,
        texts: List[str],
        truncation: bool = None,
        max_length: int = None,
        is_query: bool = False,
    ) -> EmbeddingsObject:
        """Generate embeddings for text content.

        Implements the EmbeddingsBase interface with additional support for
        query vs. passage distinction in retrieval tasks.

        Args:
            texts: List of text strings to embed.
            truncation: Whether to truncate long texts (default: True for jina-v4).
            max_length: Maximum sequence length.
            is_query: If True, use query prompt (shorter texts, search queries).
                     If False, use passage prompt (documents, longer texts).

        Returns:
            EmbeddingsObject containing embeddings and token count.
        """
        if not texts:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        # For retrieval task, use appropriate prompt
        prompt_name = "query" if is_query else "passage"

        with torch.no_grad():
            try:
                # jina-v4 uses encode_text method with task and prompt_name
                if hasattr(self.model, "encode_text"):
                    embeddings = self.model.encode_text(
                        texts=texts,
                        task=self.task,
                        prompt_name=prompt_name,
                        truncate_dim=self.truncate_dim,
                        batch_size=self.batch_size,
                    )
                else:
                    # Fallback for models without encode_text
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        truncation=truncation if truncation is not None else True,
                    )

                # Ensure numpy array
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                # Truncate if needed
                if embeddings.shape[-1] > self.truncate_dim:
                    embeddings = embeddings[..., : self.truncate_dim]

                result = EmbeddingsObject()
                result.embeddings = embeddings
                result.total_tokens = -1  # Token count not readily available

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
        """Generate embeddings for images.

        Images are embedded into the same vector space as text,
        enabling direct text-to-image similarity search.

        Args:
            images: List of image paths or URLs.

        Returns:
            EmbeddingsObject containing image embeddings.
        """
        if not images:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        with torch.no_grad():
            try:
                # jina-v4 uses encode_image method
                if hasattr(self.model, "encode_image"):
                    embeddings = self.model.encode_image(
                        images=images,
                        task=self.task,
                        truncate_dim=self.truncate_dim,
                    )
                else:
                    raise NotImplementedError(
                        "Model does not support encode_image. "
                        "Ensure you are using jina-embeddings-v4."
                    )

                # Ensure numpy array
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                # Truncate if needed
                if embeddings.shape[-1] > self.truncate_dim:
                    embeddings = embeddings[..., : self.truncate_dim]

                result = EmbeddingsObject()
                result.embeddings = embeddings
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
        """Convenience method to embed text and return raw numpy array.

        Args:
            texts: List of text strings.
            is_query: Whether this is a query (vs. passage/document).

        Returns:
            Numpy array of embeddings, shape (len(texts), truncate_dim).
        """
        result = self.get_embeddings(texts, is_query=is_query)
        return result.embeddings

    def embed_images(
        self,
        images: List[str],
    ) -> np.ndarray:
        """Convenience method to embed images and return raw numpy array.

        Args:
            images: List of image paths or URLs.

        Returns:
            Numpy array of embeddings, shape (len(images), truncate_dim).
        """
        result = self.get_image_embeddings(images)
        return result.embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.truncate_dim
