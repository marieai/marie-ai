"""Google Gemini Embedding 2 for text and multimodal embeddings.

This module provides embedding capabilities using Google's Gemini Embedding 2 API,
which supports text, images, video, audio, and PDFs.

Key features:
- Configurable output dimensionality: 768, 1536, 3072
- Task-specific embedding: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, etc.
- Multimodal support: text, images, video, audio, PDFs
- API-based (requires Google API key)

Reference: https://ai.google.dev/gemini-api/docs/embeddings
"""

import base64
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np

from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging_core.logger import MarieLogger

TaskType = Literal[
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
]

VALID_DIMENSIONS = [768, 1536, 3072]
DEFAULT_DIMENSION = 3072
DEFAULT_BATCH_SIZE = 100
DEFAULT_MODEL = "gemini-embedding-2-preview"


class GeminiEmbeddings(EmbeddingsBase):
    """Google Gemini Embedding 2 for multimodal embeddings.

    This implementation uses Google's Gemini Embedding 2 API which provides
    embeddings for text, images, video, audio, and PDFs in a unified space.

    Key features:
    - Configurable dimensions: 768, 1536, or 3072
    - Task types for optimized embeddings
    - Multimodal: embeds text AND images (and other media)

    Example:
        ```python
        embeddings = GeminiEmbeddings(
            api_key="your-api-key",  # Or set GOOGLE_API_KEY env var
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Embed text
        text_result = embeddings.get_embeddings(
            ["What is machine learning?"],
            is_query=True,  # Switches to RETRIEVAL_QUERY
        )

        # Embed images
        image_result = embeddings.get_image_embeddings(["path/to/image.jpg"])
        ```
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        output_dimensionality: int = DEFAULT_DIMENSION,
        task_type: TaskType = "RETRIEVAL_DOCUMENT",
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_error: bool = True,
        **kwargs,
    ):
        """Initialize GeminiEmbeddings.

        Args:
            model_name: Gemini embedding model name. Default: "gemini-embedding-2-preview"
            api_key: Google API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY env vars.
            output_dimensionality: Output dimension. Options: 768, 1536, 3072 (default).
            task_type: Task type for embedding optimization. Options:
                - "RETRIEVAL_DOCUMENT": For documents in search/RAG (default)
                - "RETRIEVAL_QUERY": For search queries
                - "SEMANTIC_SIMILARITY": For semantic similarity
                - "CLASSIFICATION": For classification tasks
                - "CLUSTERING": For clustering tasks
            batch_size: Batch size for API calls (max 100).
            show_error: Whether to show detailed errors.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing GeminiEmbeddings: {model_name}")

        self.model_name = model_name
        self.task_type = task_type
        self.batch_size = min(batch_size, DEFAULT_BATCH_SIZE)
        self.show_error = show_error

        # Validate and set output dimensionality
        if output_dimensionality not in VALID_DIMENSIONS:
            self.logger.warning(
                f"output_dimensionality {output_dimensionality} not in valid dimensions {VALID_DIMENSIONS}. "
                f"Using closest valid dimension."
            )
            self.output_dimensionality = min(
                VALID_DIMENSIONS, key=lambda x: abs(x - output_dimensionality)
            )
        else:
            self.output_dimensionality = output_dimensionality

        # Resolve API key
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google API key required. Provide via api_key parameter or set "
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        # Initialize the client
        self._init_client()

        self.logger.info(
            f"GeminiEmbeddings initialized: model={model_name}, "
            f"dim={self.output_dimensionality}, task={task_type}"
        )

    def _init_client(self):
        """Initialize the Google GenAI client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. "
                "Install with: uv add google-generativeai>=0.5.0"
            )

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
            truncation: Not used (API handles truncation automatically).
            max_length: Not used (API handles length limits).
            is_query: If True, use RETRIEVAL_QUERY task type.
                     If False, use the configured task_type.

        Returns:
            EmbeddingsObject containing embeddings and token count.
        """
        if not texts:
            return EmbeddingsObject(embeddings=[], total_tokens=0)

        # Switch task type for queries
        task_type = "RETRIEVAL_QUERY" if is_query else self.task_type

        try:
            all_embeddings = []
            total_tokens = 0

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                result = self._genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=batch,
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality,
                )

                # Handle batch response
                if hasattr(result, "embeddings"):
                    # Batch response - list of embedding objects
                    for emb in result.embeddings:
                        all_embeddings.append(emb.values)
                elif hasattr(result, "embedding"):
                    # Single response
                    all_embeddings.append(result.embedding.values)

            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)

            result = EmbeddingsObject()
            result.embeddings = embeddings_array
            result.total_tokens = total_tokens

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

        try:
            all_embeddings = []

            for image_path in images:
                # Load image as base64 or from URL
                image_content = self._load_image(image_path)

                result = self._genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=image_content,
                    task_type=self.task_type,
                    output_dimensionality=self.output_dimensionality,
                )

                if hasattr(result, "embedding"):
                    all_embeddings.append(result.embedding.values)

            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)

            result = EmbeddingsObject()
            result.embeddings = embeddings_array
            result.total_tokens = 0

            return result

        except Exception as e:
            self.logger.error(
                f"Error during image embedding: {e}", exc_info=self.show_error
            )
            return EmbeddingsObject()

    def _load_image(self, image_path: str) -> dict:
        """Load an image for embedding.

        Args:
            image_path: Path to local image or URL.

        Returns:
            Image content dict for the API.
        """
        from PIL import Image as PILImage

        if image_path.startswith(("http://", "https://")):
            # URL - let API handle it
            return {"image_url": image_path}

        # Local file - load and encode
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read and encode image
        with open(path, "rb") as f:
            image_data = f.read()

        # Determine mime type
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        # Return as inline data
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_data).decode("utf-8"),
            }
        }

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
            Numpy array of embeddings, shape (len(texts), output_dimensionality).
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
            Numpy array of embeddings, shape (len(images), output_dimensionality).
        """
        result = self.get_image_embeddings(images)
        return result.embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.output_dimensionality
