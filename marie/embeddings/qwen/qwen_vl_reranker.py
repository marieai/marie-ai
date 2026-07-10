import os
from typing import Any, Dict, List, Optional, Union

import torch

from marie.constants import __model_path__
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.registry.model_registry import ModelRegistry


class QwenVLReranker:
    """Qwen3-VL-Reranker: multimodal cross-encoder reranking.

    Companion to :class:`~marie.embeddings.qwen.QwenVLEmbeddings` — rerank the
    candidates retrieved by embedding search. Queries and documents may be
    text, image paths/URLs, or dicts mixing ``{"text": ..., "image": ...}``.
    Apache-2.0, loads through sentence-transformers' CrossEncoder (remote code
    maintained by the Qwen team).

    Example:
        ```python
        reranker = QwenVLReranker()
        ranked = reranker.rank(
            "claim number for patient John Smith",
            ["claim 2Z5L8Q7 John Smith", "invoice 998-A cafeteria supplies"],
            top_k=1,
        )
        # [{"index": 0, "score": 0.93, "document": "claim 2Z5L8Q7 John Smith"}]
        ```
    """

    DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike] = "Qwen/Qwen3-VL-Reranker-2B",
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 8,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing QwenVLReranker: {model_name_or_path}")

        self.batch_size = batch_size
        self.show_error = show_error
        self._torch_dtype = torch_dtype or torch.float16

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
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

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            resolved_path,
            trust_remote_code=True,
            device=str(self.device),
            model_kwargs={"dtype": self._torch_dtype},
        )
        self.logger.info(f"QwenVLReranker initialized on {self.device}")

    def score(
        self,
        query: Any,
        documents: List[Any],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Score (query, document) relevance for each document.

        Returns sigmoid-activated scores in [0, 1], one per document, in the
        input order.
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        with torch.no_grad():
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                activation_fn=torch.nn.Sigmoid(),
                prompt=instruction or self.DEFAULT_INSTRUCTION,
            )
        return [float(s) for s in scores]

    def rank(
        self,
        query: Any,
        documents: List[Any],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to the query.

        Returns ``[{"index", "score", "document"}, ...]`` sorted by descending
        score, truncated to ``top_k`` if given.
        """
        scores = self.score(query, documents, instruction=instruction)
        ranked = sorted(
            (
                {"index": i, "score": s, "document": d}
                for i, (s, d) in enumerate(zip(scores, documents))
            ),
            key=lambda r: r["score"],
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked
