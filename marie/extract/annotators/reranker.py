from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None
    ):
        self.model = CrossEncoder(model_name, device=device)

    def score(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """
        pairs must be (query, passage) -> here: (window_text, reference_text)
        Returns float np.array; many CE heads already return relevance scores.
        We'll NOT apply an extra sigmoid here.
        """
        scores = self.model.predict(pairs, batch_size=batch_size)
        return np.asarray(scores, dtype=np.float32)
