from typing import List, Optional


class EmbeddingsObject:
    """
    Embeddings object to store the embeddings and total number of tokens
    """

    def __init__(self, embeddings: Optional[List[float]] = None, total_tokens: int = 0):
        self.embeddings = embeddings or []
        self.total_tokens = total_tokens

    def __str__(self):
        return f"EmbeddingsObject(embeddings={self.embeddings}, total_tokens={self.total_tokens})"
