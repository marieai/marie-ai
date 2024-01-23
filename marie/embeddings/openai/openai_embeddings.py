from typing import List

from marie.embeddings.base import EmbeddingsBase


class OpenAIEmbeddings(EmbeddingsBase):
    """
    OpenAI Embeddings
    """

    def get_embeddings(self, data: str, **kwargs) -> List[float]:
        """
        Generate embedding
        :param data:
        :param kwargs:
        """
        raise NotImplementedError
