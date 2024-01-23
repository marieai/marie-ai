from abc import ABC, abstractmethod
from typing import List

from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.logging.logger import MarieLogger


class EmbeddingsBase(ABC):
    """
    Base class for embeddings
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__).logger

    @abstractmethod
    def get_embeddings(
        self, texts: List[str], truncation: bool = None, max_length: int = None
    ) -> EmbeddingsObject:
        """
        Generate embedding for the given texts
        :param texts:  A list of texts as a list of strings, such as ["I like cats", "I also like dogs"]
        :param truncation: Whether to truncate the input texts to fit within the context length limit of our embedding models.
            - If True, over-length input texts will be truncated to fit within the context length, before encoded by the embedding model.
            - If False, an error will be raised if any given texts exceeds the context length.
            - If not specified (defaults to None), we will truncate the input text before sending it to the embedding model if it slightly exceeds the context window length.
        :param max_length: The maximum length of the sequence to be encoded. If None, will default to the maximum input length of the model.
        :return: An EmbeddingsObject containing the embeddings and the metadata
        """
        pass
