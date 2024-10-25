from abc import ABC, abstractmethod
from typing import Optional, Set, Union

from marie.subzero.structures.unstructured_document import UnstructuredDocument


class BaseReader(ABC):
    """
    This class is a base class for reading documents of any formats.
    It allows to check if the specific reader can read the document of some format and
    to get document's text with metadata

    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        recognized_extensions: Optional[Set[str]] = None,
        recognized_mimes: Optional[Set[str]] = None
    ) -> None:
        """
        :param config: configuration of the reader, e.g. logger for logging
        """

        self.config = {} if config is None else config

    @abstractmethod
    def read(
        self, src: Union[str, dict], parameters: Optional[dict] = None
    ) -> UnstructuredDocument:
        """
        This method reads the document and returns the document's text with metadata.
        :param src: document to read, can be a path to the file or a dictionary with the document's structure
        :param parameters: additional parameters for the reader
        :return: document's text with metadata
        """
        pass
