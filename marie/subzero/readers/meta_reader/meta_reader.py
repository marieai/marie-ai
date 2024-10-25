from typing import Optional, Union

from marie.subzero.readers.base import BaseReader
from marie.subzero.structures.unstructured_document import UnstructuredDocument


class MetaReader(BaseReader):
    """
    This reader allows handling of Metadata from marie
    """

    def __init__(self, *, config: Optional[dict] = None) -> None:
        super().__init__(config=config)

    def read(
        self, src: Union[str, dict], parameters: Optional[dict] = None
    ) -> UnstructuredDocument:
        parameters = {} if parameters is None else parameters

        return None

    def __get_text(self, value: Any) -> str:  # noqa
        if isinstance(value, (dict, list)) or value is None:
            return ""

        return str(value)
