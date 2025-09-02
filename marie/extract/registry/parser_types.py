from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from omegaconf import OmegaConf

from marie.extract.schema import ExtractionResult
from marie.extract.structures import UnstructuredDocument

ParserFn = Callable[
    [UnstructuredDocument, str, str, OmegaConf], Optional[ExtractionResult]
]


@runtime_checkable
class ParserProto(Protocol):
    def parse(
        self, doc: UnstructuredDocument, working_dir: str, src_dir: str, conf: OmegaConf
    ) -> Optional[ExtractionResult]: ...


TParser = TypeVar("TParser", ParserFn, ParserProto, Type[ParserProto])
