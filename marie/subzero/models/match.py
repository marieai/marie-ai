from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from marie.subzero.models.base import Location, Page, PageDetails, Rectangle
from marie.subzero.models.definition import Layer, RowExtractionStrategy


class MatchSectionVisitor(ABC):
    @abstractmethod
    def visit(self, result: "MatchSection") -> None:
        """
        :param result: An instance of MatchSection containing the result data.
        :return: None
        """
        pass


class ResultType(str, Enum):
    ANCHOR = "ANCHOR"
    BLOB = "BLOB"


class ScanResult(BaseModel):
    owner_identifier: Optional[str] = None
    page: int
    type: ResultType
    area: Rectangle
    confidence: float
    x_offset: int
    y_offset: int
    selection_type: str = "POSITIVE"

    def __str__(self) -> str:
        return f"Owner Identifier: {self.owner_identifier}, Page: {self.page}, Type: {self.type}, Area: {self.area}, Confidence: {self.confidence}, X Offset: {self.x_offset}, Y Offset: {self.y_offset}, Selection Type: {self.selection_type}"


class ScoredMatchResult(BaseModel):
    score: float
    items: List[ScanResult]
    index: int
    candidates: List[ScanResult]

    def __str__(self) -> str:
        return f"Score: {self.score}, Items: {self.items}, Index: {self.index}, Candidates: {self.candidates}"


class LocationType(str, Enum):
    START = "START"
    STOP = "STOP"
    CONTINUE = "CONTINUE"


class TypedScanResult(ScanResult):
    def __init__(
        self,
        input: ScanResult,
        location_type: LocationType,
    ):
        super().__init__(input)
        self.location_type = location_type

    @staticmethod
    def wrap(
        candidates: List[ScanResult], location_type: LocationType
    ) -> List["TypedScanResult"]:
        if not candidates:
            return []
        return [TypedScanResult(candidate, location_type) for candidate in candidates]

    def __str__(self) -> str:
        return f"{self.location_type} - {super().__str__()}"


class Span(BaseModel):
    page: int
    y: int
    h: int
    msg: Optional[str] = None

    def __str__(self) -> str:
        return f"page = {self.page}, y = {self.y}, h = {self.h}, msg = {self.msg}"


class MatchField(BaseModel):
    owner_field_identifier: Optional[str] = None
    fid: int
    data: Optional[str] = None
    scan_result: Optional[ScanResult] = None

    def __str__(self) -> str:
        return f"Field ID {self.owner_field_identifier} {self.scan_result}"


class MatchSectionType(str, Enum):
    WRAPPER = "WRAPPER"
    CONTENT = "CONTENT"
    REJECTED = "REJECTED"


class MatchFieldRow(BaseModel):
    fields: Optional[List[MatchField]] = Field(default_factory=list)
    children: List["MatchFieldRow"] = Field(default_factory=list)
    details: Optional[PageDetails] = None

    def __str__(self) -> str:
        builder = [self.__class__.__name__]
        if self.fields:
            for field in self.fields:
                grs = field.scan_result
                builder.append(f"  > {grs}")
        if self.children:
            for child in self.children:
                builder.append(f"\n               {child}")
        return "".join(builder)


class MatchSection(BaseModel):
    sections: List["MatchSection"] = []
    parent: Optional["MatchSection"] = None
    type: MatchSectionType = MatchSectionType.CONTENT
    start_candidates: Optional[List["ScanResult"]] = None
    stop_candidates: Optional[List["ScanResult"]] = None

    # fields: List['MatchField'] = []
    # matched_field_rows: List['MatchFieldRowModel'] = []
    # matched_non_repeating_fields: List['MatchField'] = []
    # matched_document_level_fields: List['MatchField'] = []

    # Start location of this section, they do not necessarily equal the start and stop of spans
    start: Optional[Location] = None
    stop: Optional[Location] = None

    span: Optional[List["Span"]] = None

    label: str = "NO-LABEL"
    x_offset: int = 0
    y_offset: int = 0

    row_extraction_strategy: Optional[RowExtractionStrategy] = None
    owner_layer: Optional[Layer] = None

    pages: Optional[List['Page']] = None

    def __str__(self) -> str:
        return f"{self.label} : Sections [ start [ {self.start} ] stop [ {self.stop} ] size = {len(self.sections)}  span -> {self.span}"

    def set_pages(self, pages: List[Page]) -> None:
        self.pages = pages

    def visit(self, visitor: "MatchSectionVisitor") -> None:
        """
        :param visitor: An instance of MatchSectionVisitor that will be used to invoke the visit method.
        :return: None
        """
        visitor.visit(self)


class SubzeroResult(MatchSection):
    """
    Matched document information
    """

    def __init__(self, label: Optional[str] = None):
        super().__init__(label=label)
        self.type = MatchSectionType.WRAPPER
