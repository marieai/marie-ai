from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from marie.subzero.models.base import Location, Rectangle
from marie.subzero.models.definition import Layer, RowExtractionStrategy


class ResultType(str, Enum):
    ANCHOR = "ANCHOR"
    BLOB = "BLOB"


class PageDetails(BaseModel):
    page_index: int
    w: int
    h: int

    def __str__(self) -> str:
        return f"{self.w}, {self.h}"


class Page(BaseModel):
    page_number: int
    image: Optional[str] = None  # Assuming image is a file path
    details: Optional[PageDetails] = None

    def __str__(self) -> str:
        return f"Page Number: {self.page_number}, Image: {self.image}, Details: {self.details}"


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


class MatchSectionType(str):
    WRAPPER = "WRAPPER"
    CONTENT = "CONTENT"
    REJECTED = "REJECTED"


class MatchFieldRow(BaseModel):
    fields: Optional[List[MatchField]] = Field(default_factory=list)
    children: List['MatchFieldRow'] = Field(default_factory=list)
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
        return ''.join(builder)


class MatchSection(BaseModel):
    sections: List['MatchSection'] = []
    parent: Optional['MatchSection'] = None
    type: MatchSectionType = MatchSectionType.CONTENT
    start_candidates: Optional[List['ScanResult']] = None
    stop_candidates: Optional[List['ScanResult']] = None

    # fields: List['MatchField'] = []
    # matched_field_rows: List['MatchFieldRowModel'] = []
    # matched_non_repeating_fields: List['MatchField'] = []
    # matched_document_level_fields: List['MatchField'] = []

    start: Optional[Location] = None
    stop: Optional[Location] = None
    label: str = "NO-LABEL"
    x_offset: int = 0
    y_offset: int = 0

    row_extraction_strategy: Optional[RowExtractionStrategy] = None
    owner_layer: Optional[Layer] = None

    # pages: Optional[List['PageModel']] = None

    def __str__(self) -> str:
        return f"{self.label} : Sections [ start [ {self.start} ] stop [ {self.stop} ] size = {len(self.sections)} startSelectorSetOwnerIdentifier : {self.start_selector_set_owner_identifier} stopSelectorSetOwnerIdentifier : {self.stop_selector_set_owner_identifier}] -> {self.span}"
