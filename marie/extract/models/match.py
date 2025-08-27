from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from marie.extract.models.base import Location, Page, PageDetails, Rectangle
from marie.extract.models.definition import Layer, RowExtractionStrategy
from marie.extract.models.span import Span
from marie.extract.structures.line_with_meta import LineWithMeta

# Covnerted from pydantic to dataclass as it is not used for serialization or validation
# we will render the dataclass to json when needed


class MatchSectionVisitor(ABC):
    @abstractmethod
    def visit(self, result: "MatchSection") -> None:
        pass


class ResultType(str, Enum):
    ANCHOR = "ANCHOR"
    BLOB = "BLOB"
    TEXT = "TEXT"
    ANNOTATION = "ANNOTATION"
    CUTPOINT = "CUTPOINT"


@dataclass
class ScanResult:
    owner_identifier: Optional[str] = None
    page: int = 0
    type: ResultType = ResultType.TEXT
    area: Rectangle = None
    confidence: float = 0.0
    x_offset: int = 0
    y_offset: int = 0
    selection_type: str = "POSITIVE"
    line: Optional[LineWithMeta] = None


@dataclass
class ScoredMatchResult:
    score: float = 0.0
    items: List[ScanResult] = field(default_factory=list)
    index: int = 0
    candidates: List[ScanResult] = field(default_factory=list)


class LocationType(str, Enum):
    START = "START"
    STOP = "STOP"
    CONTINUATION = "CONTINUATION"


@dataclass
class TypedScanResult(ScanResult):
    location_type: LocationType = LocationType.START

    @staticmethod
    def wrap(
        candidates: List[ScanResult], location_type: LocationType
    ) -> List["TypedScanResult"]:
        if not candidates:
            return []
        return [
            TypedScanResult(location_type=location_type, **vars(candidate))
            for candidate in candidates
        ]


@dataclass
class MatchField:
    owner_field_identifier: Optional[str] = None
    fid: int = 0
    data: Optional[str] = None
    scan_result: Optional[ScanResult] = None


# this class is used to store the field final extraction result
@dataclass
class Field:
    field_name: Optional[str] = field(default=None)
    field_type: Optional[str] = field(default=None)
    is_required: bool = field(default=False)
    composite_field: bool = field(default=False)
    value: Optional[str] = field(default=None)
    x: int = field(default=0)
    y: int = field(default=0)
    width: int = field(default=0)
    height: int = field(default=0)
    date_format: Optional[str] = field(default=None)
    name_format: Optional[str] = field(default=None)
    column_name: Optional[str] = field(default=None)
    page: int = field(default=0)
    xdpi: int = field(default=0)
    ydpi: int = field(default=0)
    confidence: float = field(default=0.0)
    scrubbed: bool = field(default=False)
    uuid: Optional[str] = field(default=None)
    reference_uuid: Optional[str] = field(default=None)
    layer_name: Optional[str] = field(default=None)
    value_original: Optional[str] = field(default=None)


class MatchSectionType(str, Enum):
    WRAPPER = "WRAPPER"
    CONTENT = "CONTENT"
    REJECTED = "REJECTED"


@dataclass
class MatchFieldRow:
    # fields: List[MatchField] = field(default_factory=list)
    fields: List[Field] = field(default_factory=list)
    children: List["MatchFieldRow"] = field(default_factory=list)


@dataclass
class MatchSection:
    sections: List["MatchSection"] = field(default_factory=list)
    parent: Optional["MatchSection"] = None
    type: MatchSectionType = MatchSectionType.CONTENT
    start_candidates: Optional[List["ScanResult"]] = None
    stop_candidates: Optional[List["ScanResult"]] = None
    start: Optional[Location] = None
    stop: Optional[Location] = None
    span: Optional[List["Span"]] = field(default_factory=list)
    label: str = "NO-LABEL"
    x_offset: int = 0
    y_offset: int = 0
    row_extraction_strategy: Optional[RowExtractionStrategy] = None
    owner_layer: Optional[Layer] = None
    pages: Optional[List["Page"]] = None

    # moving away from the old way of storing the page number and y position
    matched_type: str = "LINE"  # LINE, COORDINATE

    # NEED TO HAVE BETTER way to attach results with provided schema
    # Moving away from MatchedField and using Field directly - too much abstraction
    matched_non_repeating_fields: Optional[List["Field"]] = None
    matched_field_rows: Optional[List["MatchFieldRow"]] = None

    # Unifying matched fields
    fields: List[Field] = field(default_factory=list)
    field_rows: List[MatchFieldRow] = field(default_factory=list)

    def set_pages(self, pages: List[Page]) -> None:
        self.pages = pages

    def visit(self, visitor: "MatchSectionVisitor") -> None:
        visitor.visit(self)

    def add_section(self, sec: "MatchSection") -> None:
        """Safely adds a section to the SubzeroResult instance."""
        self.sections.append(sec)


@dataclass
class SubzeroResult(MatchSection):
    def __init__(self, label: Optional[str] = None):
        super().__init__(label=label)
        self.type = MatchSectionType.WRAPPER
