from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Point2D(BaseModel):
    """
    Represents a point in 2D space with X and Y coordinates.
    """

    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x},{self.y}"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Point2D):
            return False
        return self.x == other.x and self.y == other.y


class Rectangle(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Location(BaseModel):
    page: int
    y: int

    def __str__(self) -> str:
        return f"page = {self.page} y = {self.y}"


class SelectionType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATION = "NEGATION"


class Combinator(str, Enum):
    DESCENDANT = "DESCENDANT"
    CHILD = "CHILD"
    ADJACENT_SIBLING = "ADJACENT_SIBLING"
    GENERAL_SIBLING = "GENERAL_SIBLING"


class Margin(BaseModel):
    left: int
    right: int
    top: int
    bottom: int


class Perimeter(BaseModel):
    x_min: float
    x_max: float

    def __init__(self, x_min: float, x_max: float, expansion_factor: float = 0):
        expansion = (x_max - x_min) * expansion_factor
        super().__init__(x_min=max(x_min - expansion, 0), x_max=x_max + expansion)

    @staticmethod
    def union(src1: 'Perimeter', src2: 'Perimeter') -> 'Perimeter':
        x1 = min(src1.x_min, src2.x_min)
        x2 = max(src1.x_max, src2.x_max)
        return Perimeter(x1, x2)

    def __str__(self) -> str:
        return f"Perimeter : xmin = {self.x_min} xmax = {self.x_max}"


class Dimension(BaseModel):
    width: float
    height: float

    def set_size(self, width: float, height: float):
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"{self.width},{self.height}"


class Blob(BaseModel):
    """
    Represents a Blob with coordinates and dimensions on a specific page.
    """

    x: int
    y: int
    w: int
    h: int
    page: int

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self) -> str:
        return f"Page {self.page} : x, y, w, h [{self.x}, {self.y}, {self.w}, {self.h}]"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Blob):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.w == other.w
            and self.h == other.h
            and self.page == other.page
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.w, self.h, self.page))


class Selector(BaseModel):
    combinator: Combinator = Combinator.DESCENDANT
    selection_type: Optional[SelectionType] = SelectionType.POSITIVE
    tag: Optional[str] = None
    location: Optional[Point2D] = None
    dimension: Optional[Dimension] = None
    search_perimeter: Optional[Perimeter] = None
    cut_point_y_offset: int = 0

    def __str__(self) -> str:
        return f"Selector Loc = {self.location} Dim = {self.dimension}"


class TextSelector(Selector):
    text: str


class PatternSelector(Selector):
    pattern: str


class ImageSelector(Selector):
    data: str


class RegexSelector(Selector):
    regex: str


class SelectorSet(BaseModel):
    selectors: Optional[List[Selector]] = None

    def __str__(self) -> str:
        buffer = "SelectorSet : \n"
        if self.selectors:
            for selector in self.selectors:
                buffer += f" {selector}\n"
        return buffer

    def size(self) -> int:
        return len(self.selectors) if self.selectors else 0

    def selector(self, index: int) -> Optional[Selector]:
        if self.selectors and 0 <= index < len(self.selectors):
            return self.selectors[index]
        return None

    def selection_type_count(self, selection_type: SelectionType) -> int:
        return (
            sum(
                1
                for selector in self.selectors
                if selector.selection_type == selection_type
            )
            if self.selectors
            else 0
        )
