from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from marie.subzero.models.base import SelectorSet
from marie.subzero.pydantic_models.base import (
    Combinator,
    Margin,
    Perimeter,
    SelectionType,
    Selector,
)
from marie.subzero.pydantic_models.results import ScanResult


class RowExtractionStrategy(str, Enum):
    PRIMARY_COLUMN_VARIABLE = "Primary Column / Variable Length Ordinals"
    PRIMARY_COLUMN_FIXED = "Primary Column / Fixed Length Ordinals"
    COMPOSITE_FIXED = "Composite Rows /  Fixed Length Ordinals"

    @property
    def label(self) -> str:
        return self.value


class CutpointStrategy(str, Enum):
    START_ON_STOP = ("Restart On Starts", "Description for START_ON_STOP")
    STOP_ON_STOP = ("Start After Stop", "Description for STOP_ON_STOP")
    STOP_ON_PAGE_BREAK = ("Stop on Page Break", "Description for STOP_ON_PAGE_BREAK")
    DYNAMIC = ("Dynamic", "Description for DYNAMIC")
    PATTERN_DEFINITION = ("Pattern Definition", "Description for PATTERN_DEFINITION")

    def __new__(cls, value, description):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    @property
    def label(self) -> str:
        return self.value


class MappingType(str, Enum):
    OR = "OR"
    AND = "AND"
    REFERENCE = "REFERENCE"
    VIRTUAL = "VIRTUAL"


class FieldMapping(BaseModel):
    margin: Optional['Margin'] = None
    selector_set: Optional[SelectorSet] = None
    required: bool = False
    search_perimeter: Optional[Perimeter] = None
    primary: bool = False
    name: Optional[str] = None
    functions: Optional[List[str]] = None
    ref_field_name: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name} : {self.search_perimeter}"


class Constraint(BaseModel):
    """
    Represents a constraint that can be applied to a field mapping or a layer.
    """

    name: str
    value: str

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self) -> str:
        return f"Constraint{{name='{self.name}', value='{self.value}'}}"


class Layer(BaseModel):
    search_perimeter: Optional[Perimeter] = None
    row_extraction_strategy: RowExtractionStrategy = (
        RowExtractionStrategy.PRIMARY_COLUMN_VARIABLE
    )
    cutpoint_strategy: CutpointStrategy = CutpointStrategy.START_ON_STOP
    field_mappings: List[FieldMapping] = []
    non_repeating_field_mappings: List[FieldMapping] = []
    document_level_field_mappings: List[FieldMapping] = []
    start_selector_set: SelectorSet = SelectorSet()
    stop_selector_sets: List[SelectorSet] = []
    continuation_selector_set: SelectorSet = SelectorSet()
    image_alignment_anchor_selector_set: SelectorSet = SelectorSet()
    inner_layers: List['Layer'] = []
    parent_layer_identifier: Optional[str] = None
    selection_type: SelectionType = SelectionType.POSITIVE
    layer_name: Optional[str] = None
    # field_anchors: List['FieldAnchor'] = []
    constraints: List['Constraint'] = []

    class Config:
        arbitrary_types_allowed = True


class Template(BaseModel):
    id: int
    version: int
    layers: Optional[List['Layer']] = None
    name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def set_layers(self, layers: List['Layer']):
        self.layers = layers

    def add_layer(self, layer: 'Layer'):
        if self.layers is None:
            self.layers = []
        if layer.parent_layer_identifier is not None:
            layer.color_index = 999
        elif layer.color_index <= 0:
            if len(self.layers) > 0:
                layer.color_index = self.layers[-1].color_index + 1
            else:
                layer.color_index = 1
        self.layers.append(layer)

    def remove_layer(self, layer: 'Layer') -> bool:
        if self.layers is None:
            return False
        return self.layers.remove(layer)

    def remove_item(self, obj: object):
        if not self.layers.remove(obj):
            for layer in self.layers:
                layer.delete(obj)

    def layer_color_index(self):
        index = 1
        if not self.layers:
            return
        for layer in self.layers:
            if layer.parent_layer_identifier is not None:
                layer.color_index = 999
            elif layer.color_index <= 0:
                layer.color_index = index
                index += 1
            else:
                index = layer.color_index


class ExecutionContext(BaseModel):
    template: Optional['Template'] = None
    # pages: List['GrapnelPage'] = []
    tree: Optional[Any] = None
    doc_id: str
    # selector_hits: List['ScanResult'] = []
    metadata: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def set_selector_hits(self, results: List[ScanResult]):
        self.selector_hits.extend(results)

    def get_selector_hits(self) -> List[ScanResult]:
        return self.selector_hits
