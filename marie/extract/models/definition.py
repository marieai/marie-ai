from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from marie.extract.models.base import (
    CutpointStrategy,
    Margin,
    Page,
    Perimeter,
    RowExtractionStrategy,
    SelectionType,
    SelectorSet,
)


class FieldScope(str, Enum):
    """
    Specifies the scope of a field's application within the document structure.

    Attributes:
        DOCUMENT: The field applies to the entire document.
        LAYER: The field is confined to a specific layer.
        REGION: The field is relevant only within a designated region of a layer.
    """

    DOCUMENT = "DOCUMENT"
    LAYER = "LAYER"
    REGION = "REGION"


class FieldCardinality(str, Enum):
    """
    Defines how many times a field is expected to appear.

    Attributes:
        SINGLE: The field is expected to appear only once.
        MULTIPLE: The field can appear multiple times.
    """

    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"


class MappingType(str, Enum):
    OR = "OR"
    AND = "AND"
    REFERENCE = "REFERENCE"
    VIRTUAL = "VIRTUAL"


class FieldMapping(BaseModel):
    margin: Optional["Margin"] = None
    selector_set: Optional[SelectorSet] = None
    required: bool = False
    search_perimeter: Optional[Perimeter] = None
    primary: bool = False
    name: Optional[str] = None
    functions: Optional[List[str]] = None
    ref_field_name: Optional[str] = None
    field_def: Optional[Dict[str, Any]] = None

    scope: FieldScope = FieldScope.LAYER
    cardinality: FieldCardinality = FieldCardinality.SINGLE
    role: Optional[str] = None
    min_confidence: float = 0.7

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
    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

    search_perimeter: Optional[Perimeter] = None
    row_extraction_strategy: RowExtractionStrategy = (
        RowExtractionStrategy.PRIMARY_COLUMN_VARIABLE
    )
    cutpoint_strategy: CutpointStrategy = CutpointStrategy.START_ON_STOP

    field_mappings: List[FieldMapping] = []
    non_repeating_field_mappings: List[FieldMapping] = []
    document_level_field_mappings: List[FieldMapping] = []

    # FIXME : this is initial implementationi to replace the mappings above
    fields: List[FieldMapping] = []

    # converted to selector sets
    start_selector_sets: List[SelectorSet] = []
    stop_selector_sets: List[SelectorSet] = []
    continuation_selector_sets: List[SelectorSet] = []

    layers: List["Layer"] = []
    parent_layer_identifier: Optional[str] = None
    selection_type: SelectionType = SelectionType.POSITIVE
    layer_name: Optional[str] = None
    # field_anchors: List['FieldAnchor'] = []
    constraints: List["Constraint"] = []
    color_index: int = 0

    # this is used to store the raw table config
    # TODO : Replace with a proper table config class
    table_config_raw: Optional[Dict[str, Any]] = (
        None  # DEPRECATED use regions_config_raw
    )
    regions_config_raw: Optional[Dict[str, Any]] = None


class Template(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tid: str
    version: int
    layers: Optional[List["Layer"]] = None
    name: str = ""

    def set_layers(self, layers: List["Layer"]):
        self.layers = layers

    def add_layer(self, layer: "Layer"):
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

    def remove_layer(self, layer: "Layer") -> bool:
        if self.layers is None:
            return False
        return self.layers.remove(layer)

    def remove_item(self, obj: object):
        if not self.layers.remove(obj):
            for layer in self.layers:
                layer.delete(obj)


class WorkUnit(BaseModel):
    doc_id: str
    template: Optional[Template] = None
    metadata: Optional[List] = None
    frames: Optional[List] = None
