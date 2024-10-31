from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from marie.subzero.models.base import (
    CutpointStrategy,
    Margin,
    Page,
    Perimeter,
    RowExtractionStrategy,
    SelectionType,
    SelectorSet,
)
from marie.subzero.readers.meta_reader.meta_reader import MetaReader
from marie.subzero.structures.unstructured_document import UnstructuredDocument


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

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class Template(BaseModel):
    tid: str
    version: int
    layers: Optional[List["Layer"]] = None
    name: str = ""

    class Config:
        arbitrary_types_allowed = True

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


class ExecutionContext(BaseModel):
    template: Optional["Template"] = None
    document: Optional["UnstructuredDocument"] = None
    pages: List["Page"] = []
    tree: Optional[Any] = None
    doc_id: str
    metadata: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return (
            f"ExecutionContext(doc_id={self.doc_id}, "
            f"template={self.template}, "
            f"document={self.document}, "
            f"metadata_keys={list(self.metadata.keys()) if self.metadata else []})"
        )

    def get_template(self) -> Template:
        return self.template

    @classmethod
    def create(
        cls, work_unit: WorkUnit, page_numbers: Optional[List[int]] = None
    ) -> "ExecutionContext":
        frames = work_unit.frames
        metadata = work_unit.metadata
        template = work_unit.template

        if page_numbers:
            frames = [frame for idx, frame in enumerate(frames) if idx in page_numbers]
            metadata = [
                meta for idx, meta in enumerate(metadata) if idx in page_numbers
            ]

        doc = MetaReader.from_data(frames=frames, ocr_meta=metadata)
        return ExecutionContext(
            doc_id=work_unit.doc_id, template=template, document=doc
        )
