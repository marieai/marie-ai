"""
Schema-based extractor for Marie AI.

This extractor uses field definitions from ExtractorSchema to extract
structured data from documents. Supports three training modes:
- FOUNDATION: Zero-shot extraction using LLM with field definitions as prompt
- FINE_TUNE: Uses fine-tuned LayoutLM models
- CUSTOM: Uses custom-trained NER models
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.annotators.util import route_llm_engine
from marie.logging_core.logger import MarieLogger
from marie.utils.utils import ensure_exists

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

    from marie.extract.structures.unstructured_document import UnstructuredDocument


class TrainingMode(str, Enum):
    """Training mode for schema-based extraction."""

    FOUNDATION = "FOUNDATION"  # Zero-shot LLM extraction
    FINE_TUNE = "FINE_TUNE"  # Fine-tuned LayoutLM
    CUSTOM = "CUSTOM"  # Custom trained NER model


class ExtractorFieldDataType(str, Enum):
    """Data types for extractor fields."""

    PLAIN_TEXT = "PLAIN_TEXT"
    DATETIME = "DATETIME"
    NUMBER = "NUMBER"
    CURRENCY = "CURRENCY"
    ADDRESS = "ADDRESS"
    CHECKBOX = "CHECKBOX"


class ExtractorFieldOccurrence(str, Enum):
    """Occurrence patterns for extractor fields."""

    OPTIONAL_ONCE = "OPTIONAL_ONCE"
    OPTIONAL_MULTIPLE = "OPTIONAL_MULTIPLE"
    REQUIRED_ONCE = "REQUIRED_ONCE"
    REQUIRED_MULTIPLE = "REQUIRED_MULTIPLE"


class ExtractorFieldMethod(str, Enum):
    """Extraction methods for fields."""

    EXTRACT = "EXTRACT"  # Direct text extraction
    NORMALIZE = "NORMALIZE"  # Extract and normalize (dates, currency)
    CLASSIFY = "CLASSIFY"  # Classify from options
    DETECT = "DETECT"  # Detect presence (checkbox)


@dataclass
class SchemaField:
    """Represents a field in the extraction schema."""

    name: str
    display_name: str
    description: str
    data_type: ExtractorFieldDataType = ExtractorFieldDataType.PLAIN_TEXT
    occurrence: ExtractorFieldOccurrence = ExtractorFieldOccurrence.OPTIONAL_ONCE
    method: ExtractorFieldMethod = ExtractorFieldMethod.EXTRACT
    prompt_hint: Optional[str] = None
    color: Optional[str] = None
    is_enabled: bool = True
    children: List["SchemaField"] = field(default_factory=list)

    @classmethod
    def from_json_schema_property(
        cls, name: str, prop: Dict[str, Any]
    ) -> "SchemaField":
        """Create SchemaField from JSON Schema property with x-extractor extensions."""
        x_extractor = prop.get("x-extractor", {})

        # Map JSON Schema types to data types
        json_type = prop.get("type", "string")
        format_hint = prop.get("format", "")

        if json_type == "number" or json_type == "integer":
            data_type = ExtractorFieldDataType.NUMBER
        elif format_hint == "date-time" or format_hint == "date":
            data_type = ExtractorFieldDataType.DATETIME
        elif json_type == "boolean":
            data_type = ExtractorFieldDataType.CHECKBOX
        else:
            data_type_str = x_extractor.get("dataType", "PLAIN_TEXT")
            data_type = ExtractorFieldDataType(data_type_str)

        occurrence_str = x_extractor.get("occurrence", "OPTIONAL_ONCE")
        method_str = x_extractor.get("method", "EXTRACT")

        return cls(
            name=name,
            display_name=x_extractor.get("displayName", name.replace("_", " ").title()),
            description=prop.get("description", ""),
            data_type=data_type,
            occurrence=ExtractorFieldOccurrence(occurrence_str),
            method=ExtractorFieldMethod(method_str),
            prompt_hint=x_extractor.get("promptHint"),
            color=x_extractor.get("color"),
            is_enabled=x_extractor.get("isEnabled", True),
        )


class ExtractionResult(BaseModel):
    """Result of extracting a single field."""

    field_name: str
    value: Any
    confidence: float
    page_number: int
    bounding_box: Optional[Dict[str, float]] = None  # Normalized 0-1 coordinates
    raw_text: Optional[str] = None


class DocumentExtractionResult(BaseModel):
    """Complete extraction result for a document."""

    document_id: str
    fields: List[ExtractionResult]
    metadata: Dict[str, Any] = {}


class SchemaBasedExtractor(DocumentAnnotator):
    """
    Extract fields based on ExtractorSchema definition.

    Supports three extraction modes:
    - FOUNDATION: Uses LLM with field definitions as structured prompt
    - FINE_TUNE: Uses fine-tuned LayoutLM model
    - CUSTOM: Uses custom-trained NER model

    Configuration:
        schema_config: JSON Schema with x-extractor extensions for field definitions
        model_config: Model configuration including mode, model_name, etc.
    """

    def __init__(
        self,
        working_dir: str,
        schema_config: Dict[str, Any],
        model_config: Dict[str, Any],
        run_context: Optional["RunContext"] = None,
        **kwargs,
    ):
        """
        Initialize the schema-based extractor.

        Args:
            working_dir: Working directory for the document processing
            schema_config: JSON Schema with x-extractor extensions
            model_config: Configuration dict with:
                - mode: TrainingMode (FOUNDATION, FINE_TUNE, CUSTOM)
                - model_name: Name of the model to use
                - model_path: Path to fine-tuned/custom model (for FINE_TUNE/CUSTOM)
                - multimodal: Whether to use multimodal processing
                - confidence_threshold: Minimum confidence for extraction
        """
        super().__init__()
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        self.working_dir = working_dir
        self.run_context = run_context

        # Parse schema configuration
        self.schema_config = schema_config
        self.fields = self._parse_fields(schema_config)
        self.logger.info(f"Parsed {len(self.fields)} fields from schema")

        # Model configuration
        self.mode = TrainingMode(model_config.get("mode", "FOUNDATION"))
        self.model_name = model_config.get("model_name")
        self.model_path = model_config.get("model_path")
        self.multimodal = model_config.get("multimodal", True)
        self.confidence_threshold = model_config.get("confidence_threshold", 0.7)

        # Setup output directory
        self.output_dir = ensure_exists(os.path.join(working_dir, "extraction-output"))

        # Initialize extraction engine based on mode
        self._init_engine()

    def _parse_fields(self, schema_config: Dict[str, Any]) -> List[SchemaField]:
        """Parse fields from JSON Schema configuration."""
        fields = []
        properties = schema_config.get("properties", {})

        for name, prop in properties.items():
            schema_field = SchemaField.from_json_schema_property(name, prop)
            if schema_field.is_enabled:
                fields.append(schema_field)

                # Handle nested properties (for table-like structures)
                if prop.get("type") == "object" and "properties" in prop:
                    for child_name, child_prop in prop["properties"].items():
                        child_field = SchemaField.from_json_schema_property(
                            child_name, child_prop
                        )
                        if child_field.is_enabled:
                            schema_field.children.append(child_field)

        return fields

    def _init_engine(self) -> None:
        """Initialize the extraction engine based on training mode."""
        if self.mode == TrainingMode.FOUNDATION:
            self._init_foundation_engine()
        elif self.mode == TrainingMode.FINE_TUNE:
            self._init_fine_tune_engine()
        elif self.mode == TrainingMode.CUSTOM:
            self._init_custom_engine()

    def _init_foundation_engine(self) -> None:
        """Initialize LLM-based foundation extraction."""
        if not self.model_name:
            self.model_name = "gpt-4-vision"
        self.engine = route_llm_engine(self.model_name, self.multimodal)
        self.prompt_template = self._build_extraction_prompt()

    def _init_fine_tune_engine(self) -> None:
        """Initialize fine-tuned LayoutLM extraction."""
        if not self.model_path:
            raise ValueError("model_path required for FINE_TUNE mode")
        # Placeholder for LayoutLM initialization
        self.engine = None
        self.logger.info(f"Fine-tune mode: Loading model from {self.model_path}")

    def _init_custom_engine(self) -> None:
        """Initialize custom NER model extraction."""
        if not self.model_path:
            raise ValueError("model_path required for CUSTOM mode")
        # Placeholder for custom NER model initialization
        self.engine = None
        self.logger.info(f"Custom mode: Loading model from {self.model_path}")

    def _build_extraction_prompt(self) -> str:
        """Build extraction prompt from schema fields."""
        field_descriptions = []
        for f in self.fields:
            desc = f"- **{f.display_name}** ({f.name}): {f.description}"
            if f.prompt_hint:
                desc += f" Hint: {f.prompt_hint}"
            if f.occurrence in [
                ExtractorFieldOccurrence.REQUIRED_ONCE,
                ExtractorFieldOccurrence.REQUIRED_MULTIPLE,
            ]:
                desc += " [REQUIRED]"
            if f.occurrence in [
                ExtractorFieldOccurrence.OPTIONAL_MULTIPLE,
                ExtractorFieldOccurrence.REQUIRED_MULTIPLE,
            ]:
                desc += " [Can appear multiple times]"
            field_descriptions.append(desc)

        prompt = f"""Extract the following fields from this document. Return the results as JSON.

Fields to extract:
{chr(10).join(field_descriptions)}

For each field found, provide:
- field_name: The field identifier
- value: The extracted value
- confidence: Confidence score (0.0-1.0)
- page_number: Page where the value was found (0-indexed)
- bounding_box: Normalized coordinates {{x: 0-1, y: 0-1, width: 0-1, height: 0-1}}

Return a JSON object with a "fields" array containing the extraction results.
If a field is not found, omit it from the results.
"""
        return prompt

    @property
    def capabilities(self) -> List[AnnotatorCapabilities]:
        return [AnnotatorCapabilities.EXTRACTOR]

    def annotate(self, document: "UnstructuredDocument", frames: List) -> None:
        """
        Perform schema-based extraction on the document.

        Args:
            document: UnstructuredDocument to extract from
            frames: List of document page frames/images
        """
        self.logger.info(
            f"Extracting fields from document using {self.mode.value} mode"
        )

        if self.mode == TrainingMode.FOUNDATION:
            self._extract_foundation(document, frames)
        elif self.mode == TrainingMode.FINE_TUNE:
            self._extract_fine_tune(document, frames)
        elif self.mode == TrainingMode.CUSTOM:
            self._extract_custom(document, frames)

    async def aannotate(self, document: "UnstructuredDocument", frames: List) -> None:
        """Async version of annotate."""
        self.logger.info(
            f"Async extracting fields from document using {self.mode.value} mode"
        )

        if self.mode == TrainingMode.FOUNDATION:
            await self._aextract_foundation(document, frames)
        elif self.mode == TrainingMode.FINE_TUNE:
            await self._aextract_fine_tune(document, frames)
        elif self.mode == TrainingMode.CUSTOM:
            await self._aextract_custom(document, frames)

    def _extract_foundation(
        self, document: "UnstructuredDocument", frames: List
    ) -> None:
        """Foundation mode: Use LLM for zero-shot extraction."""
        from marie.extract.annotators.util import scan_and_process_images

        frames_dir = os.path.join(self.working_dir, "frames")

        scan_and_process_images(
            frames_dir,
            self.output_dir,
            self.prompt_template,
            document,
            engine=self.engine,
            is_multimodal=self.multimodal,
            expect_output="json",
        )

    async def _aextract_foundation(
        self, document: "UnstructuredDocument", frames: List
    ) -> None:
        """Async foundation mode extraction."""
        from marie.extract.annotators.util import ascan_and_process_images

        frames_dir = os.path.join(self.working_dir, "frames")

        await ascan_and_process_images(
            frames_dir,
            self.output_dir,
            self.prompt_template,
            document,
            engine=self.engine,
            is_multimodal=self.multimodal,
            expect_output="json",
        )

    def _extract_fine_tune(
        self, document: "UnstructuredDocument", frames: List
    ) -> None:
        """Fine-tune mode: Use fine-tuned LayoutLM model."""
        # TODO: Implement LayoutLM-based extraction
        raise NotImplementedError("Fine-tune extraction not yet implemented")

    async def _aextract_fine_tune(
        self, document: "UnstructuredDocument", frames: List
    ) -> None:
        """Async fine-tune mode extraction."""
        self._extract_fine_tune(document, frames)

    def _extract_custom(self, document: "UnstructuredDocument", frames: List) -> None:
        """Custom mode: Use custom-trained NER model."""
        # TODO: Implement custom NER model extraction
        raise NotImplementedError("Custom extraction not yet implemented")

    async def _aextract_custom(
        self, document: "UnstructuredDocument", frames: List
    ) -> None:
        """Async custom mode extraction."""
        self._extract_custom(document, frames)

    def parse_output(self, raw_output: str) -> DocumentExtractionResult:
        """
        Parse raw extraction output into structured results.

        Args:
            raw_output: Raw JSON output from extraction

        Returns:
            DocumentExtractionResult with parsed fields
        """
        try:
            data = json.loads(raw_output)
            fields_data = data.get("fields", [])

            results = []
            for field_data in fields_data:
                # Filter by confidence threshold
                confidence = field_data.get("confidence", 0.0)
                if confidence < self.confidence_threshold:
                    continue

                result = ExtractionResult(
                    field_name=field_data.get("field_name", ""),
                    value=field_data.get("value"),
                    confidence=confidence,
                    page_number=field_data.get("page_number", 0),
                    bounding_box=field_data.get("bounding_box"),
                    raw_text=field_data.get("raw_text"),
                )
                results.append(result)

            return DocumentExtractionResult(
                document_id=data.get("document_id", ""),
                fields=results,
                metadata=data.get("metadata", {}),
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extraction output: {e}")
            return DocumentExtractionResult(document_id="", fields=[])

    def get_field_by_name(self, name: str) -> Optional[SchemaField]:
        """Get a schema field by name."""
        for f in self.fields:
            if f.name == name:
                return f
            for child in f.children:
                if child.name == name:
                    return child
        return None

    def validate_extraction(
        self, result: DocumentExtractionResult
    ) -> Dict[str, List[str]]:
        """
        Validate extraction results against schema requirements.

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        extracted_fields = {r.field_name for r in result.fields}

        for schema_field in self.fields:
            if schema_field.occurrence in [
                ExtractorFieldOccurrence.REQUIRED_ONCE,
                ExtractorFieldOccurrence.REQUIRED_MULTIPLE,
            ]:
                if schema_field.name not in extracted_fields:
                    errors.append(
                        f"Required field '{schema_field.display_name}' not found"
                    )

            # Check multiplicity
            field_count = sum(
                1 for r in result.fields if r.field_name == schema_field.name
            )
            if (
                schema_field.occurrence
                in [
                    ExtractorFieldOccurrence.OPTIONAL_ONCE,
                    ExtractorFieldOccurrence.REQUIRED_ONCE,
                ]
                and field_count > 1
            ):
                warnings.append(
                    f"Field '{schema_field.display_name}' expected once but found {field_count} times"
                )

        return {"errors": errors, "warnings": warnings}
