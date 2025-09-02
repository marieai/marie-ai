import logging
import os

from omegaconf import OmegaConf

from marie.extract.registry import register_validator
from marie.extract.structures import UnstructuredDocument
from marie.extract.validator.base import (
    BaseValidator,
    ValidationContext,
    ValidationResult,
    ValidationStage,
)


@register_validator("noop")
def validate_noop(context: ValidationContext) -> ValidationResult:
    """
    No-op validator for annotations. This is a placeholder and does not perform any action.
    """
    logging.info("No-op validator for annotations called. No action taken.")
    # This function is intentionally left empty to serve as a placeholder.
    pass


class AnnotationsValidator(BaseValidator):
    """Validator for annotations parser output"""

    def __init__(self):
        super().__init__(
            name="annotations", supported_stages={ValidationStage.PARSE_OUTPUT}
        )

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        doc = context.input_data

        if not isinstance(doc, UnstructuredDocument):
            result.add_error(
                "INVALID_TYPE",
                f"Expected UnstructuredDocument, got {type(doc).__name__}",
            )
            return result

        # Count annotations
        total_annotations = 0
        annotation_types = set()
        pages_with_annotations = set()
        lines_with_annotations = 0

        for line in doc.lines:
            if line.annotations:
                lines_with_annotations += 1
                total_annotations += len(line.annotations)
                pages_with_annotations.add(line.page_id)
                for ann in line.annotations:
                    if hasattr(ann, 'annotation_type'):
                        annotation_types.add(ann.annotation_type)

        # Validation checks
        if total_annotations == 0:
            src_dir = context.src_dir or context.metadata.get('src_dir', '')
            annotations_file = os.path.join(src_dir, "annotations.json")

            if os.path.exists(annotations_file):
                result.add_error(
                    "ANNOTATIONS_NOT_PROCESSED",
                    "annotations.json exists but no annotations were found in document",
                )
            else:
                result.add_warning(
                    "NO_ANNOTATIONS_FILE",
                    "No annotations.json file found - this may be expected",
                )
        elif total_annotations < 3:
            result.add_warning(
                "LOW_ANNOTATION_COUNT",
                f"Only {total_annotations} annotations found, this might indicate parsing issues",
            )

        result.metadata = {
            'total_annotations': total_annotations,
            'annotation_types': list(annotation_types),
            'pages_with_annotations': len(pages_with_annotations),
            'lines_with_annotations': lines_with_annotations,
            'total_lines': len(doc.lines),
        }

        return result


class KeyValuesValidator(BaseValidator):
    """Validator for key-values parser output"""

    def __init__(self):
        super().__init__(
            name="key-values", supported_stages={ValidationStage.PARSE_OUTPUT}
        )

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        doc = context.input_data

        if not isinstance(doc, UnstructuredDocument):
            result.add_error(
                "INVALID_TYPE",
                f"Expected UnstructuredDocument, got {type(doc).__name__}",
            )
            return result

        # Count KV annotations
        kv_annotations = 0
        kv_types = set()

        for line in doc.lines:
            if line.annotations:
                for ann in line.annotations:
                    if hasattr(ann, 'annotation_type') and ann.annotation_type == "KV":
                        kv_annotations += 1
                        if hasattr(ann, 'label'):
                            kv_types.add(ann.label)

        if kv_annotations == 0:
            result.add_warning("NO_KV_ANNOTATIONS", "No KV annotations found")
        elif kv_annotations < 2:
            result.add_warning(
                "LOW_KV_COUNT", f"Only {kv_annotations} KV annotations found"
            )

        result.metadata = {'kv_annotations': kv_annotations, 'kv_types': list(kv_types)}

        return result


class TablesValidator(BaseValidator):
    """Validator for tables parser output"""

    def __init__(self):
        super().__init__(name="tables", supported_stages={ValidationStage.PARSE_OUTPUT})

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        doc = context.input_data

        if not isinstance(doc, UnstructuredDocument):
            result.add_error(
                "INVALID_TYPE",
                f"Expected UnstructuredDocument, got {type(doc).__name__}",
            )
            return result

        # Count table annotations
        table_annotations = 0
        for line in doc.lines:
            if line.annotations:
                for ann in line.annotations:
                    if (
                        hasattr(ann, 'annotation_type')
                        and ann.annotation_type == "TABLE"
                    ):
                        table_annotations += 1

        if table_annotations == 0:
            result.add_warning("NO_TABLE_ANNOTATIONS", "No TABLE annotations found")

        result.metadata = {'table_annotations': table_annotations}
        return result


class DocumentStructureValidator(BaseValidator):
    """General document structure validator"""

    def __init__(self):
        super().__init__(
            name="document_structure", supported_stages={ValidationStage.PARSE_OUTPUT}
        )

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        doc = context.input_data

        if not isinstance(doc, UnstructuredDocument):
            result.add_error(
                "INVALID_TYPE",
                f"Expected UnstructuredDocument, got {type(doc).__name__}",
            )
            return result

        # Basic structure validation
        if not doc.lines:
            result.add_error("NO_LINES", "Document has no lines")
            return result

        # Check for reasonable document length
        total_text = sum(len(line.text) for line in doc.lines)
        if total_text < 50:
            result.add_warning(
                "VERY_SHORT_DOCUMENT",
                f"Document is very short ({total_text} characters)",
            )

        # Check empty lines ratio
        empty_lines = sum(1 for line in doc.lines if not line.text.strip())
        if doc.lines:
            empty_ratio = empty_lines / len(doc.lines)
            if empty_ratio > 0.7:
                result.add_warning(
                    "HIGH_EMPTY_LINES", f"Document has {empty_ratio:.1%} empty lines"
                )

        result.metadata = {
            'line_count': len(doc.lines),
            'total_text_length': total_text,
            'empty_lines': empty_lines,
            'has_metadata': bool(doc.metadata),
        }

        return result


class ConverterOutputValidator(BaseValidator):
    """Validator for converter output (SubzeroResult/MatchSection)"""

    def __init__(self):
        super().__init__(
            name="converter_output", supported_stages={ValidationStage.CONVERTER_OUTPUT}
        )

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        from marie.extract.models.match import MatchSection

        result = ValidationResult(valid=True, validator_name=self.name)
        original_doc = context.metadata.get('original_doc')

        if context.input_data is None:
            result.add_error("NULL_CONVERTER_OUTPUT", "Converter output is None")
            return result

        converter_output: MatchSection = context.input_data

        if converter_output.sections is None:
            result.add_error(
                "INVALID_OUTPUT_STRUCTURE", "Converter output missing 'sections' field"
            )
            return result

        sections = converter_output.sections
        if not sections:
            result.add_warning("NO_SECTIONS", "No sections found in converter output")

        # Count total fields across all sections
        total_fields = 0
        total_field_rows = 0
        sections_with_fields = 0
        empty_sections = 0

        def count_section_fields(section: MatchSection):
            nonlocal total_fields, total_field_rows, sections_with_fields, empty_sections

            # Count non-repeating fields
            if section.matched_non_repeating_fields:
                field_count = len(section.matched_non_repeating_fields)
                total_fields += field_count
                sections_with_fields += 1

            # Count field rows
            if section.matched_field_rows:
                row_count = len(section.matched_field_rows)
                total_field_rows += row_count
                sections_with_fields += 1

            if (
                not section.matched_non_repeating_fields
                and not section.matched_field_rows
            ):
                empty_sections += 1

            if section.sections:
                for subsection in section.sections:
                    count_section_fields(subsection)

        for section in sections:
            count_section_fields(section)

        if total_fields == 0 and total_field_rows == 0:
            result.add_error(
                "NO_EXTRACTED_FIELDS", "No fields extracted in any section"
            )
        elif total_fields + total_field_rows < 3:
            result.add_warning(
                "LOW_FIELD_COUNT",
                f"Only {total_fields + total_field_rows} total fields extracted",
            )

        if empty_sections > len(sections) / 2:
            result.add_warning(
                "HIGH_EMPTY_SECTIONS",
                f"{empty_sections}/{len(sections)} sections are empty",
            )

        result.metadata = {
            'total_sections': len(sections),
            'sections_with_fields': sections_with_fields,
            'empty_sections': empty_sections,
            'total_non_repeating_fields': total_fields,
            'total_field_rows': total_field_rows,
            'output_type': type(converter_output).__name__,
        }

        return result
