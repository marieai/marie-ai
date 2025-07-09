from .base_validator import ValidationContext, ValidationResult
from .registry import register_validator


# 1. Using the register_validator decorator
@register_validator("doc_length")
def validate_document_length(context: ValidationContext) -> ValidationResult:
    result = ValidationResult(valid=True, validator_name="doc_length")
    doc = context.input_data

    if len(doc.lines) < 10:
        result.add_warning("SHORT_DOC", f"Document has only {len(doc.lines)} lines")

    return result


# more complex validation
@register_validator("table_structure")
def validate_table_structure(context: ValidationContext) -> ValidationResult:
    result = ValidationResult(valid=True, validator_name="table_structure")
    doc = context.input_data
    parser_name = context.parser_name

    # Only validate if this is from the tables parser
    if parser_name != "tables":
        result.add_warning(
            "SKIP_VALIDATION", f"Skipping table validation for parser '{parser_name}'"
        )
        return result

    # Check for table annotations
    table_count = 0
    for line in doc.lines:
        if line.annotations:
            for ann in line.annotations:
                if hasattr(ann, 'annotation_type') and ann.annotation_type == "TABLE":
                    table_count += 1

    if table_count == 0:
        result.add_error("NO_TABLES", "No table annotations found")

    result.metadata = {'table_count': table_count}
    return result
