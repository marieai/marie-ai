import time
from typing import Any, List

from marie.extract.registry import component_registry
from marie.extract.structures import UnstructuredDocument
from marie.extract.validator.base import (
    ValidationContext,
    ValidationResult,
    ValidationSummary,
)
from marie.logging_core.predefined import default_logger as logger


def validate_document(
    context: ValidationContext, validator_names: List[str] = None
) -> ValidationSummary:
    """
    Validate a document using specified validators or all available validators
    """
    start_time = time.time()

    # If no validators specified, use all available validators that support the stage
    if validator_names is None:
        validator_names = [
            name
            for name, validator in component_registry.validators().items()
            if validator.supports_stage(context.stage)
        ]

    results = []
    for validator_name in validator_names:
        validator = component_registry.get_validator(validator_name)

        if validator:
            if validator.supports_stage(context.stage):
                logger.debug(f"Running validator: {validator_name}")
                result = validator.validate(context)
                results.append(result)
            else:
                logger.debug(
                    f"Skipping validator '{validator_name}' - doesn't support stage '{context.stage.value}'"
                )
        else:
            logger.warning(f"Validator '{validator_name}' not found")
            result = ValidationResult(valid=False, validator_name=validator_name)
            result.add_error(
                "VALIDATOR_NOT_FOUND", f"Validator '{validator_name}' not found"
            )
            results.append(result)

    total_errors = sum(len(result.errors) for result in results)
    total_warnings = sum(len(result.warnings) for result in results)
    overall_valid = total_errors == 0

    summary = ValidationSummary(
        overall_valid=overall_valid,
        total_errors=total_errors,
        total_warnings=total_warnings,
        results=results,
        execution_time=time.time() - start_time,
        parser_name=context.parser_name,
    )

    if overall_valid:
        logger.info(f"Validation passed ({total_warnings} warnings)")
    else:
        logger.error(
            f"Validation failed ({total_errors} errors, {total_warnings} warnings)"
        )

    return summary


def validate_parser_output(
    doc: UnstructuredDocument,
    parser_name: str,
    working_dir: str = "",
    src_dir: str = "",
    conf: Any = None,
    validator_names: List[str] = None,
    **extra_metadata,
) -> ValidationSummary:
    """
    Validate parser output using specified validators
    """
    context = ValidationContext.create_parse_context(
        doc=doc,
        parser_name=parser_name,
        working_dir=working_dir,
        src_dir=src_dir,
        conf=conf,
        **extra_metadata,
    )

    return validate_document(context, validator_names)
