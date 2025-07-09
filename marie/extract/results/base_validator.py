import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from marie.extract.structures import UnstructuredDocument
from marie.logging_core.predefined import default_logger as logger


class ValidationStage(Enum):
    """Defines the stage at which validation occurs"""

    PARSE_OUTPUT = (
        "parse_output"  # Validation after parsing output doc = UnstructuredDocument
    )
    CONVERTER_OUTPUT = "converter_output"  # Validation after converting UnstructuredDocument to MatchResult
    PRE_PROCESSING = "pre_processing"


@dataclass
class ValidationError:
    """Structured validation error"""

    code: str
    message: str
    field: Optional[str] = None
    severity: str = "ERROR"

    def __str__(self) -> str:
        field_part = f" in field '{self.field}'" if self.field else ""
        return f"[{self.severity}] {self.code}: {self.message}{field_part}"


@dataclass
class ValidationWarning:
    """Structured validation warning"""

    code: str
    message: str
    field: Optional[str] = None

    def __str__(self) -> str:
        field_part = f" in field '{self.field}'" if self.field else ""
        return f"[WARNING] {self.code}: {self.message}{field_part}"


@dataclass
class ValidationResult:
    """Structured validation result"""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validator_name: str = ""
    execution_time: float = 0.0

    def add_error(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        severity: str = "ERROR",
    ):
        """Add an error to the result"""
        self.errors.append(
            ValidationError(code=code, message=message, field=field, severity=severity)
        )
        self.valid = False

    def add_warning(self, code: str, message: str, field: Optional[str] = None):
        """Add a warning to the result"""
        self.warnings.append(ValidationWarning(code=code, message=message, field=field))


@dataclass
class ValidationContext:
    """Context information for validation"""

    stage: ValidationStage
    input_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    parser_name: Optional[str] = None
    working_dir: Optional[str] = None
    src_dir: Optional[str] = None
    conf: Optional[Any] = None

    @classmethod
    def create_parse_context(
        cls,
        doc: UnstructuredDocument,
        parser_name: str,
        working_dir: str = "",
        src_dir: str = "",
        conf: Any = None,
        **extra_metadata,
    ) -> 'ValidationContext':
        """Create context for parser output validation"""
        metadata = {
            'working_dir': working_dir,
            'src_dir': src_dir,
            'conf': conf,
            'parser_name': parser_name,
            **extra_metadata,
        }

        return cls(
            stage=ValidationStage.PARSE_OUTPUT,
            input_data=doc,
            metadata=metadata,
            parser_name=parser_name,
            working_dir=working_dir,
            src_dir=src_dir,
            conf=conf,
        )

    @classmethod
    def create_converter_context(
        cls,
        original_doc: UnstructuredDocument,
        converter_output: Any,
        working_dir: str = "",
        conf: Any = None,
        **extra_metadata,
    ) -> 'ValidationContext':
        """Create context for converter output validation"""
        metadata = {
            'original_doc': original_doc,
            'working_dir': working_dir,
            'conf': conf,
            **extra_metadata,
        }

        return cls(
            stage=ValidationStage.CONVERTER_OUTPUT,
            input_data=converter_output,
            metadata=metadata,
            working_dir=working_dir,
            conf=conf,
        )


@dataclass
class ValidationSummary:
    """Summary of all validation results"""

    overall_valid: bool
    total_errors: int
    total_warnings: int
    results: List[ValidationResult] = field(default_factory=list)
    execution_time: float = 0.0
    parser_name: Optional[str] = None

    def get_errors_by_severity(self, severity: str = "ERROR") -> List[ValidationError]:
        """Get errors by severity level"""
        errors = []
        for result in self.results:
            errors.extend([e for e in result.errors if e.severity == severity])
        return errors


class BaseValidator(ABC):
    """Base validator"""

    def __init__(self, name: str, supported_stages: Set[ValidationStage] = None):
        self.name = name
        self.supported_stages = supported_stages or {ValidationStage.PARSE_OUTPUT}

    @abstractmethod
    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        """Internal validation logic to be implemented by subclasses"""
        pass

    def validate(self, context: ValidationContext) -> ValidationResult:
        """Validation method with timing and error handling"""
        start_time = time.time()

        if not self.supports_stage(context.stage):
            result = ValidationResult(valid=False, validator_name=self.name)
            result.add_error(
                code="UNSUPPORTED_STAGE",
                message=f"Validator '{self.name}' does not support stage '{context.stage.value}'",
            )
            return result

        try:
            result = self._validate_internal(context)
            result.validator_name = self.name
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            result = ValidationResult(valid=False, validator_name=self.name)
            result.add_error(
                code="VALIDATION_EXCEPTION",
                message=f"Validation failed with exception: {str(e)}",
                severity="CRITICAL",
            )
            result.execution_time = time.time() - start_time
            logger.error(f"Validator '{self.name}' failed with exception: {e}")
            return result

    def supports_stage(self, stage: ValidationStage) -> bool:
        """Check if validator supports the given stage"""
        return stage in self.supported_stages


class FunctionValidatorWrapper(BaseValidator):
    """
    Wrapper that allows plain functions to be used as validators

    Functions must have the signature: func(context: ValidationContext) -> ValidationResult
    """

    def __init__(
        self,
        name: str,
        func: Callable[[ValidationContext], ValidationResult],
        supported_stages: Set[ValidationStage] = None,
    ):
        """
        Initialize function validator wrapper

        Args:
            name: Validator name
            func: Function that takes ValidationContext and returns ValidationResult
            supported_stages: Stages this validator supports
        """
        super().__init__(name, supported_stages)
        self.func = func
        logger.debug(f"Created FunctionValidatorWrapper '{name}'")

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        """Execute the wrapped function with ValidationContext"""
        logger.debug(f"Calling validator function: {self.func.__name__}")

        result = self.func(context)

        if not isinstance(result, ValidationResult):
            raise ValueError(
                f"Validator function '{self.func.__name__}' must return ValidationResult, got {type(result)}"
            )

        return result
