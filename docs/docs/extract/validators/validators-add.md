---
sidebar_position: 2
---

## Hello World Validator

This minimal example shows how to create, register, and use a simple validator. It demonstrates:
- Declaring supported stages
- Reading metadata from the validation context
- Returning a ValidationResult with errors/warnings and metadata
- Registering the validator so it’s available by name in configs

### Implementation

```python
# Python
from typing import Optional

from marie.extract.validator import (
    BaseValidator,
    ValidationResult,
    ValidationContext,
    ValidationStage,
)
from marie.extract.registry import register_validator


@register_validator("hello_world")
class HelloWorldValidator(BaseValidator):
    """
    A minimal validator that checks for a 'greeting' key in context.metadata.
    - If greeting == 'hello' (case-insensitive), it passes.
    - If greeting is present but not 'hello', it adds a warning.
    - If greeting is missing, it adds an error.
    """

    def __init__(self) -> None:
        # Supports both PRE_PROCESSING and PARSE_OUTPUT for demo purposes
        super().__init__(
            name="hello_world",
            supported_stages={ValidationStage.PRE_PROCESSING, ValidationStage.PARSE_OUTPUT},
        )

    def _validate_internal(self, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)

        greeting: Optional[str] = None
        if context.metadata and isinstance(context.metadata, dict):
            greeting = context.metadata.get("greeting")

        if greeting is None:
            result.valid = False
            result.add_error(
                code="MISSING_GREETING",
                message="Expected 'greeting' in context.metadata",
            )
        elif greeting.lower() != "hello":
            result.add_warning(
                code="NON_HELLO_GREETING",
                message=f"Received greeting='{greeting}', expected 'hello'",
            )

        # Attach useful diagnostics for debugging/observability
        result.metadata = {
            "received_greeting": greeting,
            "stage": context.stage.value,
        }
        return result
```


### Add to an annotator config

Attach your new validator by name under the annotator’s validators list:

```yaml
# YAML
annotators:
  claims:
    annotator_type: "llm"
    model_config:
      model_name: deepseek_r1_32
      prompt_path: "./claims.j2"
      expect_output: "json"
    parser_name: default
    validators:
      - "hello_world"         # our hello-world validator
      - "document_structure"  # any other validators you need
```


### What to expect

- If context.metadata contains greeting: "hello" (any case), the validator passes with no issues.
- If greeting exists but is not "hello", a warning is reported.
- If greeting is missing, the validator fails with an error.
- result.metadata includes the received greeting and the current stage for easy debugging.