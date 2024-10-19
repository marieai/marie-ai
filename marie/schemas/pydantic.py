from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticOmit, core_schema

from marie.logging_core.predefined import default_logger as logger


class GenerateJsonSchemaBypassingValidation(GenerateJsonSchema):
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        logger.error(f'Error in schema: {error_info}')
        raise PydanticOmit
