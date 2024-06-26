import pydantic

major_version = int(pydantic.__version__.split('.')[0])

print(major_version)


def patch_pydantic_schema(cls):
    print(major_version)
    raise NotImplementedError


if major_version >= 2:
    from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
    from pydantic_core import PydanticOmit, core_schema

    class PydanticJsonSchema(GenerateJsonSchema):
        def handle_invalid_for_json_schema(
            self, schema: core_schema.CoreSchema, error_info: str
        ) -> JsonSchemaValue:
            if "core_schema.PlainValidatorFunctionSchema" in error_info:
                raise PydanticOmit
            return super().handle_invalid_for_json_schema(schema, error_info)

    def patch_pydantic_schema(cls):
        major_version = int(pydantic.__version__.split('.')[0])
        # Check if the major version is 2 or higher
        if major_version < 2:
            schema = cls.model_json_schema(mode="validation")
        else:
            schema = cls.model_json_schema(
                mode="validation", schema_generator=PydanticJsonSchema
            )
        return schema


patch_pydantic_schema_2x = patch_pydantic_schema
