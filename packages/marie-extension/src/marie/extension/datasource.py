from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from jsonschema import ValidationError as SchemaValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from marie.extension.settings import ExtensionModel, ImplementationRef, SchemaTrack
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable


class DatasourceDefinition(ExtensionModel):
    ref: str
    name: str
    display_name: str | dict[str, str] | None = Field(default=None, alias="displayName")
    description: str | dict[str, str] | None = None
    implementation: ImplementationRef | None = None
    provider_ref: str | None = Field(default=None, alias='providerRef')
    configuration_schema: SchemaTrack = Field(
        default_factory=SchemaTrack, alias="configurationSchema"
    )
    output_schema: dict[str, object] | None = Field(default=None, alias="outputSchema")


class DatasourceRequest(BaseModel):
    """One bounded page request for a saved datasource binding."""

    model_config = ConfigDict(extra='forbid')
    type: Literal['datasource']
    action: Literal['fetch_page']
    datasource: str
    source_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    configuration: dict[str, Any]
    credentials: dict[str, str] = Field(default_factory=dict, repr=False)
    user_id: str = ''
    cursor: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=100, strict=True)


class DatasourcePage(BaseModel):
    """Datasource-defined records and an optional pagination continuation."""

    model_config = ConfigDict(extra='forbid')
    records: list[Any] = Field(max_length=100)
    next_cursor: str | None = Field(default=None, min_length=1, max_length=4096)


class DatasourceInputError(ValueError):
    """Invalid datasource action, scope or dataset inputs."""


class DatasourceOutputError(ValueError):
    """Handler output exceeds its declared schema or page bounds."""


def invoke_datasource(
    definition: DatasourceDefinition,
    payload: dict[str, Any],
    fetch: Callable[[DatasourceRequest], DatasourcePage | dict[str, Any]],
) -> DatasourcePage:
    """Validate a declared datasource invocation and its typed page output."""
    try:
        request = DatasourceRequest.model_validate(payload)
        if request.datasource != definition.ref:
            raise DatasourceInputError('Datasource action is not declared')
        request.configuration = _validate_configuration(
            definition, request.configuration
        )
    except (ValidationError, SchemaValidationError) as error:
        raise DatasourceInputError('Invalid datasource request') from error

    try:
        if definition.output_schema is None:
            raise DatasourceOutputError('Datasource requires a declared outputSchema')
        validator_type = validator_for(definition.output_schema)
        validator_type.check_schema(definition.output_schema)
        validator = validator_type(definition.output_schema, registry=Registry())
    except SchemaError as error:
        raise DatasourceOutputError('Invalid datasource output declaration') from error

    output = fetch(request)
    try:
        page = DatasourcePage.model_validate(output)
        if len(page.records) > request.limit:
            raise DatasourceOutputError('Datasource page exceeds the requested limit')
        for record in page.records:
            validator.validate(record)
        encoded = json.dumps(page.model_dump(), ensure_ascii=False, allow_nan=False)
        if len(encoded.encode('utf-8')) > 1024 * 1024:
            raise DatasourceOutputError(
                'Datasource page exceeds 1 MiB; use artifact references'
            )
        return page
    except (
        ValidationError,
        SchemaValidationError,
        SchemaError,
        Unresolvable,
        TypeError,
        ValueError,
    ) as error:
        raise DatasourceOutputError('Invalid or unbounded datasource output') from error


def _validate_configuration(
    definition: DatasourceDefinition, values: dict[str, Any]
) -> dict[str, Any]:
    types = {
        'string': 'string',
        'text': 'string',
        'content': 'string',
        'paragraph': 'string',
        'select': 'string',
        'enum': 'string',
        'boolean': 'boolean',
        'bool': 'boolean',
        'int': 'integer',
        'integer': 'integer',
        'float': 'number',
        'number': 'number',
        'object': 'object',
        'array': 'array',
    }
    properties: dict[str, Any] = {}
    required = set(definition.configuration_schema.required)
    configuration = dict(values)
    for parameter in definition.configuration_schema.parameters:
        if parameter.type not in types:
            raise DatasourceInputError(
                'Unsupported datasource configuration declaration'
            )
        field: dict[str, Any] = {'type': types[parameter.type]}
        if parameter.options:
            field['enum'] = [
                value if isinstance(value, str) else value.value
                for value in parameter.options
            ]
        if parameter.minimum is not None:
            field['minimum'] = parameter.minimum
        if parameter.maximum is not None:
            field['maximum'] = parameter.maximum
        if parameter.required:
            required.add(parameter.key)
        if parameter.default is not None and parameter.key not in configuration:
            configuration[parameter.key] = parameter.default
        properties[parameter.key] = field
    if not required.issubset(properties):
        raise DatasourceInputError('Required datasource input is not declared')
    validator_type = validator_for({'type': 'object'})
    validator_type(
        {
            'type': 'object',
            'properties': properties,
            'required': sorted(required),
            'additionalProperties': False,
        }
    ).validate(configuration)
    if any(
        isinstance(configuration[key], str) and not configuration[key].strip()
        for key in required
    ):
        raise DatasourceInputError('Required datasource input is empty')
    return configuration
