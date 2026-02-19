from collections import defaultdict
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from google.protobuf.descriptor import Descriptor, FieldDescriptor
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from marie.proto.jina_pb2 import (
    JinaInfoProto,
    RouteProto,
    StatusProto,
)

if TYPE_CHECKING:  # pragma: no cover
    from google.protobuf.pyext.cpp_message import GeneratedProtocolMessageType

PROTO_TO_PYDANTIC_MODELS = SimpleNamespace()
PROTOBUF_TO_PYTHON_TYPE = {
    FieldDescriptor.TYPE_INT32: int,
    FieldDescriptor.TYPE_INT64: int,
    FieldDescriptor.TYPE_UINT32: int,
    FieldDescriptor.TYPE_UINT64: int,
    FieldDescriptor.TYPE_SINT32: int,
    FieldDescriptor.TYPE_SINT64: int,
    FieldDescriptor.TYPE_BOOL: bool,
    FieldDescriptor.TYPE_FLOAT: float,
    FieldDescriptor.TYPE_DOUBLE: float,
    FieldDescriptor.TYPE_FIXED32: float,
    FieldDescriptor.TYPE_FIXED64: float,
    FieldDescriptor.TYPE_SFIXED32: float,
    FieldDescriptor.TYPE_SFIXED64: float,
    FieldDescriptor.TYPE_BYTES: bytes,
    FieldDescriptor.TYPE_STRING: str,
    FieldDescriptor.TYPE_ENUM: Enum,
    FieldDescriptor.TYPE_MESSAGE: None,
}

DESCRIPTION_DATA = 'Data to send, a list of dict/string/bytes that can be converted into a list of `Document` objects'
DESCRIPTION_TARGET_EXEC = (
    'A regex string representing the specific pods/deployments targeted by the request.'
)
DESCRIPTION_PARAMETERS = 'A dictionary of parameters to be sent to the executor.'
DESCRIPTION_EXEC_ENDPOINT = (
    'The endpoint string, by convention starts with `/`. '
    'All executors bind with `@requests(on=exec_endpoint)` will receive this request.'
)


# Pydantic V2 model config for enum handling and allowing field population by name
CUSTOM_MODEL_CONFIG = ConfigDict(use_enum_values=True, populate_by_name=True)


def _get_oneof_validator(oneof_fields: List, oneof_key: str) -> Callable:
    """
    Pydantic model validator (before) classmethod generator to confirm only one oneof field is passed

    :param oneof_fields: list of field names for oneof
    :type oneof_fields: List
    :param oneof_key: oneof key
    :type oneof_key: str
    :return: classmethod for validating oneof fields
    """

    @model_validator(mode='before')
    @classmethod
    def oneof_validator(cls, values):
        if (
            isinstance(values, dict)
            and len(set(oneof_fields).intersection(set(values))) > 1
        ):
            raise ValueError(
                f'only one field among {oneof_fields} can be set for key {oneof_key}!'
            )
        return values

    oneof_validator.__qualname__ = 'validate_' + oneof_key
    return oneof_validator


def _get_oneof_setter(oneof_fields: List, oneof_key: str) -> Callable:
    """
    Pydantic model validator (after) classmethod generator to set the oneof key

    :param oneof_fields: list of field names for oneof
    :type oneof_fields: List
    :param oneof_key: oneof key
    :type oneof_key: str
    :return: classmethod for setting oneof fields in Pydantic models
    """

    @model_validator(mode='before')
    @classmethod
    def oneof_setter(cls, values):
        if isinstance(values, dict):
            for oneof_field in oneof_fields:
                if (
                    oneof_field in values
                    and oneof_field in cls.model_fields
                    and values[oneof_field] == cls.model_fields[oneof_field].default
                ):
                    values.pop(oneof_field)
        return values

    oneof_setter.__qualname__ = 'set_' + oneof_key
    return oneof_setter


def protobuf_to_pydantic_model(
    protobuf_model: Union[Descriptor, 'GeneratedProtocolMessageType'],
) -> BaseModel:
    """
    Converts Protobuf messages to Pydantic model for jsonschema creation/validattion

    ..note:: Model gets assigned in the global Namespace :data:PROTO_TO_PYDANTIC_MODELS

    :param protobuf_model: message from marie.proto file
    :type protobuf_model: Union[Descriptor, GeneratedProtocolMessageType]
    :return: Pydantic model
    """

    all_fields = {}
    oneof_fields = defaultdict(list)
    oneof_field_validators = {}

    desc = (
        protobuf_model
        if isinstance(protobuf_model, Descriptor)
        else getattr(protobuf_model, 'DESCRIPTOR', None)
    )
    if desc:
        model_name = desc.name
        protobuf_fields = desc.fields
    else:
        raise ValueError(
            f'protobuf_model is of type {type(protobuf_model)} and has no attribute "DESCRIPTOR"'
        )

    if model_name in vars(PROTO_TO_PYDANTIC_MODELS):
        return PROTO_TO_PYDANTIC_MODELS.__getattribute__(model_name)

    for f in protobuf_fields:
        field_name = f.name
        camel_case_alias = f.camelcase_name

        field_type = PROTOBUF_TO_PYTHON_TYPE[f.type]
        default_value = f.default_value
        default_factory = None

        if f.containing_oneof:
            # Proto Field type: oneof
            # NOTE: oneof fields are handled as a post-processing step
            oneof_fields[f.containing_oneof.name].append(field_name)

        if field_type is Enum:
            # Proto Field Type: enum
            enum_dict = {}

            for enum_field in f.enum_type.values:
                enum_dict[enum_field.name] = enum_field.number

            field_type = Enum(f.enum_type.name, enum_dict)

        if f.message_type:
            if f.message_type.name == 'Struct':
                # Proto Field Type: google.protobuf.Struct
                field_type = Dict
                default_factory = dict
            elif f.message_type.name == 'Timestamp':
                # Proto Field Type: google.protobuf.Timestamp
                field_type = datetime
                default_factory = datetime.now
            else:
                # Proto field type: Proto message defined in marie.proto
                if f.message_type.name == model_name:
                    # Self-referencing models
                    field_type = model_name
                else:
                    # This field_type itself a Pydantic model
                    field_type = protobuf_to_pydantic_model(f.message_type)
                    PROTO_TO_PYDANTIC_MODELS.model_name = field_type

        if f.label == FieldDescriptor.LABEL_REPEATED:
            field_type = List[field_type]

        all_fields[field_name] = (
            field_type,
            (
                Field(default_factory=default_factory, alias=camel_case_alias)
                if default_factory
                else Field(default=default_value, alias=camel_case_alias)
            ),
        )

    # Post-processing (Handle oneof fields)
    for oneof_k, oneof_v_list in oneof_fields.items():
        oneof_field_validators[f'oneof_validator_{oneof_k}'] = _get_oneof_validator(
            oneof_fields=oneof_v_list, oneof_key=oneof_k
        )
        oneof_field_validators[f'oneof_setter_{oneof_k}'] = _get_oneof_setter(
            oneof_fields=oneof_v_list, oneof_key=oneof_k
        )

    model = create_model(
        model_name,
        __config__=CUSTOM_MODEL_CONFIG,
        __validators__=oneof_field_validators,
        **all_fields,
    )
    model.model_rebuild()
    PROTO_TO_PYDANTIC_MODELS.__setattr__(model_name, model)

    return model


for proto in (RouteProto, StatusProto, JinaInfoProto):
    protobuf_to_pydantic_model(proto)


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])
