from collections import defaultdict
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from google.protobuf.descriptor import Descriptor, FieldDescriptor
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from marie._docarray import docarray_v2
from marie.proto.jina_pb2 import (
    DataRequestProto,
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


def _get_oneof_validator(oneof_fields: List[str], oneof_key: str) -> Callable:
    """
    Generate a callable one-of validator for specified fields.

    This function creates a Pydantic model validator that ensures only one field
    from a specified list of fields (`oneof_fields`) is set for the given
    `oneof_key`. If multiple fields from the specified group are present in a
    dict-like input, a `ValueError` is raised.

    Parameters:
        oneof_fields: List[str] - A list of field names that belong to the one-of group.
        oneof_key: A unique identifier used to name the validator function.

    Returns:
        A callable Pydantic `before` model validator that validates the one-of group.

    Raises:
        ValueError: If more than one field in the `oneof_fields` list is present.
    """

    @model_validator(mode='before')
    def oneof_validator(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        # Check intersection of keys present in values vs the oneof group
        present_fields = set(oneof_fields).intersection(set(values.keys()))
        if len(present_fields) > 1:
            raise ValueError(
                f'only one field among {oneof_fields} can be set for key {oneof_key}!'
            )
        return values

    oneof_validator.__qualname__ = 'validate_' + oneof_key
    return oneof_validator


def _get_oneof_setter(oneof_fields: List[str], oneof_key: str) -> Callable:
    """
    Creates a setter function for managing mutually exclusive fields in a model (Oneof fields).

    This function generates a "Oneof" setter using a Pydantic validator. The setter ensures
    that only one field in a mutually exclusive set (Oneof fields) is set at any time in a
    Pydantic model instance. If a field in the given set is found with its default value,
    it will be unset (removed) from the instance, mimicking protocol buffers' Oneof
    behavior.

    Parameters:
        oneof_fields: List[str] of mutually exclusive field names to be managed by the setter.
        oneof_key: A unique key representing the Oneof field group.

    Returns:
        A callable function that serves as a validator for the specified Oneof fields.
    """

    @model_validator(mode='after')
    def oneof_setter(self):
        # In an 'after' validator, 'self' is the model instance
        for oneof_field in oneof_fields:
            if hasattr(self, oneof_field):
                val = getattr(self, oneof_field)
                # Check against the field's default value
                field_info = self.model_fields.get(oneof_field)
                if field_info and val == field_info.default:
                    # We want to unset it if it matches default (Proto behavior) instead of "pop"
                    delattr(self, oneof_field)
        return self

    oneof_setter.__qualname__ = 'set_' + oneof_key
    return oneof_setter


def protobuf_to_pydantic_model(
    protobuf_model: Union[Descriptor, 'GeneratedProtocolMessageType']
) -> BaseModel:
    """
    Converts a Protocol Buffer model to a Pydantic model.

    This function takes a Protocol Buffer model (or its descriptor) and generates a
    corresponding Pydantic model. The generated model includes fields and their
    respective types as defined in the input Protocol Buffer model. It also handles
    special Protocol Buffer field types such as enums, oneof, and nested message types.

    ..note:: Model gets assigned in the global Namespace :data:PROTO_TO_PYDANTIC_MODELS

    Parameters:
        protobuf_model: The Protocol Buffer model or descriptor to be converted to a Pydantic model.
            If a descriptor, it must be an instance of Descriptor. If a model, it must define the
            `DESCRIPTOR` attribute.

    Returns:
        The generated Pydantic model corresponding to the input Protocol Buffer model.

    Raises:
        ValueError: If the provided protobuf_model is not a valid Protocol Buffer model or lacks a
            `DESCRIPTOR` attribute.
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

        # Construct Field with alias and default
        field_args = {'alias': f.camelcase_name}
        if default_factory:
            field_args['default_factory'] = default_factory
        else:
            field_args['default'] = default_value

        all_fields[field_name] = (field_type, Field(**field_args))

    # Post-processing (Handle oneof fields)
    for oneof_k, oneof_v_list in oneof_fields.items():
        # Add generated validators to the dict (passed to create_model)
        oneof_field_validators[f'oneof_validator_{oneof_k}'] = _get_oneof_validator(
            oneof_fields=oneof_v_list, oneof_key=oneof_k
        )
        # Note: Setters that modify the instance after creation are trickier with
        # immutable models, but assuming standard models, this works.
        oneof_field_validators[f'oneof_setter_{oneof_k}'] = _get_oneof_setter(
            oneof_fields=oneof_v_list, oneof_key=oneof_k
        )

    if model_name == 'DocumentProto':
        from docarray.document.pydantic_model import PydanticDocument

        model = PydanticDocument
    elif model_name == 'DocumentArrayProto':
        from docarray.document.pydantic_model import PydanticDocumentArray

        model = PydanticDocumentArray
    else:
        model = create_model(
            model_name,
            **all_fields,
            __config__=ConfigDict(use_enum_values=True, populate_by_name=True),
            __validators__=oneof_field_validators,
        )
        model.model_rebuild()
    PROTO_TO_PYDANTIC_MODELS.__setattr__(model_name, model)

    return model


if not docarray_v2:
    for proto in (RouteProto, StatusProto, DataRequestProto, JinaInfoProto):
        protobuf_to_pydantic_model(proto)
else:
    for proto in (RouteProto, StatusProto, JinaInfoProto):
        protobuf_to_pydantic_model(proto)


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])


if not docarray_v2:
    from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray

    class JinaRequestModel(BaseModel):
        """
        Jina HTTP request model.
        """

        # the dict one is only for compatibility.
        # So we will accept data: {[Doc1.to_dict, Doc2...]} and data: {docs: [[Doc1.to_dict, Doc2...]}
        data: Optional[
            Union[
                PydanticDocumentArray,
                Dict[str, PydanticDocumentArray],
            ]
        ] = Field(
            None,
            examples=[
                [
                    {'text': 'hello, world!'},
                    {'uri': 'https://docs.marie.ai/_static/logo-light.svg'},
                ]
            ],
            description=DESCRIPTION_DATA,
        )
        target_executor: Optional[str] = Field(
            None,
            examples=[''],
            description=DESCRIPTION_TARGET_EXEC,
        )
        parameters: Optional[Dict] = Field(
            None,
            examples=[{}],
            description=DESCRIPTION_PARAMETERS,
        )
        model_config = ConfigDict(alias_generator=_to_camel_case, populate_by_name=True)

    class JinaResponseModel(BaseModel):
        """
        Jina HTTP Response model. Only `request_id` and `data` are preserved.
        """

        header: PROTO_TO_PYDANTIC_MODELS.HeaderProto = None
        parameters: Dict = None
        routes: List[PROTO_TO_PYDANTIC_MODELS.RouteProto] = None
        data: Optional[PydanticDocumentArray] = None
        model_config = ConfigDict(alias_generator=_to_camel_case, populate_by_name=True)

    class JinaEndpointRequestModel(JinaRequestModel):
        """
        Jina HTTP request model that allows customized endpoint.
        """

        exec_endpoint: str = Field(
            default='/',
            examples=['/'],
            description='The endpoint string, by convention starts with `/`. '
            'If you specify it as `/foo`, then all executors bind with `@requests(on="/foo")` will receive the request.',
        )
