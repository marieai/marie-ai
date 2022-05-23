from pydantic import BaseModel
from pydantic import BaseConfig, BaseModel, Field, create_model, root_validator

from collections import defaultdict
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Union, Any

DESCRIPTION_DATA = "Data to send, a list of dict/string/bytes that can be converted into a list of `Document` objects"
DESCRIPTION_TARGET_EXEC = "A regex string representing the specific pods/deployments targeted by the request."
DESCRIPTION_PARAMETERS = "A dictionary of parameters to be sent to the executor."
DESCRIPTION_EXEC_ENDPOINT = (
    "The endpoint string, by convention starts with `/`. "
    "All executors bind with `@requests(on=exec_endpoint)` will receive this request."
)


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


class JinaHealthModel(BaseModel):
    """Pydantic BaseModel for Jina health check, used as the response model in REST app."""

    ...


class JinaStatusModel(BaseModel):
    """Pydantic BaseModel for Jina status, used as the response model in REST app."""

    jina: Dict
    envs: Dict
    used_memory: str

    class Config:
        alias_generator = _to_camel_case
        allow_population_by_field_name = True


class JinaRequestModel(BaseModel):
    """
    Jina HTTP request model.
    """

    # the dict one is only for compatibility.
    # So we will accept data: {[Doc1.to_dict, Doc2...]} and data: {docs: [[Doc1.to_dict, Doc2...]}
    data: Optional[Any] = Field(
        None,
        example={},
        description=DESCRIPTION_DATA,
    )
    target_executor: Optional[str] = Field(
        None,
        example="",
        description=DESCRIPTION_TARGET_EXEC,
    )
    parameters: Optional[Dict] = Field(
        None,
        example={},
        description=DESCRIPTION_PARAMETERS,
    )

    class Config:
        alias_generator = _to_camel_case
        allow_population_by_field_name = True


PROTO_TO_PYDANTIC_MODELS = SimpleNamespace()


class JinaResponseModel(BaseModel):
    """
    Jina HTTP Response model. Only `request_id` and `data` are preserved.
    """

    header: PROTO_TO_PYDANTIC_MODELS.HeaderProto = None
    parameters: Dict = None
    routes: List[PROTO_TO_PYDANTIC_MODELS.RouteProto] = None
    data: Optional[Any] = None
    # data: Optional[PydanticDocumentArray] = None

    class Config:
        alias_generator = _to_camel_case
        allow_population_by_field_name = True


class JinaEndpointRequestModel(JinaRequestModel):
    """
    Jina HTTP request model that allows customized endpoint.
    """

    exec_endpoint: str = Field(
        ...,
        example="/foo",
        description="The endpoint string, by convention starts with `/`. "
        'All executors bind with `@requests(on="/foo")` will receive this request.',
    )
