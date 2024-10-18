from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


class JinaHealthModel(BaseModel):
    """Pydantic BaseModel for Jina health check, used as the response model in REST app."""

    ...


class JinaInfoModel(BaseModel):
    """Pydantic BaseModel for Jina status, used as the response model in REST app."""

    jina: Optional[Dict] = None
    marie: Optional[Dict] = None
    envs: Dict
    model_config = ConfigDict(alias_generator=_to_camel_case, populate_by_name=True)
