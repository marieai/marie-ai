from typing import Callable, Protocol, Type, TypeVar, runtime_checkable

from omegaconf import OmegaConf

from marie.extract.models.definition import Template

TemplateBuilderFn = Callable[[OmegaConf], Template]


@runtime_checkable
class TemplateBuilderProto(Protocol):
    def build_template(self, config: OmegaConf) -> Template: ...


TBuilder = TypeVar(
    "TBuilder",
    TemplateBuilderFn,
    TemplateBuilderProto,
    Type[TemplateBuilderProto],
)
