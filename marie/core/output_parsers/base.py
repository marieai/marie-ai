"""Base output parser class."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from marie.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from marie.core.bridge.pydantic import Field, ConfigDict
from marie.core.types import BaseOutputParser


@dataclass
class StructuredOutput:
    """Structured output class."""

    raw_output: str
    parsed_output: Optional[Any] = None


class OutputParserException(Exception):
    pass


class ChainableOutputParser(BaseOutputParser, ChainableMixin):
    """Chainable output parser."""

    # TODO: consolidate with base at some point if possible.

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Get query component."""
        return OutputParserComponent(output_parser=self)


class OutputParserComponent(QueryComponent):
    """Output parser component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output_parser: BaseOutputParser = Field(..., description="Output parser.")

    def _run_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        output = self.output_parser.parse(kwargs["input"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run component."""
        # NOTE: no native async for output parser
        return self._run_component(**kwargs)

    def _validate_component_inputs(self, input: Any) -> Any:
        """Validate component inputs during run_component."""
        input["input"] = validate_and_convert_stringable(input["input"])
        return input

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""

    @property
    def input_keys(self) -> Any:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> Any:
        """Output keys."""
        return OutputKeys.from_keys({"output"})
