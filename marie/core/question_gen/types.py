from abc import abstractmethod
from typing import List, Sequence

from marie.core.bridge.pydantic import BaseModel
from marie.core.instrumentation import DispatcherSpanMixin
from marie.core.prompts.mixin import PromptMixin, PromptMixinType
from marie.core.schema import QueryBundle
from marie.core.tools.types import ToolMetadata


class SubQuestion(BaseModel):
    sub_question: str
    tool_name: str


class SubQuestionList(BaseModel):
    """A pydantic object wrapping a list of sub-questions.

    This is mostly used to make getting a json schema easier.
    """

    items: List[SubQuestion]


class BaseQuestionGenerator(PromptMixin, DispatcherSpanMixin):
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    @abstractmethod
    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass

    @abstractmethod
    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass
