"""Wrapper functions around an LLM chain."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from marie.core.base.llms.types import LLMMetadata
from marie.core.callbacks.base import CallbackManager
from marie.core.instrumentation import DispatcherSpanMixin
from marie.core.llms.llm import LLM
from marie.core.prompts.base import BasePromptTemplate
from marie.core.schema import BaseComponent
from marie.core.types import TokenAsyncGen, TokenGen


logger = logging.getLogger(__name__)


class BaseLLMPredictor(BaseComponent, DispatcherSpanMixin, ABC):
    """Base LLM Predictor."""

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        print("here", flush=True)
        data = super().model_dump(**kwargs)
        data["llm"] = self.llm.to_dict()
        return data

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Keep for backwards compatibility."""
        return self.model_dump(**kwargs)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict()
        return data

    @property
    @abstractmethod
    def llm(self) -> LLM:
        """Get LLM."""

    @property
    @abstractmethod
    def callback_manager(self) -> CallbackManager:
        """Get callback manager."""

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""

    @abstractmethod
    def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Predict the answer to a query."""

    @abstractmethod
    def stream(self, prompt: BasePromptTemplate, **prompt_args: Any) -> TokenGen:
        """Stream the answer to a query."""

    @abstractmethod
    async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Async predict the answer to a query."""

    @abstractmethod
    async def astream(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> TokenAsyncGen:
        """Async predict the answer to a query."""


class LLMPredictor(BaseLLMPredictor):
    """LLM predictor class.

    NOTE: Deprecated. Use any LLM class directly.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        raise ValueError("This class is deprecated. Use any LLM class directly.")
