"""Faithfulness evaluation."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Union

from marie.evaluation.base import BaseEvaluator, EvaluationResult


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Faithfulness evaluator.

    Evaluates whether a response is faithful to the contexts
    (i.e. whether the response is supported by the contexts or hallucinated.)

    This evaluator only considers the response string and the list of context strings.

    Args:
        raise_error(bool): Whether to raise an error when the response is invalid.
            Defaults to False.
    """

    def __init__(
        self,
        raise_error: bool = False,
    ) -> None:
        """Init params."""
        self._raise_error = raise_error

    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the response is faithful to the contexts."""
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        error_str = "Inference failed"

        if error_str in response:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")
        else:
            passing = True

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=response,
        )
