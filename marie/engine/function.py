import asyncio
import functools
import time
from abc import ABC, abstractmethod
from typing import Any, Coroutine, List, Union

FunctionReturnType = Union[str, List[str]]


class Function(ABC):
    """
    The class to define a function that can be called by the model.
    Supports both synchronous and asynchronous execution.
    """

    def __call__(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        return result

    async def acall(self, *args, **kwargs):
        try:
            return await self.aforward(*args, **kwargs)
        except NotImplementedError:
            import warnings

            warnings.warn(
                "Running synchronous forward() in thread pool. "
                "Consider implementing aforward() for better performance."
            )
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(self.forward, *args, **kwargs)
            )

    def forward(self, *args: Any, **kwargs: Any) -> FunctionReturnType:
        raise NotImplementedError("forward() must be implemented")

    async def aforward(self, *args: Any, **kwargs: Any) -> FunctionReturnType:
        raise NotImplementedError("aforward() not implemented")


class TextProcessingFunction(Function):
    """Demo implementation of Function that processes text"""

    def forward(self, text: str) -> str:
        # Simulate a time-consuming operation
        time.sleep(1)
        return f"Processed: {text.upper()}"


class AsyncTextProcessingFunction(Function):
    """Demo implementation with an async forward method"""

    async def aforward(self, text: str) -> str:
        # Simulate an async operation
        await asyncio.sleep(1)
        return f"Async processed: {text.upper()}"


async def main():
    # Synchronous function used in both ways
    sync_func = TextProcessingFunction()

    # Sync usage
    result1 = sync_func("hello")
    print(result1)

    # Async usage of sync function (runs in thread pool)
    result2 = await sync_func.acall("world")
    print(result2)

    # Async function
    async_func = AsyncTextProcessingFunction()

    # CORRECT: Async usage for async functions
    result3 = await async_func.acall("testing")
    print(result3)

    # Another async usage example
    result4 = await async_func.acall("async")
    print(result4)

    # Parallel execution
    results = await asyncio.gather(
        sync_func.acall("parallel1"), async_func.acall("parallel2")
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
