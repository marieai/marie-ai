from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Optional

from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.queue_io import ValkeyListQueueClient
from marie.engine.openai_compat import (
    build_async_openai_client,
    resolve_openai_base_url_from_env,
)
from marie.excepts import RuntimeFailToStart
from marie.logging_core.logger import MarieLogger


class GatewayLlmDispatchRuntime:
    """Gateway-owned lifecycle wrapper for the LLM dispatch runtime."""

    def __init__(
        self,
        *,
        logger: Optional[MarieLogger] = None,
        queue_config: Optional[LlmQueueConfig] = None,
        queue_client_factory: Optional[Callable[[str], Any]] = None,
        openai_client_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
        dispatcher_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        # self.logger = logger
        self.logger = MarieLogger("GatewayLlmDispatchRuntime")
        self.config = queue_config or LlmQueueConfig.from_env()
        self._queue_client_factory = queue_client_factory or ValkeyListQueueClient
        self._openai_client_factory = openai_client_factory or build_async_openai_client
        self._dispatcher_factory = dispatcher_factory or _build_dispatcher
        self._queue_client = None
        self._dispatcher = None
        self._last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def health(self) -> dict[str, object]:
        if self._dispatcher is not None:
            health = dict(self._dispatcher.health())
            health.setdefault("last_error", self._last_error)
            return health

        return {
            "enabled": self.config.enabled,
            "pool_id": self.config.pool_id,
            "valkey_configured": bool(self.config.valkey_url),
            "running": False,
            "last_error": self._last_error,
        }

    async def start(self) -> None:
        if not self.enabled:
            self.logger.info("LLM dispatch runtime disabled")
            return
        if self._dispatcher is not None:
            self.logger.info("LLM dispatch runtime already started")
            return

        queue_client = None
        dispatcher = None
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeFailToStart(
                    "LLM dispatch runtime is enabled but OPENAI_API_KEY is not configured."
                )
            if not self.config.valkey_url:
                raise RuntimeFailToStart(
                    "LLM dispatch runtime is enabled but LLM_QUEUE_VALKEY_URL is not configured."
                )

            queue_client = self._queue_client_factory(self.config.valkey_url)
            queue_client.request_queue_depth(self.config.pool_id)

            openai_base_url = resolve_openai_base_url_from_env()
            dispatcher = self._dispatcher_factory(
                queue_client=queue_client,
                client=None,
                client_factory=partial(
                    self._openai_client_factory,
                    api_key,
                    openai_base_url,
                ),
                config=self.config,
                logger=self.logger,
                backend_address=openai_base_url,
            )
            dispatcher.start()

            self._queue_client = queue_client
            self._dispatcher = dispatcher
            self._last_error = None
            self.logger.info(
                "Started LLM dispatch runtime for pool '%s' against %s",
                self.config.pool_id,
                openai_base_url or "default OpenAI endpoint",
            )
        except Exception as exc:
            self._last_error = str(exc)
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    self.logger.exception(
                        "Failed stopping partially started LLM dispatch runtime"
                    )
            if queue_client is not None:
                try:
                    queue_client.close()
                except Exception:
                    self.logger.exception(
                        "Failed closing queue client after LLM dispatch startup error"
                    )
            if isinstance(exc, RuntimeFailToStart):
                raise
            raise RuntimeFailToStart(
                f"Failed to start LLM dispatch runtime: {exc}"
            ) from exc

    async def stop(self) -> None:
        dispatcher = self._dispatcher
        queue_client = self._queue_client

        self._dispatcher = None
        self._queue_client = None

        if dispatcher is not None:
            try:
                dispatcher.stop()
            except Exception:
                self.logger.exception("Failed stopping LLM dispatch runtime")

        if queue_client is not None:
            try:
                queue_client.close()
            except Exception:
                self.logger.exception("Failed closing LLM dispatch queue client")

        if self.enabled:
            self.logger.info(
                "Stopped LLM dispatch runtime for pool '%s'", self.config.pool_id
            )


def _build_dispatcher(
    *,
    queue_client,
    client,
    client_factory=None,
    config,
    logger,
    backend_address: Optional[str] = None,
):
    from marie.engine.llm_queue.adapters.openai_compatible import (
        OpenAICompatibleExecutionAdapter,
    )
    from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher

    return QueuedBatchDispatcher(
        queue_client=queue_client,
        execution_adapter=OpenAICompatibleExecutionAdapter(
            client=client,
            client_factory=client_factory,
            logger=logger,
            default_timeout=None,
            backend_address=backend_address,
        ),
        config=config,
        logger=logger,
    )
