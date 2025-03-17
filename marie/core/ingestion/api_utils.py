import os
import httpx
from typing import Optional, TYPE_CHECKING

from marie.core.constants import (
    DEFAULT_APP_URL,
    DEFAULT_BASE_URL,
)

if TYPE_CHECKING:
    from llama_cloud.client import AsyncLlamaCloud, LlamaCloud


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    app_url: Optional[str] = None,
    timeout: int = 60,
    httpx_client: Optional[httpx.Client] = None,
) -> "LlamaCloud":
    """Get the sync platform API client."""
    from llama_cloud.client import LlamaCloud

    base_url = base_url or os.environ.get("LLAMA_CLOUD_BASE_URL", DEFAULT_BASE_URL)
    app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
    api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY", None)

    return LlamaCloud(
        base_url=base_url, token=api_key, timeout=timeout, httpx_client=httpx_client
    )


def get_aclient(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    app_url: Optional[str] = None,
    timeout: int = 60,
    httpx_client: Optional[httpx.AsyncClient] = None,
) -> "AsyncLlamaCloud":
    """Get the async platform API client."""
    from llama_cloud.client import AsyncLlamaCloud

    base_url = base_url or os.environ.get("LLAMA_CLOUD_BASE_URL", DEFAULT_BASE_URL)
    app_url = app_url or os.environ.get("LLAMA_CLOUD_APP_URL", DEFAULT_APP_URL)
    api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY", None)

    return AsyncLlamaCloud(
        base_url=base_url, token=api_key, timeout=timeout, httpx_client=httpx_client
    )
