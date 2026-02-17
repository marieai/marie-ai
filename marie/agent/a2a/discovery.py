"""A2A agent discovery with caching.

This module provides utilities for discovering and caching
information about external A2A agents.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from marie.agent.a2a.client import A2AClient
from marie.agent.a2a.constants import (
    AGENT_CARD_PATH,
    DEFAULT_DISCOVERY_CACHE_TTL,
    DEFAULT_REQUEST_TIMEOUT,
)
from marie.agent.a2a.errors import AgentDiscoveryError
from marie.agent.a2a.types import AgentCard

logger = logging.getLogger(__name__)


@dataclass
class CachedAgent:
    """Cached agent card with metadata."""

    card: AgentCard
    url: str
    discovered_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self, ttl: float) -> bool:
        """Check if the cache entry is expired."""
        return time.time() - self.discovered_at > ttl


class A2AAgentDiscovery:
    """Service for discovering and caching A2A agents.

    Provides agent discovery with automatic caching and TTL management.
    Useful for multi-agent scenarios where agents need to discover
    and communicate with each other.

    Example:
        discovery = A2AAgentDiscovery()

        # Discover an agent
        card = await discovery.discover("http://agent.example.com")

        # Get a client for the agent
        client = await discovery.get_client("http://agent.example.com")

        # List cached agents
        agents = discovery.list_agents()
    """

    def __init__(
        self,
        cache_ttl: float = DEFAULT_DISCOVERY_CACHE_TTL,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize the discovery service.

        Args:
            cache_ttl: Cache time-to-live in seconds.
            timeout: Request timeout in seconds.
            http_client: Optional shared HTTP client.
        """
        self._cache: dict[str, CachedAgent] = {}
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._http_client = http_client
        self._owns_client = http_client is None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        """Close resources."""
        if self._owns_client and self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "A2AAgentDiscovery":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._http_client

    async def discover(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> AgentCard:
        """Discover an agent at the given URL.

        Args:
            url: Base URL of the A2A agent.
            force_refresh: If True, bypass cache and refresh.

        Returns:
            The agent's AgentCard.

        Raises:
            AgentDiscoveryError: If discovery fails.
        """
        normalized_url = self._normalize_url(url)

        async with self._lock:
            # Check cache
            if not force_refresh and normalized_url in self._cache:
                cached = self._cache[normalized_url]
                if not cached.is_expired(self._cache_ttl):
                    cached.last_accessed = time.time()
                    return cached.card

            # Discover agent
            card = await self._fetch_agent_card(normalized_url)

            # Cache result
            self._cache[normalized_url] = CachedAgent(
                card=card,
                url=normalized_url,
            )

            return card

    async def get_client(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> A2AClient:
        """Get an A2AClient for an agent.

        Discovers the agent if not cached, then creates a client.

        Args:
            url: Base URL of the A2A agent.
            force_refresh: If True, refresh agent card.

        Returns:
            Configured A2AClient instance.
        """
        card = await self.discover(url, force_refresh=force_refresh)
        return A2AClient(
            agent_card=card,
            http_client=self._get_client() if not self._owns_client else None,
            timeout=self._timeout,
        )

    async def discover_many(
        self,
        urls: list[str],
        ignore_errors: bool = True,
    ) -> dict[str, AgentCard]:
        """Discover multiple agents concurrently.

        Args:
            urls: List of agent URLs to discover.
            ignore_errors: If True, continue on individual failures.

        Returns:
            Dictionary mapping URLs to AgentCards.
        """
        results: dict[str, AgentCard] = {}

        async def discover_one(url: str) -> None:
            try:
                card = await self.discover(url)
                results[url] = card
            except AgentDiscoveryError as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Failed to discover agent at {url}: {e}")

        await asyncio.gather(
            *[discover_one(url) for url in urls],
            return_exceptions=ignore_errors,
        )

        return results

    def list_agents(self) -> list[CachedAgent]:
        """List all cached agents.

        Returns:
            List of cached agent entries.
        """
        # Clean expired entries
        now = time.time()
        expired = [
            url
            for url, cached in self._cache.items()
            if cached.is_expired(self._cache_ttl)
        ]
        for url in expired:
            del self._cache[url]

        return list(self._cache.values())

    def get_cached(self, url: str) -> Optional[AgentCard]:
        """Get a cached agent card without discovery.

        Args:
            url: Agent URL.

        Returns:
            Cached AgentCard or None.
        """
        normalized_url = self._normalize_url(url)
        cached = self._cache.get(normalized_url)

        if cached and not cached.is_expired(self._cache_ttl):
            cached.last_accessed = time.time()
            return cached.card

        return None

    def invalidate(self, url: str) -> bool:
        """Invalidate a cached agent.

        Args:
            url: Agent URL to invalidate.

        Returns:
            True if entry was removed, False if not found.
        """
        normalized_url = self._normalize_url(url)
        if normalized_url in self._cache:
            del self._cache[normalized_url]
            return True
        return False

    def clear_cache(self) -> int:
        """Clear all cached agents.

        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    async def _fetch_agent_card(self, base_url: str) -> AgentCard:
        """Fetch agent card from URL."""
        client = self._get_client()
        card_url = f"{base_url}{AGENT_CARD_PATH}"

        try:
            response = await client.get(card_url)
            response.raise_for_status()

            card_data = response.json()
            return AgentCard(**card_data)

        except httpx.HTTPStatusError as e:
            raise AgentDiscoveryError(
                base_url,
                f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            raise AgentDiscoveryError(base_url, f"Request failed: {e}")
        except Exception as e:
            raise AgentDiscoveryError(base_url, str(e))

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for caching."""
        return url.rstrip("/")


class AgentRegistry:
    """Registry for managing known agents.

    Provides a higher-level interface for managing a set of
    agents that can be discovered and used by name.

    Example:
        registry = AgentRegistry()
        registry.register("calculator", "http://calc-agent:9000")
        registry.register("search", "http://search-agent:9000")

        calc_client = await registry.get_client("calculator")
        result = await calc_client.send_message("2 + 2")
    """

    def __init__(
        self,
        discovery: Optional[A2AAgentDiscovery] = None,
    ):
        """Initialize the registry.

        Args:
            discovery: Optional discovery service to use.
        """
        self._discovery = discovery or A2AAgentDiscovery()
        self._agents: dict[str, str] = {}  # name -> url
        self._owns_discovery = discovery is None

    async def close(self) -> None:
        """Close resources."""
        if self._owns_discovery:
            await self._discovery.close()

    def register(self, name: str, url: str) -> None:
        """Register an agent by name.

        Args:
            name: Friendly name for the agent.
            url: Agent's A2A endpoint URL.
        """
        self._agents[name] = url

    def unregister(self, name: str) -> bool:
        """Unregister an agent.

        Args:
            name: Agent name.

        Returns:
            True if removed, False if not found.
        """
        if name in self._agents:
            url = self._agents.pop(name)
            self._discovery.invalidate(url)
            return True
        return False

    async def get_card(self, name: str) -> AgentCard:
        """Get agent card by name.

        Args:
            name: Agent name.

        Returns:
            The agent's AgentCard.

        Raises:
            KeyError: If agent not registered.
        """
        if name not in self._agents:
            raise KeyError(f"Agent not registered: {name}")

        return await self._discovery.discover(self._agents[name])

    async def get_client(self, name: str) -> A2AClient:
        """Get a client for an agent by name.

        Args:
            name: Agent name.

        Returns:
            A2AClient for the agent.

        Raises:
            KeyError: If agent not registered.
        """
        if name not in self._agents:
            raise KeyError(f"Agent not registered: {name}")

        return await self._discovery.get_client(self._agents[name])

    def list_registered(self) -> dict[str, str]:
        """List all registered agents.

        Returns:
            Dictionary mapping names to URLs.
        """
        return dict(self._agents)

    async def discover_all(self) -> dict[str, AgentCard]:
        """Discover all registered agents.

        Returns:
            Dictionary mapping names to AgentCards.
        """
        result: dict[str, AgentCard] = {}
        for name, url in self._agents.items():
            try:
                result[name] = await self._discovery.discover(url)
            except AgentDiscoveryError as e:
                logger.warning(f"Failed to discover {name}: {e}")
        return result
