"""
Circuit Breaker implementation for load balancer reliability.

The circuit breaker tracks failures passively and temporarily excludes
nodes that exceed the failure threshold, preventing cascading failures.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from marie.logging_core.logger import MarieLogger


class CircuitState(Enum):
    """State of the circuit breaker for a given address."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Circuit tripped, node excluded from selection
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker.

    Attributes:
        failure_threshold: Number of consecutive failures before opening the circuit.
        success_threshold: Number of successes in half-open state to close the circuit.
        recovery_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN.
        half_open_max_calls: Maximum concurrent requests allowed in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1


@dataclass
class CircuitStats:
    """Statistics for a single address's circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    open_time: Optional[float] = None
    half_open_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Per-address circuit breaker with passive failure tracking.

    State machine:
    - CLOSED → OPEN: After `failure_threshold` consecutive failures
    - OPEN → HALF_OPEN: After `recovery_timeout` seconds
    - HALF_OPEN → CLOSED: After `success_threshold` successes
    - HALF_OPEN → OPEN: On any failure

    Thread-safe: All operations are protected by a lock.
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        logger: Optional[MarieLogger] = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration. Uses defaults if not provided.
            logger: Logger instance for debug output.
        """
        self._config = config or CircuitBreakerConfig()
        self._logger = logger or MarieLogger(self.__class__.__name__)
        self._stats: Dict[str, CircuitStats] = {}
        self._lock = threading.Lock()

    def is_available(self, address: str) -> bool:
        """
        Check if a connection to the address is available.

        A connection is available if:
        - Circuit is CLOSED
        - Circuit is HALF_OPEN and we haven't exceeded max half-open calls
        - Circuit is OPEN but recovery timeout has passed (transitions to HALF_OPEN)

        Args:
            address: The address to check.

        Returns:
            True if the address can be used for requests, False otherwise.
        """
        with self._lock:
            stats = self._get_or_create_stats(address)
            current_time = time.monotonic()

            if stats.state == CircuitState.CLOSED:
                return True

            if stats.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (
                    stats.open_time is not None
                    and (current_time - stats.open_time)
                    >= self._config.recovery_timeout
                ):
                    # Transition to half-open
                    self._transition_to_half_open(stats, address)
                    return stats.half_open_calls < self._config.half_open_max_calls
                return False

            if stats.state == CircuitState.HALF_OPEN:
                return stats.half_open_calls < self._config.half_open_max_calls

            return False

    def record_failure(self, address: str) -> None:
        """
        Record a failure for the given address.

        This may cause the circuit to open if the failure threshold is reached.

        Args:
            address: The address that experienced a failure.
        """
        with self._lock:
            stats = self._get_or_create_stats(address)
            current_time = time.monotonic()

            stats.consecutive_failures += 1
            stats.consecutive_successes = 0
            stats.last_failure_time = current_time
            stats.total_failures += 1

            if stats.state == CircuitState.CLOSED:
                if stats.consecutive_failures >= self._config.failure_threshold:
                    self._transition_to_open(stats, address, current_time)

            elif stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._transition_to_open(stats, address, current_time)

    def record_success(self, address: str) -> None:
        """
        Record a success for the given address.

        In HALF_OPEN state, this may close the circuit if enough successes occur.

        Args:
            address: The address that experienced a success.
        """
        with self._lock:
            stats = self._get_or_create_stats(address)
            current_time = time.monotonic()

            stats.consecutive_successes += 1
            stats.consecutive_failures = 0
            stats.last_success_time = current_time
            stats.total_successes += 1

            if stats.state == CircuitState.HALF_OPEN:
                if stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to_closed(stats, address)

    def increment_half_open_calls(self, address: str) -> None:
        """
        Increment the count of half-open calls for the address.

        Should be called when a request is sent to a half-open circuit.

        Args:
            address: The address being called.
        """
        with self._lock:
            stats = self._stats.get(address)
            if stats and stats.state == CircuitState.HALF_OPEN:
                stats.half_open_calls += 1

    def decrement_half_open_calls(self, address: str) -> None:
        """
        Decrement the count of half-open calls for the address.

        Should be called when a request to a half-open circuit completes.

        Args:
            address: The address that completed a call.
        """
        with self._lock:
            stats = self._stats.get(address)
            if stats and stats.state == CircuitState.HALF_OPEN:
                stats.half_open_calls = max(0, stats.half_open_calls - 1)

    def remove_address(self, address: str) -> None:
        """
        Remove tracking for an address.

        Called when a connection is removed from the pool.

        Args:
            address: The address to remove.
        """
        with self._lock:
            if address in self._stats:
                del self._stats[address]
                self._logger.debug(f"Circuit breaker removed tracking for: {address}")

    def get_state(self, address: str) -> CircuitState:
        """
        Get the current circuit state for an address.

        Args:
            address: The address to check.

        Returns:
            The current CircuitState.
        """
        with self._lock:
            stats = self._stats.get(address)
            if stats is None:
                return CircuitState.CLOSED
            return stats.state

    def get_stats(self, address: str) -> Optional[CircuitStats]:
        """
        Get detailed statistics for an address.

        Args:
            address: The address to get stats for.

        Returns:
            CircuitStats for the address, or None if not tracked.
        """
        with self._lock:
            stats = self._stats.get(address)
            if stats is None:
                return None
            # Return a copy to avoid external modification
            return CircuitStats(
                state=stats.state,
                consecutive_failures=stats.consecutive_failures,
                consecutive_successes=stats.consecutive_successes,
                last_failure_time=stats.last_failure_time,
                last_success_time=stats.last_success_time,
                open_time=stats.open_time,
                half_open_calls=stats.half_open_calls,
                total_failures=stats.total_failures,
                total_successes=stats.total_successes,
            )

    def get_all_stats(self) -> Dict[str, CircuitStats]:
        """
        Get statistics for all tracked addresses.

        Returns:
            Dictionary mapping addresses to their CircuitStats.
        """
        with self._lock:
            return {
                addr: CircuitStats(
                    state=stats.state,
                    consecutive_failures=stats.consecutive_failures,
                    consecutive_successes=stats.consecutive_successes,
                    last_failure_time=stats.last_failure_time,
                    last_success_time=stats.last_success_time,
                    open_time=stats.open_time,
                    half_open_calls=stats.half_open_calls,
                    total_failures=stats.total_failures,
                    total_successes=stats.total_successes,
                )
                for addr, stats in self._stats.items()
            }

    def reset(self, address: str) -> None:
        """
        Reset the circuit breaker for an address to closed state.

        Useful after a manual intervention or connection reset.

        Args:
            address: The address to reset.
        """
        with self._lock:
            if address in self._stats:
                self._stats[address] = CircuitStats()
                self._logger.info(f"Circuit breaker reset for: {address}")

    def _get_or_create_stats(self, address: str) -> CircuitStats:
        """Get stats for an address, creating if necessary. Must hold lock."""
        if address not in self._stats:
            self._stats[address] = CircuitStats()
        return self._stats[address]

    def _transition_to_open(
        self, stats: CircuitStats, address: str, current_time: float
    ) -> None:
        """Transition circuit to OPEN state. Must hold lock."""
        stats.state = CircuitState.OPEN
        stats.open_time = current_time
        stats.half_open_calls = 0
        self._logger.warning(
            f"Circuit OPEN for {address} after {stats.consecutive_failures} failures"
        )

    def _transition_to_half_open(self, stats: CircuitStats, address: str) -> None:
        """Transition circuit to HALF_OPEN state. Must hold lock."""
        stats.state = CircuitState.HALF_OPEN
        stats.half_open_calls = 0
        stats.consecutive_failures = 0
        stats.consecutive_successes = 0
        self._logger.info(f"Circuit HALF_OPEN for {address}, testing recovery")

    def _transition_to_closed(self, stats: CircuitStats, address: str) -> None:
        """Transition circuit to CLOSED state. Must hold lock."""
        stats.state = CircuitState.CLOSED
        stats.open_time = None
        stats.half_open_calls = 0
        stats.consecutive_failures = 0
        self._logger.info(
            f"Circuit CLOSED for {address} after {stats.consecutive_successes} successes"
        )
