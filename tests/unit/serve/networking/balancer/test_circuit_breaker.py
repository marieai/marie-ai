"""Unit tests for CircuitBreaker implementation."""

import time
from unittest.mock import MagicMock

import pytest

from marie.serve.networking.balancer.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitStats,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig defaults."""

    def test_default_config(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 1

    def test_custom_config(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=10.0,
            half_open_max_calls=2,
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.recovery_timeout == 10.0
        assert config.half_open_max_calls == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with fast timeouts for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=0.1,  # 100ms for faster tests
            half_open_max_calls=1,
        )
        return CircuitBreaker(config=config)

    def test_initial_state_is_closed(self, breaker):
        """New addresses start in CLOSED state."""
        assert breaker.get_state("127.0.0.1:8080") == CircuitState.CLOSED
        assert breaker.is_available("127.0.0.1:8080") is True

    def test_single_failure_keeps_circuit_closed(self, breaker):
        """Single failure doesn't open the circuit."""
        address = "127.0.0.1:8080"
        breaker.record_failure(address)
        assert breaker.get_state(address) == CircuitState.CLOSED
        assert breaker.is_available(address) is True

    def test_circuit_opens_after_threshold_failures(self, breaker):
        """Circuit opens after failure_threshold consecutive failures."""
        address = "127.0.0.1:8080"

        # Record failures up to threshold
        for _ in range(3):
            breaker.record_failure(address)

        assert breaker.get_state(address) == CircuitState.OPEN
        assert breaker.is_available(address) is False

    def test_success_resets_failure_count(self, breaker):
        """Success resets consecutive failure count."""
        address = "127.0.0.1:8080"

        # Record 2 failures (below threshold)
        breaker.record_failure(address)
        breaker.record_failure(address)

        # Success resets the count
        breaker.record_success(address)

        # 2 more failures shouldn't open circuit (total consecutive = 2)
        breaker.record_failure(address)
        breaker.record_failure(address)
        assert breaker.get_state(address) == CircuitState.CLOSED

    def test_circuit_transitions_to_half_open_after_timeout(self, breaker):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        address = "127.0.0.1:8080"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(address)
        assert breaker.get_state(address) == CircuitState.OPEN

        # Wait for recovery timeout (100ms + buffer)
        time.sleep(0.15)

        # Circuit should now be available (transitions to HALF_OPEN)
        assert breaker.is_available(address) is True
        assert breaker.get_state(address) == CircuitState.HALF_OPEN

    def test_half_open_closes_after_success_threshold(self, breaker):
        """Circuit closes after success_threshold successes in HALF_OPEN state."""
        address = "127.0.0.1:8080"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(address)

        # Wait for recovery timeout
        time.sleep(0.15)

        # Trigger transition to HALF_OPEN
        breaker.is_available(address)
        assert breaker.get_state(address) == CircuitState.HALF_OPEN

        # Record successes up to threshold
        breaker.record_success(address)
        assert breaker.get_state(address) == CircuitState.HALF_OPEN
        breaker.record_success(address)
        assert breaker.get_state(address) == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self, breaker):
        """Any failure in HALF_OPEN state reopens the circuit."""
        address = "127.0.0.1:8080"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(address)

        # Wait for recovery timeout
        time.sleep(0.15)

        # Trigger transition to HALF_OPEN
        breaker.is_available(address)
        assert breaker.get_state(address) == CircuitState.HALF_OPEN

        # Failure reopens the circuit
        breaker.record_failure(address)
        assert breaker.get_state(address) == CircuitState.OPEN

    def test_remove_address_clears_state(self, breaker):
        """Removing an address clears all its state."""
        address = "127.0.0.1:8080"

        # Build up some state
        breaker.record_failure(address)
        breaker.record_failure(address)
        breaker.record_success(address)

        # Remove the address
        breaker.remove_address(address)

        # State should be clean (defaults to CLOSED)
        assert breaker.get_state(address) == CircuitState.CLOSED
        stats = breaker.get_stats(address)
        assert stats is None  # No stats for unknown address

    def test_reset_clears_address_state(self, breaker):
        """Reset returns address to initial CLOSED state."""
        address = "127.0.0.1:8080"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(address)
        assert breaker.get_state(address) == CircuitState.OPEN

        # Reset
        breaker.reset(address)
        assert breaker.get_state(address) == CircuitState.CLOSED
        assert breaker.is_available(address) is True

    def test_get_stats_returns_copy(self, breaker):
        """get_stats returns a copy that can't affect internal state."""
        address = "127.0.0.1:8080"
        breaker.record_failure(address)

        stats = breaker.get_stats(address)
        assert stats is not None
        assert stats.consecutive_failures == 1

        # Modifying returned stats shouldn't affect internal state
        stats.consecutive_failures = 999
        internal_stats = breaker.get_stats(address)
        assert internal_stats.consecutive_failures == 1

    def test_get_all_stats(self, breaker):
        """get_all_stats returns stats for all tracked addresses."""
        breaker.record_failure("addr1")
        breaker.record_success("addr2")

        all_stats = breaker.get_all_stats()
        assert "addr1" in all_stats
        assert "addr2" in all_stats
        assert all_stats["addr1"].consecutive_failures == 1
        assert all_stats["addr2"].consecutive_successes == 1

    def test_half_open_max_calls_limit(self, breaker):
        """Only half_open_max_calls requests allowed in HALF_OPEN state."""
        address = "127.0.0.1:8080"

        # Open the circuit
        for _ in range(3):
            breaker.record_failure(address)

        # Wait for recovery timeout
        time.sleep(0.15)

        # First call should be allowed (transitions to HALF_OPEN)
        assert breaker.is_available(address) is True
        breaker.increment_half_open_calls(address)

        # Second call should be blocked (max_calls = 1)
        assert breaker.is_available(address) is False

        # After decrement, another call should be allowed
        breaker.decrement_half_open_calls(address)
        assert breaker.is_available(address) is True

    def test_stats_tracking(self, breaker):
        """Total failure and success counts are tracked correctly."""
        address = "127.0.0.1:8080"

        breaker.record_success(address)
        breaker.record_failure(address)
        breaker.record_success(address)
        breaker.record_success(address)
        breaker.record_failure(address)

        stats = breaker.get_stats(address)
        assert stats.total_successes == 3
        assert stats.total_failures == 2


class TestCircuitBreakerThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_operations(self):
        """Circuit breaker handles concurrent operations safely."""
        import threading

        config = CircuitBreakerConfig(failure_threshold=100)
        breaker = CircuitBreaker(config=config)
        address = "127.0.0.1:8080"
        errors = []

        def record_operations():
            try:
                for _ in range(100):
                    breaker.record_failure(address)
                    breaker.record_success(address)
                    breaker.is_available(address)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_operations) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
