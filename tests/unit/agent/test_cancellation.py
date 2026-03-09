"""Tests for cancellation primitives."""

import asyncio

import pytest

from marie.agent.cancellation import AbortController, AbortError, AbortSignal


class TestAbortSignal:
    def test_initial_state(self):
        signal = AbortSignal()
        assert not signal.aborted
        assert signal.reason is None

    def test_throw_if_aborted_noop_when_not_aborted(self):
        signal = AbortSignal()
        signal.throw_if_aborted()  # Should not raise

    def test_throw_if_aborted_raises_when_aborted(self):
        controller = AbortController()
        controller.abort("cancelled")
        with pytest.raises(AbortError, match="cancelled"):
            controller.signal.throw_if_aborted()

    def test_listener_called_on_abort(self):
        controller = AbortController()
        reasons = []
        controller.signal.on_abort(lambda r: reasons.append(r))

        controller.abort("test reason")
        assert reasons == ["test reason"]

    def test_listener_called_immediately_if_already_aborted(self):
        controller = AbortController()
        controller.abort("early")

        reasons = []
        controller.signal.on_abort(lambda r: reasons.append(r))
        assert reasons == ["early"]

    def test_abort_idempotent(self):
        controller = AbortController()
        count = []
        controller.signal.on_abort(lambda r: count.append(1))

        controller.abort("first")
        controller.abort("second")  # Should be a no-op
        assert len(count) == 1
        assert controller.signal.reason == "first"


class TestAbortController:
    def test_abort_sets_signal(self):
        controller = AbortController()
        assert not controller.signal.aborted
        controller.abort("done")
        assert controller.signal.aborted
        assert controller.signal.reason == "done"

    def test_default_reason(self):
        controller = AbortController()
        controller.abort()
        assert controller.signal.reason == "Operation aborted"


class TestAbortSignalTimeout:
    @pytest.mark.asyncio
    async def test_timeout_aborts_after_delay(self):
        signal = AbortSignal.timeout(0.05)
        assert not signal.aborted
        await asyncio.sleep(0.1)
        assert signal.aborted
        assert "Timeout" in (signal.reason or "")

    @pytest.mark.asyncio
    async def test_timeout_not_aborted_before_delay(self):
        signal = AbortSignal.timeout(1.0)
        await asyncio.sleep(0.01)
        assert not signal.aborted
