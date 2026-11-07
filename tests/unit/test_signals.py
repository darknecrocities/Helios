"""Unit and boundary tests for Helios POSIX Signal Trapping and Watchdog Engine."""

from __future__ import annotations

import signal
import threading
import time
from unittest.mock import MagicMock

import pytest

from helios.training.signals import (
    EmergencyWatchdog,
    InterruptionHandler,
    SignalState,
    get_default_signals,
)


@pytest.mark.unit
def test_signal_state_enum_values() -> None:
    """Validate all states exist and have expected string values."""
    assert SignalState.IDLE.value == "IDLE"
    assert SignalState.TRAINING.value == "TRAINING"
    assert SignalState.INTERRUPT_REQUESTED.value == "INTERRUPT_REQUESTED"
    assert SignalState.SAVING.value == "SAVING"
    assert SignalState.SHUTDOWN_CLEAN.value == "SHUTDOWN_CLEAN"


@pytest.mark.unit
def test_initialization_defaults() -> None:
    """Verify default initial state of InterruptionHandler."""
    handler = InterruptionHandler()
    assert handler.state == SignalState.IDLE
    assert not handler.interrupted
    assert handler.received_signal is None
    assert handler.signal_count == 0
    assert not handler.should_stop()


@pytest.mark.unit
def test_context_manager_enter_exit_lifecycle() -> None:
    """Verify state transitions from IDLE -> TRAINING on enter, and back to IDLE on exit."""
    handler = InterruptionHandler()
    assert handler.state == SignalState.IDLE
    with handler:
        assert handler.state == SignalState.TRAINING
        assert not handler.should_stop()
    assert handler.state == SignalState.IDLE


@pytest.mark.unit
def test_original_signal_handler_restoration() -> None:
    """Verify original signal handlers are captured and cleanly restored."""
    sig = signal.SIGINT
    original = signal.getsignal(sig)
    handler = InterruptionHandler(signals=(sig,))

    with handler:
        current = signal.getsignal(sig)
        assert current != original

    restored = signal.getsignal(sig)
    assert restored == original


@pytest.mark.unit
def test_microbatch_boundary_should_stop() -> None:
    """Simulate a loop; trigger interrupt and confirm should_stop() halts at clean boundary."""
    handler = InterruptionHandler()
    processed_batches = 0

    with handler:
        for batch_idx in range(5):
            if handler.should_stop():
                break
            processed_batches += 1
            if batch_idx == 1:
                handler.trigger_interrupt(signal.SIGINT)

    assert processed_batches == 2
    assert handler.should_stop()
    assert handler.interrupted
    assert handler.received_signal == signal.SIGINT
    assert handler.signal_count == 1


@pytest.mark.unit
def test_execute_with_watchdog_success() -> None:
    """Calling execute_with_watchdog executes action and transitions to SHUTDOWN_CLEAN."""
    handler = InterruptionHandler()
    executed = False

    def save_action() -> str:
        nonlocal executed
        executed = True
        return "saved_path"

    result = handler.execute_with_watchdog(save_action)
    assert executed is True
    assert result == "saved_path"
    assert handler.state == SignalState.SHUTDOWN_CLEAN
    assert not handler.watchdog.is_running
    assert not handler.watchdog.is_timed_out


@pytest.mark.unit
def test_watchdog_timeout_triggers_fallback() -> None:
    """Watchdog triggers fallback callback if persistence takes longer than timeout."""
    fallback_mock = MagicMock()
    watchdog = EmergencyWatchdog(timeout=0.1, fallback_fn=fallback_mock)
    watchdog.start()
    time.sleep(0.25)
    assert watchdog.is_timed_out
    fallback_mock.assert_called_once()
    watchdog.cancel()


@pytest.mark.unit
def test_watchdog_cancel_prevents_fallback() -> None:
    """Cancelling watchdog before timeout prevents fallback execution."""
    fallback_mock = MagicMock()
    watchdog = EmergencyWatchdog(timeout=0.4, fallback_fn=fallback_mock)
    watchdog.start()
    time.sleep(0.05)
    watchdog.cancel()
    time.sleep(0.45)
    assert not watchdog.is_timed_out
    fallback_mock.assert_not_called()


@pytest.mark.unit
def test_double_signal_handling() -> None:
    """First signal sets interrupt requested; second signal increments count without auto_exit."""
    handler = InterruptionHandler(auto_exit_on_second_signal=False)
    handler.trigger_interrupt(signal.SIGINT)
    assert handler.state == SignalState.INTERRUPT_REQUESTED
    assert handler.signal_count == 1

    handler.trigger_interrupt(signal.SIGINT)
    assert handler.signal_count == 2
    assert handler.interrupted


@pytest.mark.unit
def test_non_main_thread_graceful_degradation() -> None:
    """Running InterruptionHandler inside worker thread degrades gracefully without ValueError."""
    errors: list[Exception] = []

    def worker() -> None:
        try:
            handler = InterruptionHandler()
            with handler:
                assert not handler.interrupted
                handler.trigger_interrupt(signal.SIGINT)
                assert handler.interrupted
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2.0)
    assert len(errors) == 0


@pytest.mark.unit
def test_exception_in_context_restores_signals() -> None:
    """Exceptions inside context manager bubble up but cleanly restore signal handlers."""
    sig = signal.SIGINT
    original = signal.getsignal(sig)
    handler = InterruptionHandler(signals=(sig,))

    with pytest.raises(RuntimeError, match="Intentional failure"):
        with handler:
            raise RuntimeError("Intentional failure")

    assert signal.getsignal(sig) == original
    assert not handler.watchdog.is_running


@pytest.mark.unit
def test_get_default_signals_includes_sigint() -> None:
    """Platform signal detector returns at least SIGINT."""
    sigs = get_default_signals()
    assert signal.SIGINT in sigs
