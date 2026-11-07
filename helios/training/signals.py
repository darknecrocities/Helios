"""Helios Signal Trapping & Microbatch Interruption Engine.

Provides POSIX signal interception (SIGINT, SIGTERM), a thread-safe state machine,
microbatch boundary checking, context manager lifecycle management, and a 30-second
emergency watchdog thread to guarantee crash-resilient shutdown.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger("helios.training.signals")

T = TypeVar("T")


class SignalState(str, Enum):
    """Lifecycle state machine for signal trapping and graceful shutdown."""

    IDLE = "IDLE"
    TRAINING = "TRAINING"
    INTERRUPT_REQUESTED = "INTERRUPT_REQUESTED"
    SAVING = "SAVING"
    SHUTDOWN_CLEAN = "SHUTDOWN_CLEAN"


def get_default_signals() -> tuple[int, ...]:
    """Return default termination signals supported on the current platform."""
    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)
    return tuple(signals)


class EmergencyWatchdog:
    """Background daemon watchdog thread that triggers emergency fallback if saving hangs.

    Prevents ungraceful SIGKILL by external orchestrators (Slurm, Kubernetes, Spot preemption)
    when checkpoint persistence is blocked or deadlocked.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        fallback_fn: Callable[[], None] | None = None,
        name: str = "HeliosWatchdog",
    ) -> None:
        self.timeout = max(0.1, float(timeout))
        self.fallback_fn = fallback_fn
        self.name = name
        self._stop_event = threading.Event()
        self._timed_out = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the watchdog countdown in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._timed_out.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        signaled = self._stop_event.wait(timeout=self.timeout)
        if not signaled:
            self._timed_out.set()
            logger.critical(
                f"[WATCHDOG] Emergency timeout ({self.timeout:.1f}s) exceeded during saving! "
                "Triggering fallback to prevent ungraceful process slaughter."
            )
            if self.fallback_fn is not None:
                try:
                    self.fallback_fn()
                except Exception as exc:
                    logger.error(f"[WATCHDOG] Fallback handler raised exception: {exc}")

    def cancel(self) -> None:
        """Cancel the countdown and join the watchdog thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def is_timed_out(self) -> bool:
        """Return True if the watchdog reached its timeout without being cancelled."""
        return self._timed_out.is_set()

    @property
    def is_running(self) -> bool:
        """Return True if the watchdog thread is active."""
        return self._thread is not None and self._thread.is_alive()


class InterruptionHandler:
    """Thread-safe POSIX signal trapping context manager for microbatch training loops.

    Traps SIGINT and SIGTERM, transitions an explicit state machine, supports clean
    boundary checks (`should_stop()`), and arms an emergency watchdog during checkpoint
    persistence.
    """

    def __init__(
        self,
        signals: tuple[int, ...] | None = None,
        watchdog_timeout: float = 30.0,
        emergency_fallback_fn: Callable[[], None] | None = None,
        on_interrupt: Callable[[int], None] | None = None,
        auto_exit_on_second_signal: bool = True,
    ) -> None:
        self.signals = signals if signals is not None else get_default_signals()
        self.watchdog_timeout = watchdog_timeout
        self.emergency_fallback_fn = emergency_fallback_fn or self._default_emergency_fallback
        self.on_interrupt = on_interrupt
        self.auto_exit_on_second_signal = auto_exit_on_second_signal

        self._state = SignalState.IDLE
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._received_signal: int | None = None
        self._signal_count = 0
        self._original_handlers: dict[int, Any] = {}
        self._installed = False

        self.watchdog = EmergencyWatchdog(
            timeout=self.watchdog_timeout,
            fallback_fn=self.emergency_fallback_fn,
        )

    @property
    def state(self) -> SignalState:
        """Current lifecycle state in the signal state machine."""
        with self._lock:
            return self._state

    @property
    def interrupted(self) -> bool:
        """Return True if an interruption signal was received."""
        return self._stop_requested.is_set()

    @property
    def received_signal(self) -> int | None:
        """The signal number (e.g. SIGINT, SIGTERM) that initiated the interruption."""
        with self._lock:
            return self._received_signal

    @property
    def signal_count(self) -> int:
        """Number of signals intercepted during the current session."""
        with self._lock:
            return self._signal_count

    def transition_to(self, new_state: SignalState) -> None:
        """Thread-safely advance the state machine to a new state."""
        with self._lock:
            old_state = self._state
            self._state = new_state
            logger.info(f"SignalState transition: {old_state.value} -> {new_state.value}")

    def should_stop(self) -> bool:
        """Clean microbatch boundary polling method.

        Returns True if an interruption has been intercepted, signaling the training
        loop to complete the current microbatch and execute an orderly shutdown.
        """
        return self._stop_requested.is_set()

    def trigger_interrupt(self, signum: int = signal.SIGINT) -> None:
        """Programmatically trigger interruption without waiting for OS signal.

        Useful for unit tests, external orchestration APIs, and RPC calls.
        """
        self._handle_signal(signum, None)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Internal callback invoked when an OS signal or trigger is intercepted."""
        with self._lock:
            self._signal_count += 1
            current_count = self._signal_count
            self._received_signal = signum
            self._stop_requested.set()

            if current_count == 1:
                sig_name = (
                    signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
                )
                logger.warning(
                    f"Intercepted signal {sig_name} ({signum}). Requesting microbatch shutdown..."
                )
                self.transition_to(SignalState.INTERRUPT_REQUESTED)
                if self.on_interrupt is not None:
                    try:
                        self.on_interrupt(signum)
                    except Exception as exc:
                        logger.error(f"Error in on_interrupt callback: {exc}")
            else:
                logger.critical(
                    f"Repeated signal {signum} intercepted ({current_count} times)! "
                    "Forcing immediate exit."
                )
                if self.auto_exit_on_second_signal:
                    self.restore()
                    os._exit(128 + signum)

    def register(self) -> None:
        """Attach signal handlers to the main interpreter thread."""
        with self._lock:
            if self._installed:
                return

            if threading.current_thread() is not threading.main_thread():
                logger.debug("InterruptionHandler not in main thread; OS signals skipped.")
                return

            for sig in self.signals:
                try:
                    self._original_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._handle_signal)
                except (ValueError, OSError, AttributeError) as exc:
                    logger.warning(f"Could not register handler for signal {sig}: {exc}")

            self._installed = True
            if self._state == SignalState.IDLE:
                self.transition_to(SignalState.TRAINING)

    def restore(self) -> None:
        """Restore all original OS signal handlers and cancel watchdog."""
        with self._lock:
            self.watchdog.cancel()

            if not self._installed:
                return

            if threading.current_thread() is threading.main_thread():
                for sig, old_handler in self._original_handlers.items():
                    try:
                        signal.signal(sig, old_handler)
                    except (ValueError, OSError, AttributeError):
                        pass

            self._original_handlers.clear()
            self._installed = False
            if self._state not in (SignalState.SHUTDOWN_CLEAN, SignalState.SAVING):
                self.transition_to(SignalState.IDLE)

    def start_watchdog(self, timeout: float | None = None) -> None:
        """Arm the emergency watchdog before initiating checkpoint disk persistence."""
        if timeout is not None:
            self.watchdog.timeout = max(0.1, float(timeout))
        self.watchdog.start()

    def stop_watchdog(self) -> None:
        """Disarm the watchdog after persistence completes successfully."""
        self.watchdog.cancel()

    def execute_with_watchdog(self, action: Callable[[], T]) -> T:
        """Execute a checkpoint persistence action wrapped with the watchdog timer."""
        self.transition_to(SignalState.SAVING)
        self.start_watchdog()
        try:
            result = action()
            self.transition_to(SignalState.SHUTDOWN_CLEAN)
            return result
        finally:
            self.stop_watchdog()

    def _default_emergency_fallback(self) -> None:
        """Default emergency callback if checkpoint persistence exceeds the watchdog timeout."""
        sys.stderr.write(
            f"\n[CRITICAL HELIOS EMERGENCY] Checkpoint persistence hung for "
            f">{self.watchdog_timeout}s! Terminating process cleanly before external SIGKILL.\n"
        )
        sys.stderr.flush()
        os._exit(128 + (self._received_signal or signal.SIGTERM))

    def __enter__(self) -> InterruptionHandler:
        self.register()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.restore()
