"""POSIX signal trapping and 30-second emergency watchdog for safe termination."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class GracefulInterruptionHandler:
    """Traps SIGINT/SIGTERM to allow finishing current optimizer step and saving checkpoint."""

    def __init__(self, watchdog_timeout_seconds: float = 30.0) -> None:
        self.interrupted: bool = False
        self.signal_received: int | None = None
        self.watchdog_timeout = watchdog_timeout_seconds
        self._watchdog_thread: threading.Thread | None = None
        self._orig_sigint: Any = None
        self._orig_sigterm: Any = None

    def __enter__(self) -> GracefulInterruptionHandler:
        self.interrupted = False
        self.signal_received = None
        self._orig_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._orig_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._orig_sigint is not None:
            signal.signal(signal.SIGINT, self._orig_sigint)
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._orig_sigterm)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning(f"Received {sig_name}. Requesting graceful shutdown at step boundary.")
        self.interrupted = True
        self.signal_received = signum

        # Arm emergency watchdog if not already armed
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_target,
                daemon=True,
                name="HeliosEmergencyWatchdog",
            )
            self._watchdog_thread.start()

    def _watchdog_target(self) -> None:
        """Force process exit if graceful termination stalls longer than watchdog timeout."""
        time.sleep(self.watchdog_timeout)
        logger.critical(
            f"Emergency watchdog timer ({self.watchdog_timeout}s) expired during graceful shutdown. "
            f"Forcing immediate exit."
        )
        os._exit(130)
