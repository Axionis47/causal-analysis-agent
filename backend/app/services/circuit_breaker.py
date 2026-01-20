"""Simple in-memory circuit breaker for external services."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CircuitState:
    failures: int = 0
    disabled_until: float | None = None


class CircuitBreaker:
    def __init__(
        self,
        threshold: int | None = None,
        timeout_seconds: int | None = None,
        *,
        failure_threshold: int | None = None,
        recovery_timeout_seconds: int | None = None,
    ) -> None:
        resolved_threshold = threshold if threshold is not None else failure_threshold
        resolved_timeout = timeout_seconds if timeout_seconds is not None else recovery_timeout_seconds
        if resolved_threshold is None or resolved_timeout is None:
            raise ValueError("CircuitBreaker requires threshold and timeout values")
        self._threshold = resolved_threshold
        self._timeout_seconds = resolved_timeout
        self._state: dict[str, CircuitState] = {}

    def record_success(self, key: str) -> None:
        old_state = self._state_label(self._state.get(key))
        self._state[key] = CircuitState()
        new_state = self._state_label(self._state.get(key))
        if old_state != new_state:
            logger.info(
                "Circuit breaker state changed",
                breaker_key=key,
                old_state=old_state,
                new_state=new_state,
                failure_count=0,
            )

    def record_failure(self, key: str) -> None:
        state = self._state.get(key, CircuitState())
        old_state = self._state_label(state)
        state.failures += 1
        if state.failures >= self._threshold:
            state.disabled_until = time.time() + self._timeout_seconds
        self._state[key] = state
        new_state = self._state_label(state)
        if old_state != new_state:
            logger.warning(
                "Circuit breaker state changed",
                breaker_key=key,
                old_state=old_state,
                new_state=new_state,
                failure_count=state.failures,
            )

    def is_available(self, key: str) -> bool:
        state = self._state.get(key)
        if state is None or state.disabled_until is None:
            return True
        if time.time() >= state.disabled_until:
            old_state = self._state_label(state)
            self._state[key] = CircuitState()
            new_state = self._state_label(self._state.get(key))
            if old_state != new_state:
                logger.info(
                    "Circuit breaker state changed",
                    breaker_key=key,
                    old_state=old_state,
                    new_state=new_state,
                    failure_count=0,
                )
            return True
        return False

    def retry_after_seconds(self, key: str) -> int | None:
        state = self._state.get(key)
        if state is None or state.disabled_until is None:
            return None
        remaining = int(state.disabled_until - time.time())
        return remaining if remaining > 0 else 0

    @staticmethod
    def _state_label(state: CircuitState | None) -> str:
        if state is None or (state.failures == 0 and state.disabled_until is None):
            return "closed"
        if state.disabled_until is not None and time.time() < state.disabled_until:
            return "open"
        return "closed"
