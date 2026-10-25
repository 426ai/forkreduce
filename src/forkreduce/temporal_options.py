"""Workflow activity settings without importing the optional Temporal SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityOptions:
    """Temporal-only activity timeouts and retries, with durations in seconds.

    These settings do not affect local execution or the frozen prefix. The
    default start-to-close timeout is five minutes. Retry fields left as
    ``None`` inherit the session's :class:`~forkreduce.policy.Policy` settings;
    ``backoff_coefficient`` defaults to exponential backoff with a factor of two.
    All values are JSON-native, so workflow inputs need no SDK-specific types.
    """

    start_to_close: float = 300
    heartbeat_timeout: float | None = None
    schedule_to_close: float | None = None
    maximum_attempts: int | None = None
    retry_backoff: float | None = None
    max_backoff: float | None = None
    backoff_coefficient: float = 2.0

    def __post_init__(self) -> None:
        for name in ("start_to_close", "heartbeat_timeout", "schedule_to_close"):
            value = getattr(self, name)
            if value is None and name != "start_to_close":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0 seconds")
        if self.maximum_attempts is not None and (
            isinstance(self.maximum_attempts, bool)
            or not isinstance(self.maximum_attempts, int)
            or self.maximum_attempts < 1
        ):
            raise ValueError("maximum_attempts must be an integer >= 1")
        for name in ("retry_backoff", "max_backoff"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and >= 0 seconds")
        if (
            isinstance(self.backoff_coefficient, bool)
            or not isinstance(self.backoff_coefficient, (int, float))
            or not math.isfinite(self.backoff_coefficient)
            or self.backoff_coefficient < 1
        ):
            raise ValueError("backoff_coefficient must be finite and >= 1")
