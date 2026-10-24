"""Core-only validation and prefix identity for Temporal activity settings."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from forkreduce import ActivityOptions, build


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize("name", ["start_to_close", "heartbeat_timeout", "schedule_to_close"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), True, "30"])
def test_activity_timeouts_must_be_finite_positive_seconds(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        ActivityOptions(**{name: value})


def test_start_to_close_is_required_even_when_other_timeouts_are_optional() -> None:
    with pytest.raises(ValueError, match="start_to_close"):
        ActivityOptions(start_to_close=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), float("inf"), True, "2"])
def test_maximum_attempts_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="maximum_attempts"):
        ActivityOptions(maximum_attempts=value)


@pytest.mark.parametrize("name", ["retry_backoff", "max_backoff"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_backoffs_must_be_finite_nonnegative_seconds(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        ActivityOptions(**{name: value})


@pytest.mark.parametrize("value", [0, 0.5, -1, float("nan"), float("inf"), -float("inf"), True, None])
def test_backoff_coefficient_must_be_finite_and_at_least_one(value: Any) -> None:
    with pytest.raises(ValueError, match="backoff_coefficient"):
        ActivityOptions(backoff_coefficient=value)


def test_valid_activity_options_allow_fractional_seconds_and_disabled_backoff() -> None:
    options = ActivityOptions(
        start_to_close=0.1,
        heartbeat_timeout=0.01,
        schedule_to_close=0.5,
        maximum_attempts=1,
        retry_backoff=0,
        max_backoff=0,
        backoff_coefficient=1,
    )
    assert ActivityOptions(**json.loads(json.dumps(asdict(options), allow_nan=False))) == options


@pytest.mark.parametrize("output_mode", ["tools", "prompted"])
@pytest.mark.parametrize(
    "options",
    [
        ActivityOptions(),
        ActivityOptions(
            start_to_close=10,
            heartbeat_timeout=1,
            schedule_to_close=90,
            maximum_attempts=1,
            retry_backoff=0,
            max_backoff=0,
            backoff_coefficient=1,
        ),
        ActivityOptions(maximum_attempts=9, retry_backoff=2, max_backoff=30),
    ],
)
def test_temporal_options_do_not_change_prefix_identity(output_mode: Any, options: ActivityOptions) -> None:
    baseline = build(
        TestModel(),
        documents="shared document",
        instructions="Use only the document.",
        outputs=[Answer],
        output_mode=output_mode,
        warmup=False,
    )
    configured = build(
        TestModel(),
        documents="shared document",
        instructions="Use only the document.",
        outputs=[Answer],
        output_mode=output_mode,
        warmup=False,
        activity_options=options,
    )
    assert configured.prefix_hash == baseline.prefix_hash
