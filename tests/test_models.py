"""OpenAI-compatible model settings and per-call overrides."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from tests.support import Scripted, tool_result

from forkreduce import Query, build, openai_compatible_model


class Answer(BaseModel):
    """A short answer."""

    text: str


def test_openai_compatible_model_accepts_profile_timeout_and_extra_body() -> None:
    model = openai_compatible_model(
        "demo-model",
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        timeout=12.5,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        profile={"supports_tools": True},
        settings={"seed": 7, "temperature": 1, "extra_headers": {"X-Test": "1"}},
    )
    assert model.settings is not None
    assert model.settings["timeout"] == 12.5
    assert model.settings["temperature"] == 0
    assert model.settings["seed"] == 7
    assert model.settings["extra_headers"] == {"X-Test": "1"}
    assert model.settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert model.profile["supports_tools"] is True
    assert model.client.timeout == 12.5
    plain = openai_compatible_model("demo-model", api_key="test-key", base_url="http://127.0.0.1:9/v1")
    assert plain.settings is None


@pytest.mark.asyncio
async def test_query_model_settings_override_the_session() -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "Answer", {"text": "ok"})

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        model_settings={"temperature": 0, "extra_body": {"a": 1}, "timeout": 5},
    )
    seen: list[object] = []
    original = session._agent.run

    async def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs.get("model_settings"))
        return await original(*args, **kwargs)

    session._agent.run = spy  # type: ignore[method-assign]
    await session.ask(Query(tail="one", output=Answer, id="q1"))
    await session.ask(
        Query(
            tail="two",
            output=Answer,
            id="q2",
            model_settings={"temperature": 0.7, "extra_body": {"b": 2}},
        )
    )
    first, second = seen
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first["temperature"] == 0
    assert first["extra_body"] == {"a": 1}
    assert first["timeout"] == 5
    assert second["temperature"] == 0.7
    assert second["extra_body"] == {"b": 2}
    assert second["timeout"] == 5
