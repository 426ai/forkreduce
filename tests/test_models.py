"""OpenAI-compatible model settings and per-call overrides."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import BaseModel
from tests.support import Scripted, tool_result

from forkreduce import Query, build, openai_compatible_model


class Answer(BaseModel):
    """A short answer."""

    text: str


def test_openai_compatible_model_accepts_profile_timeout_and_extra_body() -> None:
    pytest.importorskip("openai")
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


def test_core_without_openai_and_actionable_provider_error() -> None:
    script = """
import asyncio
import importlib.abc
import sys

class NoOpenAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openai" or fullname.startswith("openai."):
            raise ModuleNotFoundError("OpenAI deliberately unavailable")

sys.meta_path.insert(0, NoOpenAI())
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from forkreduce import Query, build, openai_compatible_model

class Answer(BaseModel):
    text: str

async def main():
    session = build(TestModel(custom_output_args={"text": "offline"}),
                    documents="Synthetic evidence", instructions="Answer", outputs=[Answer], warmup=False)
    result = await session.ask(Query(tail="Question", output=Answer))
    assert result.output.text == "offline"
    try:
        openai_compatible_model("synthetic-model")
    except ImportError as error:
        assert "forkreduce[openai]" in str(error)
    else:
        raise AssertionError("Missing SDK must produce an installation hint")

asyncio.run(main())
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
