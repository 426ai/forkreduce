"""Provider capabilities for benchmark models routed through an OpenAI proxy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider


def model_profile(name: str) -> OpenAIModelProfile:
    """Resolve capabilities explicitly: a proxy base URL does not select a provider profile."""
    base = OpenAIProvider.model_profile(name)
    if name == "mimo-v2.6-pro":
        # https://mimo.mi.com/docs/api/chat/openai-api
        # https://mimo.mi.com/docs/en-US/api/chat/responses
        # Only auto tool_choice is honored; thinking is enabled by default.
        # https://mimo.mi.com/docs/usage-guide/passing-back-reasoning_content
        # Preserve reasoning_content across tool turns, rather than rendering it as text.
        return cast(
            OpenAIModelProfile,
            merge_profile(
                base,
                OpenAIModelProfile(
                    supports_thinking=True,
                    thinking_enabled_by_default=True,
                    supports_forced_tool_choice=False,
                    default_structured_output_mode="prompted",
                    openai_chat_thinking_field="reasoning_content",
                    openai_chat_send_back_thinking_parts="field",
                    openai_chat_thinking_requires_field_on_tool_calls=True,
                ),
            ),
        )
    if name == "deepseek-flash" or name.startswith("deepseek-v4-"):
        # The smoke's proxy alias uses the V4 Flash capability profile, without
        # renaming the wire model. Reuse upstream's thinking-aware tool forcing
        # and reasoning_content handling, including its schema transformer.
        upstream = DeepSeekProvider.model_profile("deepseek-v4-flash" if name == "deepseek-flash" else name)
        # One wire-compatibility override: the existing proxy smoke sends
        # max_tokens, not max_completion_tokens. Upstream leaves this at True.
        return cast(
            OpenAIModelProfile,
            merge_profile(base, upstream, OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False)),
        )
    return cast(OpenAIModelProfile, base or OpenAIModelProfile())


def output_mode_decision(profile: ModelProfile, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Mirror the SDK's request-time thinking/forcing rules for benchmark output modes."""
    settings = settings or {}
    effort = settings.get("openai_reasoning_effort")
    thinking = settings.get("thinking")
    if effort is not None:
        active = effort != "none"
    elif thinking is not None:
        active = thinking is not False or profile.get("thinking_always_enabled", False)
    else:
        active = profile.get("thinking_enabled_by_default", False)
    fields = {
        "supports_forced_tool_choice": profile.get("supports_forced_tool_choice", True),
        "supports_forced_tool_choice_with_thinking": profile.get("supports_forced_tool_choice_with_thinking", True),
        "forced_tool_choice_disables_thinking": profile.get("forced_tool_choice_disables_thinking", False),
        "supports_thinking": profile.get("supports_thinking", False),
        "thinking_always_enabled": profile.get("thinking_always_enabled", False),
        "thinking_enabled_by_default": profile.get("thinking_enabled_by_default", False),
        "default_structured_output_mode": profile.get("default_structured_output_mode", "tool"),
    }
    available = fields["supports_forced_tool_choice"] and (
        not active
        or (fields["supports_forced_tool_choice_with_thinking"] and not fields["forced_tool_choice_disables_thinking"])
    )
    return {
        "profile": fields,
        "thinking_active": active,
        "forced_tool_choice_available": bool(available),
        "profile_output_mode": "tools" if available else "prompted",
    }


def profile_output_mode(
    profile: ModelProfile, settings: Mapping[str, Any] | None = None
) -> Literal["tools", "prompted"]:
    return cast(Literal["tools", "prompted"], output_mode_decision(profile, settings)["profile_output_mode"])
