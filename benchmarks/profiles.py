"""Provider profiles and historical output-mode decisions for benchmark models.

The benchmark resolver is deliberately retained rather than delegated to the
framework: only ``thinking`` and ``openai_reasoning_effort`` affect its decision.
The framework additionally understands Anthropic/Google thinking controls and
recognized reasoning controls in ``extra_body``/``chat_template_kwargs``; it
rejects unknown conditional controls instead of treating them as defaults. It
also honors ``supports_tools``, gates unified thinking on ``supports_thinking``,
and treats always-enabled thinking as unavailable for conditional tool forcing.
``output_mode_comparison`` records agreement for the actual model and effective
settings without changing any measured method's explicit output-mode override.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic_ai.models import Model
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from forkreduce.errors import ForkReduceError
from forkreduce.output_modes import effective_model_settings, resolve_output_mode


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


def output_mode_comparison(model: Model, build_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare both resolvers against identical actual profile and shallow settings.

    Model settings are shallow-merged with build/agent settings using the same
    precedence as framework sessions. Unknown conditional thinking controls are
    reported by their framework error code, not hidden by a benchmark fallback.
    This is diagnostic metadata only: explicit method output modes still win.
    Settings are not returned because they may contain provider credentials.
    """
    settings = effective_model_settings(model, cast(ModelSettings | None, build_settings))
    decision = output_mode_decision(model.profile, settings)
    framework_mode = None
    error_code = None
    try:
        framework_mode = resolve_output_mode(model, settings)
    except ForkReduceError as exc:
        if exc.code != "unknown_thinking_control":
            raise
        error_code = exc.code
    return {
        **decision,
        "framework_output_mode": framework_mode,
        "output_modes_equivalent": framework_mode == decision["profile_output_mode"],
        "framework_error_code": error_code,
    }
