"""Profile-driven output mode selection, without provider calls or name heuristics."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Literal

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, merge_model_settings

from forkreduce.errors import ForkReduceError, UnforcedToolsError, UnforcedToolsWarning

OutputMode = Literal["auto", "tools", "prompted"]
ResolvedOutputMode = Literal["tools", "prompted"]
UnforcedToolsPolicy = Literal["warn", "error", "ignore"]


def effective_model_settings(
    model: Model,
    session_settings: ModelSettings | None,
    query_settings: ModelSettings | None = None,
) -> ModelSettings | None:
    """Match the SDK's shallow model -> session -> query settings precedence."""
    return merge_model_settings(model.settings, merge_model_settings(session_settings, query_settings))


def _thinking_active(profile: Mapping[str, object], settings: ModelSettings | None) -> bool | None:
    """Return None for controls whose effective thinking cannot be resolved safely."""
    values: Mapping[str, object] = settings or {}
    overrides: list[bool] = []
    unknown = False
    for key, value in values.items():
        if value is None:
            continue
        if key == "openai_reasoning_effort":
            if isinstance(value, str) and value in ("none", "minimal", "low", "medium", "high", "xhigh"):
                overrides.append(value != "none")
            else:
                unknown = True
        elif key == "anthropic_thinking":
            if isinstance(value, Mapping) and value.get("type") in ("enabled", "adaptive", "disabled"):
                overrides.append(value["type"] != "disabled")
            else:
                unknown = True
        elif key == "google_thinking_config":
            if not isinstance(value, Mapping):
                unknown = True
                continue
            for name, control in value.items():
                if control is None or name == "include_thoughts":
                    continue
                if name == "thinking_budget" and isinstance(control, int) and not isinstance(control, bool):
                    overrides.append(control != 0)
                elif (
                    name == "thinking_level"
                    and isinstance(control, str)
                    and control.lower() in ("minimal", "low", "medium", "high")
                ):
                    # Minimal thinking is still thinking, not disabled.
                    overrides.append(True)
                else:
                    unknown = True
        elif key in ("extra_body", "chat_template_kwargs"):
            if not isinstance(value, Mapping):
                unknown = True
                continue
            for name, control in value.items():
                if control is None:
                    continue
                if (
                    name == "thinking"
                    and isinstance(control, Mapping)
                    and control.get("type") in ("enabled", "adaptive", "disabled")
                ):
                    overrides.append(control["type"] != "disabled")
                elif (
                    name == "reasoning_effort"
                    and isinstance(control, str)
                    and control in ("none", "minimal", "low", "medium", "high", "xhigh")
                ):
                    overrides.append(control != "none")
                elif name == "enable_thinking" and isinstance(control, bool):
                    overrides.append(control)
                elif name == "chat_template_kwargs" and isinstance(control, Mapping):
                    for option, enabled in control.items():
                        if option == "enable_thinking" and isinstance(enabled, bool):
                            overrides.append(enabled)
                        elif "thinking" in option or "reasoning" in option:
                            unknown = True
                elif "thinking" in name or "reasoning" in name:
                    unknown = True
        elif (
            key != "thinking"
            and ("thinking" in key or "reasoning" in key)
            and key not in ("openai_reasoning_summary", "openai_reasoning_generate_summary")
        ):
            unknown = True
    if unknown:
        return None
    if overrides:
        # Provider controls override unified thinking. Contradictory controls
        # have no portable precedence, so any enabled control means active.
        return any(overrides)
    thinking = values.get("thinking")
    if profile.get("supports_thinking", False) and thinking is not None:
        return thinking is not False
    return bool(profile.get("thinking_enabled_by_default", False))


def forced_tools_available(model: Model, settings: ModelSettings | None, *, strict: bool = False) -> bool:
    """Whether forced output tools preserve the effective request's thinking.

    Only profiles where thinking affects forcing need their controls resolved.
    Auto rejects unknown controls in that conditional case; explicit tools
    treat them as unavailable and apply their diagnostics policy.
    """
    profile = model.profile
    if not profile.get("supports_tools", True) or not profile.get("supports_forced_tool_choice", True):
        return False
    if profile.get("supports_forced_tool_choice_with_thinking", True) and not profile.get(
        "forced_tool_choice_disables_thinking", False
    ):
        return True
    if profile.get("thinking_always_enabled", False):
        return False
    thinking = _thinking_active(profile, settings)
    if thinking is None:
        if strict:
            raise ForkReduceError(
                "Cannot resolve output_mode='auto': an explicit thinking/reasoning control is not "
                "recognized. Use supported thinking settings or choose an explicit output_mode.",
                code="unknown_thinking_control",
            )
        return False
    return not thinking


def resolve_output_mode(model: Model, settings: ModelSettings | None) -> ResolvedOutputMode:
    """Resolve auto to one of forkreduce's two explicit SDK output modes."""
    return "tools" if forced_tools_available(model, settings, strict=True) else "prompted"


def diagnose_unforced_tools(policy: UnforcedToolsPolicy) -> None:
    """Apply the explicit-tools policy without changing the requested mode."""
    message = (
        "output_mode='tools' was requested, but the model profile and effective settings "
        "do not support forced output tools without disabling thinking. "
        "Use output_mode='auto' or 'prompted', or adjust the profile/thinking settings. "
        "Explicit tools mode is not automatically changed."
    )
    if policy == "error":
        raise UnforcedToolsError(message, code="unforced_tools")
    if policy == "warn":
        warnings.warn(message, UnforcedToolsWarning, stacklevel=3)
