"""Serializable per-query JSON Schema outputs, validated without coercion."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import jsonschema  # type: ignore[import-untyped]
from pydantic_ai import ModelRetry
from referencing import Registry

DynamicOutputs = Literal["generic_tool", "prompted"]

DYNAMIC_OUTPUT_TOOL_NAME = "forkreduce_dynamic_output"
DYNAMIC_OUTPUT_TOOL_DESCRIPTION = "Return the JSON object requested by the schema in the query."
DYNAMIC_OUTPUT_TOOL_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


def _json_value(value: Any) -> None:
    """Reject non-JSON schema data instead of silently rewriting it."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("json_schema must contain only JSON values and finite numbers")


def _local_refs(value: Any) -> None:
    """Inspect schema positions, not literal values in const/enum/default."""
    if not isinstance(value, dict):
        return
    for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
        reference = value.get(keyword)
        if isinstance(reference, str) and reference and not reference.startswith("#"):
            raise ValueError(f"OutputSpec only supports local references, not {reference!r}")
    for keyword in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies"):
        children = value.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                _local_refs(child)
    for keyword in (
        "additionalProperties",
        "unevaluatedProperties",
        "additionalItems",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    ):
        _local_refs(value.get(keyword))
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems", "items"):
        children = value.get(keyword)
        if isinstance(children, list):
            for child in children:
                _local_refs(child)
        else:
            _local_refs(children)


@dataclass(frozen=True)
class OutputSpec:
    """A copied JSON Schema for one query, with a plain ``dict`` result.

    Both strict modes validate the complete schema without coercion. ``strict``
    additionally checks registered JSON Schema formats and rejects non-finite
    JSON numbers. References must be local fragments; resolution never fetches
    resources. Only these four public, serializable fields are retained.
    """

    name: str
    json_schema: dict[str, Any]
    description: str | None = field(default=None, kw_only=True)
    strict: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OutputSpec.name must be a nonempty string")
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("OutputSpec.description must be a string or None")
        if not isinstance(self.strict, bool):
            raise ValueError("OutputSpec.strict must be a bool")
        if not isinstance(self.json_schema, dict):
            raise ValueError("OutputSpec.json_schema must be a dict")
        try:
            json.dumps(self.json_schema, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OutputSpec.json_schema must be serializable JSON: {exc}") from exc
        _json_value(self.json_schema)
        _local_refs(self.json_schema)
        schema = copy.deepcopy(self.json_schema)
        try:
            jsonschema.validators.validator_for(schema).check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ValueError(f"Invalid OutputSpec JSON schema: {exc.message}") from exc
        object.__setattr__(self, "json_schema", schema)

    def validate_output(self, output: Any) -> dict[str, Any]:
        """Validate a parsed object, raising native feedback for a bad response."""
        if not isinstance(output, dict):
            raise ModelRetry(f"Output {self.name!r} must be a JSON object")
        if self.strict:
            try:
                json.dumps(output, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ModelRetry(f"Output {self.name!r} must contain valid JSON numbers and values: {exc}") from exc
        # Compile locally, never on the serializable specification. The schema
        # is intentionally not handed to StructuredDict, which rewrites $defs.
        _local_refs(self.json_schema)
        validator_type = jsonschema.validators.validator_for(self.json_schema)
        validator = validator_type(
            self.json_schema,
            format_checker=jsonschema.FormatChecker() if self.strict else None,
            registry=Registry(),  # The empty registry refuses retrieval by default.
        )
        try:
            validator.validate(output)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path) or "(root)"
            raise ModelRetry(f"Output {self.name!r} at {path}: {exc.message}") from exc
        return output
