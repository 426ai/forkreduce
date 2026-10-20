"""Text cleanup applied before a document is frozen into the shared prefix."""

from __future__ import annotations

import re

# C0 controls and DEL. Tab, newline, and carriage return are kept so ordinary
# paragraphs still render. JSON tool arguments reject raw controls, and models
# sometimes copy them out of the source text.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def strip_control_chars(text: str) -> str:
    """Remove ASCII control characters other than tab and newlines."""
    return _CONTROL_CHARS.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()
