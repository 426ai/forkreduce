"""forkreduce: one frozen prefix, many queries, then a semantic reduce.

The framework is built on pydantic-ai. Pass any ``Model``. Nothing in this
package imports a workflow engine; the optional Temporal adapter lives in
``forkreduce.contrib.temporal``.
"""

from __future__ import annotations

from forkreduce.checks import CheckReport, CitationCheck, CitationRepair
from forkreduce.context import ContextEstimateWarning, ContextWarning
from forkreduce.coverage import AcceptUncovered, Provenance, ReduceReport
from forkreduce.documents import END_MARKER, Block, Document, DocumentFormat
from forkreduce.errors import (
    ForkReduceError,
    OutputModeConflictError,
    PrefixMismatchWarning,
    UnforcedToolsError,
    UnforcedToolsWarning,
    UnregisteredOutputError,
)
from forkreduce.executor import Executor, LocalExecutor
from forkreduce.failures import Failure, FailureKind
from forkreduce.followup import AcceptLeads, GapFill, LeadReport, RoundState, Scope
from forkreduce.models import openai_compatible_model
from forkreduce.output_modes import OutputMode, ResolvedOutputMode
from forkreduce.output_specs import DynamicOutputs, OutputSpec
from forkreduce.plan import Plan, RoundRecord, RunResult
from forkreduce.policy import Accept, Budget, BudgetReport, Drop, Fail, Policy, RequireReason, Retry
from forkreduce.prefix import QUERY_SEPARATOR, MessageLayout
from forkreduce.query import CheckFailure, OutputValidationContext, Query, QueryResult
from forkreduce.reduce import ReduceItem, ReduceOutcome
from forkreduce.schemas import CITATION_INSTRUCTIONS, Citation, DiscardExplanation, DiscardExplanations, Traced
from forkreduce.session import OutputValidator, Session, build
from forkreduce.stats import UsageSummary
from forkreduce.temporal_options import ActivityOptions

__version__ = "0.2.0"

__all__ = [
    "END_MARKER",
    "QUERY_SEPARATOR",
    "CITATION_INSTRUCTIONS",
    "Accept",
    "AcceptLeads",
    "AcceptUncovered",
    "ActivityOptions",
    "Block",
    "Budget",
    "BudgetReport",
    "CheckFailure",
    "CheckReport",
    "Citation",
    "ContextEstimateWarning",
    "ContextWarning",
    "CitationCheck",
    "CitationRepair",
    "DiscardExplanation",
    "DiscardExplanations",
    "Document",
    "DocumentFormat",
    "Drop",
    "DynamicOutputs",
    "Executor",
    "Fail",
    "Failure",
    "FailureKind",
    "ForkReduceError",
    "GapFill",
    "LeadReport",
    "LocalExecutor",
    "MessageLayout",
    "OutputMode",
    "OutputModeConflictError",
    "OutputSpec",
    "OutputValidationContext",
    "OutputValidator",
    "Plan",
    "Policy",
    "PrefixMismatchWarning",
    "Provenance",
    "Query",
    "QueryResult",
    "ReduceItem",
    "ReduceOutcome",
    "ReduceReport",
    "ResolvedOutputMode",
    "RequireReason",
    "Retry",
    "RoundRecord",
    "RoundState",
    "RunResult",
    "Scope",
    "Session",
    "Traced",
    "UnforcedToolsError",
    "UnforcedToolsWarning",
    "UnregisteredOutputError",
    "UsageSummary",
    "build",
    "openai_compatible_model",
]
