"""forkreduce: one frozen prefix, many queries, then a semantic reduce.

The framework is built on pydantic-ai. Pass any ``Model``. Nothing in this
package imports a workflow engine; the optional Temporal adapter lives in
``forkreduce.contrib.temporal``.
"""

from __future__ import annotations

from forkreduce.checks import CheckReport, CitationCheck, CitationRepair
from forkreduce.context import ContextEstimateWarning, ContextWarning
from forkreduce.coverage import AcceptUncovered, Provenance, ReduceReport
from forkreduce.documents import END_MARKER, Block, Document
from forkreduce.errors import ForkReduceError, PrefixMismatchWarning, UnregisteredOutputError
from forkreduce.executor import Executor, LocalExecutor
from forkreduce.failures import Failure, FailureKind
from forkreduce.followup import AcceptLeads, GapFill, LeadReport, RoundState, Scope
from forkreduce.models import openai_compatible_model
from forkreduce.plan import Plan, RoundRecord, RunResult
from forkreduce.policy import Accept, Budget, BudgetReport, Drop, Fail, Policy, RequireReason, Retry
from forkreduce.query import CheckFailure, Query, QueryResult
from forkreduce.reduce import ReduceItem, ReduceOutcome
from forkreduce.schemas import DiscardExplanation, DiscardExplanations, Traced
from forkreduce.session import Session, build
from forkreduce.stats import UsageSummary

__version__ = "0.1.0"

__all__ = [
    "END_MARKER",
    "Accept",
    "AcceptLeads",
    "AcceptUncovered",
    "Block",
    "Budget",
    "BudgetReport",
    "CheckFailure",
    "CheckReport",
    "ContextEstimateWarning",
    "ContextWarning",
    "CitationCheck",
    "CitationRepair",
    "DiscardExplanation",
    "DiscardExplanations",
    "Document",
    "Drop",
    "Executor",
    "Fail",
    "Failure",
    "FailureKind",
    "ForkReduceError",
    "GapFill",
    "LeadReport",
    "LocalExecutor",
    "Plan",
    "Policy",
    "PrefixMismatchWarning",
    "Provenance",
    "Query",
    "QueryResult",
    "ReduceItem",
    "ReduceOutcome",
    "ReduceReport",
    "RequireReason",
    "Retry",
    "RoundRecord",
    "RoundState",
    "RunResult",
    "Scope",
    "Session",
    "Traced",
    "UnregisteredOutputError",
    "UsageSummary",
    "build",
    "openai_compatible_model",
]
