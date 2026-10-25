"""Pinned, checksum-verified CUAD v1 SQuAD data and deterministic workload selection.

The official archive is CC BY 4.0: https://www.atticusprojectai.org/cuad .
Only the public data download uses the network; cached loads and selection are offline.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.schema import Contract, Question, Span

CUAD_REVISION = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
CUAD_URL = f"https://raw.githubusercontent.com/The-Atticus-Project/cuad/{CUAD_REVISION}/data.zip"
CUAD_ARCHIVE_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
CUAD_JSON_SHA256 = "ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae"
CUAD_MEMBER = "CUADv1.json"
DEFAULT_SEED = 20261024
CATEGORIES = (
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Agreement Date",
    "Anti-Assignment",
    "Audit Rights",
    "Cap On Liability",
    "Change Of Control",
    "Competitive Restriction Exception",
    "Covenant Not To Sue",
    "Document Name",
    "Effective Date",
    "Exclusivity",
    "Expiration Date",
    "Governing Law",
    "Insurance",
    "Ip Ownership Assignment",
    "Irrevocable Or Perpetual License",
    "Joint Ip Ownership",
    "License Grant",
    "Liquidated Damages",
    "Minimum Commitment",
    "Most Favored Nation",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Compete",
    "Non-Disparagement",
    "Non-Transferable License",
    "Notice Period To Terminate Renewal",
    "Parties",
    "Post-Termination Services",
    "Price Restrictions",
    "Renewal Term",
    "Revenue/Profit Sharing",
    "Rofr/Rofo/Rofn",
    "Source Code Escrow",
    "Termination For Convenience",
    "Third Party Beneficiary",
    "Uncapped Liability",
    "Unlimited/All-You-Can-Eat-License",
    "Volume Restriction",
    "Warranty Duration",
)


def _verified(raw: bytes, expected: str, label: str) -> bytes:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"CUAD {label} SHA256 mismatch: expected {expected}, got {actual}")
    return raw


def parse_contracts(payload: dict[str, Any]) -> list[Contract]:
    """Aggregate repeated contract/category records, preserving every distinct gold span.

    Repeated paragraph contexts share the same character offsets. Distinct contexts
    are joined with two newlines and their local SQuAD offsets are shifted accordingly.
    Question ids are canonical category names, not contract-specific SQuAD ids.
    """
    records: dict[str, dict[str, Any]] = {}
    canonical_text: dict[str, str] = {}
    for document in payload["data"]:
        contract_id = document["title"]
        record = records.setdefault(contract_id, {"contexts": {}, "parts": [], "length": 0, "questions": {}})
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            if context not in record["contexts"]:
                separator = "\n\n" if record["parts"] else ""
                offset = record["length"] + len(separator)
                record["parts"].append(separator + context)
                record["length"] = offset + len(context)
                record["contexts"][context] = offset
            offset = record["contexts"][context]
            for qa in paragraph["qas"]:
                category = qa["id"].rsplit("__", 1)[-1]
                text = qa["question"]
                previous = canonical_text.setdefault(category, text)
                if previous != text:
                    raise ValueError(f"Inconsistent canonical question for {category!r}")
                spans = record["questions"].setdefault(category, {})
                for answer in qa.get("answers", []):
                    start, quote = answer["answer_start"], answer["text"]
                    if start < 0 or context[start : start + len(quote)] != quote:
                        raise ValueError(f"Invalid gold span in {contract_id!r}, {category!r}")
                    span = Span(text=quote, start=offset + start)
                    spans[(span.start, span.text)] = span
    return [
        Contract(
            id=contract_id,
            text="".join(record["parts"]),
            questions=tuple(
                Question(
                    id=category, text=canonical_text[category], answers=tuple(span for _, span in sorted(spans.items()))
                )
                for category, spans in sorted(record["questions"].items())
            ),
        )
        for contract_id, record in sorted(records.items())
    ]


def load_contracts(cache: Path) -> list[Contract]:
    """Download the official pinned archive on a cache miss; verify every cached load."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    json_path = cache / CUAD_MEMBER
    if json_path.exists():
        raw = _verified(json_path.read_bytes(), CUAD_JSON_SHA256, CUAD_MEMBER)
    else:
        archive_path = cache / "data.zip"
        if not archive_path.exists():
            temporary: Path | None = None
            try:
                with urllib.request.urlopen(CUAD_URL, timeout=120) as response:
                    archive_raw = _verified(response.read(), CUAD_ARCHIVE_SHA256, "archive")
                with tempfile.NamedTemporaryFile(dir=cache, prefix="cuad-", suffix=".part", delete=False) as target:
                    temporary = Path(target.name)
                    target.write(archive_raw)
                temporary.replace(archive_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        _verified(archive_path.read_bytes(), CUAD_ARCHIVE_SHA256, "archive")
        with zipfile.ZipFile(archive_path) as archive:
            raw = _verified(archive.read(CUAD_MEMBER), CUAD_JSON_SHA256, CUAD_MEMBER)
        json_path.write_bytes(raw)
    contracts = parse_contracts(json.loads(raw))
    if len(contracts) != 510 or any(tuple(q.id for q in c.questions) != CATEGORIES for c in contracts):
        raise ValueError("Pinned CUAD must contain 510 contracts with all 41 canonical categories")
    return contracts


def sample_contracts(
    contracts: Sequence[Contract],
    count: int = 20,
    seed: int = DEFAULT_SEED,
) -> list[Contract]:
    """Sample sorted ids, then return sampled contracts in sorted id order."""
    ordered = sorted(contracts, key=lambda contract: contract.id)
    if len({contract.id for contract in ordered}) != len(ordered):
        raise ValueError("Contract ids must be unique")
    if not 0 <= count <= len(ordered):
        raise ValueError("Sample count must be between zero and the contract count")
    return sorted(random.Random(seed).sample(ordered, count), key=lambda contract: contract.id)


def select_questions(contract: Contract, n: int, seed: int = DEFAULT_SEED) -> tuple[Question, ...]:
    """Use one category permutation for every contract, with nested N=5/20/41 prefixes."""
    ordered = sorted(contract.questions, key=lambda question: question.id)
    if len({question.id for question in ordered}) != len(ordered):
        raise ValueError("Question category ids must be unique within a contract")
    if not 0 <= n <= len(ordered):
        raise ValueError("Question count must be between zero and the category count")
    random.Random(seed).shuffle(ordered)
    return tuple(ordered[:n])
