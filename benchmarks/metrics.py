"""Offline CUAD scoring and conservative aggregation of observed provider usage.

Gold spans are cumulative positives, not interchangeable reference answers. Word
scores use SQuAD/CUAD normalization and multiset overlap across every span. Clause
scores use maximum one-to-one matching at normalized word F1 >= 0.5. Citation
coverage uses the union of *all* literal occurrences, never a favorable occurrence.
"""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from benchmarks.schema import Answers, Contract, Question


def normalize(text: str) -> str:
    """Apply CUAD/SQuAD lowercasing, punctuation and article normalization."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _prf(overlap: int, predicted: int, gold: int) -> tuple[float, float, float]:
    if predicted == gold == 0:
        return 1.0, 1.0, 1.0
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / gold if gold else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _word_counts(texts: Iterable[str]) -> Counter[str]:
    return Counter(token for text in texts for token in normalize(text).split())


def _word_prf(predicted: Sequence[str], gold: Sequence[str]) -> tuple[float, float, float]:
    p, g = _word_counts(predicted), _word_counts(gold)
    return _prf(sum((p & g).values()), sum(p.values()), sum(g.values()))


def _matches(predicted: Sequence[str], gold: Sequence[str], threshold: float) -> int:
    # Augmenting paths give maximum cardinality rather than order-dependent greedy matches.
    edges = [
        [j for j, target in enumerate(gold) if _word_prf([quote], [target])[2] >= threshold] for quote in predicted
    ]
    owners: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j], seen):
                owners[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def _union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _length(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    i = j = overlap = 0
    while i < len(left) and j < len(right):
        overlap += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def _occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    if not quote:
        return result
    start = text.find(quote)
    while start >= 0:
        result.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return result


def _mean(values: Iterable[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def score(contract: Contract, questions: Sequence[Question], answers: Answers | Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe macro scores plus detailed, position-aware question records.

    A missing or duplicate answer is an invalid answer, including on absence
    questions: it receives zero rather than credit for an implicit abstention.
    Annotation validity and exact citation location are intentionally independent
    from normalized semantic word scores. Invalid/nonliteral gold offsets make
    character metrics unknown, never fabricated; word and clause scores retain
    every gold text. sequence_position is the requested one-based question order,
    including missing answers, distinct from the extra gold document position.
    """
    ids = [question.id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Requested question ids must be unique")
    raw = answers.model_dump() if isinstance(answers, Answers) else answers
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for answer in raw.get("answers", []):
        grouped[answer["question_id"]].append(answer)
    details: list[dict[str, Any]] = []
    for sequence_position, question in enumerate(questions, 1):
        supplied = grouped.get(question.id, [])
        valid = len(supplied) == 1
        quotes = [citation["quote"] for answer in supplied for citation in answer.get("spans", [])]
        gold = [span.text for span in question.answers]
        word = (0.0, 0.0, 0.0) if quotes and not gold else _word_prf(quotes, gold)
        matched = _matches(quotes, gold, 0.5)
        clause = _prf(matched, len(quotes), len(gold))
        exact = float(Counter(map(normalize, quotes)) == Counter(map(normalize, gold)))
        locations = []
        nonliteral = omitted = literal = 0
        invalid_chars = 0
        intervals: list[tuple[int, int]] = []
        for quote in quotes:
            annotated = "<omitted>" in quote.lower()
            found = [] if annotated else _occurrences(contract.text, quote)
            locations.append(
                {
                    "quote": quote,
                    "occurrences": [[a, b] for a, b in found],
                    "literal": bool(found),
                    "omitted_annotation": annotated,
                }
            )
            omitted += int(annotated)
            if found:
                literal += 1
                intervals.extend(found)
            else:
                nonliteral += 1
                invalid_chars += max(1, len(quote))
        predicted_intervals = _union(intervals)
        gold_intervals_list = []
        gold_nonliteral = gold_omitted = gold_out_of_bounds = 0
        for span in question.answers:
            end = span.start + len(span.text)
            bounded = 0 <= span.start < end <= len(contract.text)
            annotated = "<omitted>" in span.text.lower()
            literal_gold = bounded and not annotated and contract.text[span.start : end] == span.text
            gold_out_of_bounds += int(not bounded)
            gold_omitted += int(annotated)
            gold_nonliteral += int(not literal_gold)
            if literal_gold:
                gold_intervals_list.append((span.start, end))
        gold_intervals = _union(gold_intervals_list)
        citation_scoring_valid = gold_nonliteral == 0
        citation = (
            _prf(
                _intersection(predicted_intervals, gold_intervals),
                _length(predicted_intervals) + invalid_chars,
                _length(gold_intervals),
            )
            if citation_scoring_valid
            else (None, None, None)
        )
        if not valid:
            word = clause = (0.0, 0.0, 0.0)
            citation = (0.0, 0.0, 0.0) if citation_scoring_valid else (None, None, None)
            exact = 0.0
        position = min((span.start for span in question.answers), default=None)
        relative = position / max(1, len(contract.text)) if position is not None else None
        bucket = (
            "absent" if relative is None else ("start" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "end")
        )
        detail: dict[str, Any] = {
            "question_id": question.id,
            "answerable": bool(gold),
            "valid_answer": valid,
            "sequence_position": sequence_position,
            "missing_answer": not supplied,
            "duplicate_answer": len(supplied) > 1,
            "gold_spans": len(gold),
            "predicted_spans": len(quotes),
            "matched_spans": matched if valid else 0,
            "exact_match": exact,
            "absence_correct": float(valid and not quotes) if not gold else None,
            "literal_citations": literal,
            "nonliteral_citations": nonliteral,
            "omitted_annotations": omitted,
            "gold_nonliteral_spans": gold_nonliteral,
            "gold_omitted_spans": gold_omitted,
            "gold_out_of_bounds": gold_out_of_bounds,
            "citation_scoring_valid": citation_scoring_valid,
            "literal_gold_chars": _length(gold_intervals),
            "citation_exact_match": (literal / len(quotes) if quotes else float(not gold)) if valid else 0.0,
            "gold_position": relative,
            "position_bucket": bucket,
            "citation_locations": locations,
        }
        for prefix, values in [("word", word), ("span", clause), ("citation", citation)]:
            detail.update(
                {f"{prefix}_{name}": value for name, value in zip(("precision", "recall", "f1"), values, strict=True)}
            )
        details.append(detail)
    result: dict[str, Any] = {"questions": len(details), "per_question": details, "span_match_threshold": 0.5}
    for name in (
        "word_precision",
        "word_recall",
        "word_f1",
        "span_precision",
        "span_recall",
        "span_f1",
        "exact_match",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "citation_exact_match",
    ):
        result[name] = _mean(item[name] for item in details)
    result["absence_accuracy"] = _mean(item["absence_correct"] for item in details if not item["answerable"])
    result["answerable_f1"] = _mean(item["word_f1"] for item in details if item["answerable"])
    for name in (
        "literal_citations",
        "nonliteral_citations",
        "omitted_annotations",
        "gold_spans",
        "predicted_spans",
        "matched_spans",
        "missing_answer",
        "duplicate_answer",
    ):
        result[name if name.endswith("s") else name + "s"] = sum(item[name] for item in details)
    for name in ("gold_nonliteral_spans", "gold_omitted_spans", "gold_out_of_bounds"):
        result[name] = sum(item[name] for item in details)
    result["citation_scored_questions"] = sum(item["citation_scoring_valid"] for item in details)
    result["invalid_gold_questions"] = len(details) - result["citation_scored_questions"]
    result["answerable_questions"] = sum(item["answerable"] for item in details)
    result["absence_questions"] = len(details) - result["answerable_questions"]
    result["unexpected_answers"] = sum(len(value) for key, value in grouped.items() if key not in set(ids))
    result["position"] = {
        bucket: {
            "questions": len(items),
            "word_f1": _mean(item["word_f1"] for item in items),
            "citation_f1": _mean(item["citation_f1"] for item in items),
        }
        for bucket in ("start", "middle", "end", "absent")
        if (items := [item for item in details if item["position_bucket"] == bucket])
    }
    return result


def _count(record: Mapping[str, Any], *names: str) -> int | None:
    containers = [record]
    for container in containers:
        for key in ("details", "input_tokens_details", "prompt_tokens_details"):
            if isinstance(container.get(key), Mapping):
                containers.append(container[key])
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and int(value) == value:
                return int(value)
    return None


def summarize_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum observed provider counters; an absent counter stays unknown (None).

    Input is cache-inclusive. Miss, hit and write categories are disjoint, so
    known misses equal input minus hits minus writes. Neither absent cache data
    nor an empty response list is interpreted as a zero-token response.
    """
    records = list(records)
    categories = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "cache_write_tokens",
    )
    observed: dict[str, list[int | None]] = {name: [] for name in categories}
    for record in records:
        input_tokens = _count(record, "input_tokens", "prompt_tokens")
        output_tokens = _count(record, "output_tokens", "completion_tokens")
        hit = _count(
            record,
            "cache_hit_tokens",
            "cache_read_tokens",
            "cache_read_input_tokens",
            "prompt_cache_hit_tokens",
            "cached_tokens",
        )
        write = _count(record, "cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")
        miss = _count(record, "cache_miss_tokens", "prompt_cache_miss_tokens")
        if input_tokens is not None and hit is not None and write is not None and miss is None:
            difference = input_tokens - hit - write
            miss = difference if difference >= 0 else None
        total = _count(record, "total_tokens")
        if total is None and input_tokens is not None and output_tokens is not None:
            total = input_tokens + output_tokens
        values = (input_tokens, output_tokens, total, hit, miss, write)
        for name, value in zip(categories, values, strict=True):
            observed[name].append(value)
    result: dict[str, Any] = {
        name: sum(values) if values and all(value is not None for value in values) else None
        for name, values in observed.items()
    }
    result["observed_subtotals"] = {
        name: sum(value for value in values if value is not None) for name, values in observed.items()
    }
    result["unknown_records"] = {name: sum(value is None for value in values) for name, values in observed.items()}
    result["records"] = len(records)
    result["usage_missing"] = (
        not records
        or any(record.get("usage_missing", False) for record in records)
        or any(result[name] is None for name in ("input_tokens", "output_tokens", "total_tokens"))
    )
    result["cache_hit_rate"] = (
        result["cache_hit_tokens"] / result["input_tokens"]
        if result["cache_hit_tokens"] is not None and result["input_tokens"]
        else None
    )
    return result
