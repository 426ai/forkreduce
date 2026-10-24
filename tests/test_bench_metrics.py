"""Synthetic offline metric and reporting behavior, without provider calls."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET

import pytest
from benchmarks.metrics import normalize, score, summarize_usage
from benchmarks.reporting import bootstrap_ci, paired_comparison, summarize_rows, write_reports
from benchmarks.schema import Answer, Answers, Citation, Contract, Question, Span


def answers(*quotes: str, question_id: str = "q") -> Answers:
    return Answers(
        answers=[Answer(question_id=question_id, spans=[Citation(quote=quote, block_id="c") for quote in quotes])]
    )


def contract(text: str, spans: tuple[Span, ...] = (), question_id: str = "q") -> Contract:
    return Contract("c", text, (Question(question_id, "Synthetic clause?", spans),))


def row(contract_id: str = "c", method: str = "a", n: int = 1, value: float = 0.5) -> dict:
    return {
        "contract_id": contract_id,
        "method": method,
        "n": n,
        "metrics": {"word_f1": value, "citation_f1": value, "per_question": []},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "cache_hit_tokens": 40,
            "cache_miss_tokens": 50,
            "cache_write_tokens": 10,
            "usage_missing": False,
        },
        "cost_usd": 0.1,
        "elapsed_seconds": 2.5,
        "concurrency": 2,
        "schema_failures": 0,
        "retries": 0,
        "failures": [],
    }


def test_cuad_word_normalization_and_nonliteral_citation_are_separate() -> None:
    data = contract("The quick, Brown fox.", (Span("The quick, Brown fox.", 0),))
    result = score(data, data.questions, answers("quick brown fox"))
    assert normalize(" An EXAMPLE, of the clause! ") == "example of clause"
    assert result["word_f1"] == result["exact_match"] == 1
    assert result["citation_f1"] == 0
    assert result["nonliteral_citations"] == 1


def test_gold_spans_are_all_positives_not_alternative_references() -> None:
    data = contract("red green then blue gold", (Span("red green", 0), Span("blue gold", 15)))
    partial = score(data, data.questions, answers("red green"))
    complete = score(data, data.questions, answers("red green", "blue gold"))
    assert partial["word_precision"] == 1
    assert partial["word_recall"] == 0.5
    assert partial["word_f1"] == pytest.approx(2 / 3)
    assert partial["span_recall"] == 0.5
    assert partial["exact_match"] == 0
    assert complete["word_f1"] == complete["span_f1"] == complete["exact_match"] == 1


def test_duplicate_predicted_span_is_not_matched_twice() -> None:
    data = contract("red green", (Span("red green", 0),))
    result = score(data, data.questions, answers("red green", "red green"))
    assert result["span_precision"] == 0.5
    assert result["span_recall"] == 1
    assert result["word_precision"] == 0.5
    assert result["exact_match"] == 0


def test_matching_is_maximum_one_to_one_not_greedy() -> None:
    data = contract("red green red blue", (Span("red green", 0), Span("red blue", 10)))
    result = score(data, data.questions, answers("red green blue", "green"))
    assert result["matched_spans"] == 2
    assert result["span_f1"] == 1


def test_clause_matching_threshold_is_inclusive() -> None:
    data = contract("red green blue", (Span("red green blue", 0),))
    assert score(data, data.questions, answers("red"))["span_f1"] == 1
    assert score(data, data.questions, answers("red purple"))["span_f1"] == 0


def test_correct_absence_is_one_but_missing_or_duplicate_never_abstention() -> None:
    data = contract("No relevant provision.")
    correct = score(data, data.questions, answers())
    missing = score(data, data.questions, Answers(answers=[]))
    duplicate = score(data, data.questions, Answers(answers=[Answer(question_id="q"), Answer(question_id="q")]))
    assert correct["word_f1"] == correct["span_f1"] == correct["citation_f1"] == correct["absence_accuracy"] == 1
    assert correct["answerable_f1"] is None
    for result in (missing, duplicate):
        assert result["word_f1"] == result["span_f1"] == result["citation_f1"] == result["absence_accuracy"] == 0
        assert result["exact_match"] == result["citation_exact_match"] == 0
    assert missing["missing_answers"] == 1
    assert duplicate["duplicate_answers"] == 1
    assert score(data, data.questions, answers("the"))["word_f1"] == 0


def test_answerable_and_absence_results_are_separate() -> None:
    questions = (Question("positive", "Clause?", (Span("yes", 0),)), Question("negative", "Other?"))
    data = Contract("c", "yes", questions)
    result = score(data, questions, Answers(answers=[Answer(question_id="positive"), Answer(question_id="negative")]))
    assert result["word_f1"] == 0.5
    assert result["answerable_f1"] == 0
    assert result["absence_accuracy"] == 1
    assert result["answerable_questions"] == result["absence_questions"] == 1


def test_literal_citations_locate_all_occurrences_and_penalize_extra_coverage() -> None:
    data = contract("yes yes", (Span("yes", 0),))
    result = score(data, data.questions, answers("yes"))
    assert result["literal_citations"] == 1
    assert result["citation_exact_match"] == 1
    assert result["per_question"][0]["citation_locations"][0]["occurrences"] == [[0, 3], [4, 7]]
    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 1
    assert result["citation_f1"] == pytest.approx(2 / 3)


def test_overlapping_occurrences_and_gold_spans_use_character_union() -> None:
    data = contract("aaaa", (Span("aaa", 0), Span("aaa", 1)))
    result = score(data, data.questions, answers("aaa"))
    assert result["citation_precision"] == result["citation_recall"] == result["citation_f1"] == 1
    assert result["per_question"][0]["citation_locations"][0]["occurrences"] == [[0, 3], [1, 4]]


def test_nonliteral_and_omitted_annotations_are_explicit_and_penalized() -> None:
    data = contract("red green <omitted>", (Span("red green", 0),))
    result = score(data, data.questions, answers("red green", "paraphrase", "<omitted>"))
    assert result["literal_citations"] == 1
    assert result["nonliteral_citations"] == 2
    assert result["omitted_annotations"] == 1
    assert result["citation_exact_match"] == pytest.approx(1 / 3)
    assert result["citation_precision"] < 1
    assert result["citation_recall"] == 1
    assert result["per_question"][0]["citation_locations"][2]["occurrences"] == []


def test_question_positions_and_unexpected_answers_are_recorded() -> None:
    text = "start" + " " * 40 + "middle" + " " * 40 + "end"
    questions = (
        Question("s", "Start?", (Span("start", 0),)),
        Question("m", "Middle?", (Span("middle", 45),)),
        Question("e", "End?", (Span("end", len(text) - 3),)),
        Question("a", "Absent?"),
    )
    data = Contract("c", text, questions)
    result = score(
        data, questions, Answers(answers=[Answer(question_id=q.id) for q in questions] + [Answer(question_id="extra")])
    )
    assert [q["position_bucket"] for q in result["per_question"]] == ["start", "middle", "end", "absent"]
    assert result["position"]["middle"]["questions"] == 1
    assert result["unexpected_answers"] == 1
    json.dumps(result, allow_nan=False)


def test_duplicate_requested_questions_are_rejected() -> None:
    data = contract("yes")
    with pytest.raises(ValueError, match="unique"):
        score(data, data.questions * 2, answers())


def test_usage_is_inclusive_and_cache_writes_are_disjoint() -> None:
    usage = summarize_usage(
        [
            {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 40, "cache_write_tokens": 10},
            {
                "prompt_tokens": 200,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 60},
                "cache_creation_input_tokens": 20,
            },
        ]
    )
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 350
    assert usage["cache_hit_tokens"] == 100
    assert usage["cache_write_tokens"] == 30
    assert usage["cache_miss_tokens"] == 170
    assert usage["cache_hit_rate"] == pytest.approx(1 / 3)
    assert not usage["usage_missing"]


def test_usage_uses_provider_hit_miss_detail_fallbacks() -> None:
    usage = summarize_usage(
        [
            {
                "input_tokens": 80,
                "output_tokens": 4,
                "details": {"prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 40, "cache_write_tokens": 10},
            }
        ]
    )
    assert usage["cache_hit_tokens"] == 30
    assert usage["cache_miss_tokens"] == 40
    assert usage["cache_write_tokens"] == 10
    assert usage["total_tokens"] == 84


def test_missing_usage_is_unknown_not_zero_or_partial_total() -> None:
    usage = summarize_usage([{"input_tokens": 100, "output_tokens": 5}, {"output_tokens": 2}])
    assert usage["input_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["output_tokens"] == 7
    assert usage["cache_hit_tokens"] is usage["cache_miss_tokens"] is usage["cache_write_tokens"] is None
    assert usage["observed_subtotals"]["input_tokens"] == 100
    assert usage["unknown_records"]["input_tokens"] == 1
    assert usage["usage_missing"]
    empty = summarize_usage([])
    assert empty["input_tokens"] is empty["total_tokens"] is None
    assert empty["usage_missing"]


def test_negative_derived_cache_miss_is_not_fabricated() -> None:
    usage = summarize_usage([{"input_tokens": 10, "output_tokens": 1, "cache_hit_tokens": 20, "cache_write_tokens": 0}])
    assert usage["cache_miss_tokens"] is None


def test_bootstrap_is_deterministic_and_clusters_all_n_of_one_contract() -> None:
    records = [row(n=1, value=0), row(n=2, value=1)]
    first = bootstrap_ci(records, samples=100, seed=37)
    assert first == bootstrap_ci(records, samples=100, seed=37)
    assert first["mean"] == first["low"] == first["high"] == 0.5
    assert first["contracts"] == 1
    mixed = bootstrap_ci([row("a", value=0), row("b", value=1)], samples=200, seed=0)
    assert mixed["mean"] == 0.5
    assert mixed["low"] == 0 and mixed["high"] == 1


def test_bootstrap_empty_unknown_and_invalid_sample_count() -> None:
    assert bootstrap_ci([])["mean"] is None
    record = row()
    record["metrics"]["word_f1"] = None
    assert bootstrap_ci([record])["contracts"] == 0
    with pytest.raises(ValueError, match="positive"):
        bootstrap_ci([], samples=0)


def test_paired_comparison_uses_contract_plus_n_and_shared_clusters() -> None:
    records = [
        row(method="a", n=1, value=1),
        row(method="b", n=1, value=0),
        row(method="a", n=2, value=0),
        row(method="b", n=2, value=1),
    ]
    result = paired_comparison(list(reversed(records)), "a", "b", bootstrap_samples=100, seed=5)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
    assert result["pairs"] == 2 and result["contracts"] == 1
    assert result == paired_comparison(records, "a", "b", bootstrap_samples=100, seed=5)


@pytest.mark.parametrize(
    "records",
    [
        [row(method="a")],
        [row(method="a"), row(method="b", n=2)],
        [row(method="a"), row(method="a"), row(method="b")],
    ],
)
def test_paired_comparison_rejects_missing_or_duplicate_pairs(records: list[dict]) -> None:
    with pytest.raises(ValueError):
        paired_comparison(records, "a", "b", bootstrap_samples=10)


def test_paired_comparison_rejects_unknown_metric() -> None:
    unknown = row(method="b")
    unknown["metrics"]["word_f1"] = None
    with pytest.raises(ValueError, match="known finite"):
        paired_comparison([row(method="a"), unknown], "a", "b", bootstrap_samples=10)


def test_summary_preserves_unknown_cost_and_usage_and_operational_totals() -> None:
    first, second = row("a"), row("b")
    second["usage"]["input_tokens"] = None
    second["cost_usd"] = None
    second["retries"] = 2
    second["schema_failures"] = 1
    second["failures"] = ["request_failed"]
    second["concurrency"] = 4
    result = summarize_rows([first, second], bootstrap_samples=20)[0]
    assert result["input_tokens"] is None
    assert result["input_tokens_unknown_records"] == 1
    assert result["output_tokens"] == 20
    assert result["cost_usd"] is None
    assert result["known_cost_usd"] == 0.1
    assert result["cost_usd_unknown_records"] == 1
    assert result["elapsed_seconds"] == 5
    assert result["concurrency_min"] == 2 and result["concurrency_max"] == 4
    assert result["retries"] == 2 and result["schema_failures"] == result["failures"] == 1
    assert result["cache_hit_rate"] is None


def test_reports_write_measured_csv_markdown_svg_and_position(tmp_path) -> None:
    record = row(method="a<&")
    data = contract("yes", (Span("yes", 0),))
    record["metrics"] = score(data, data.questions, answers("yes"))
    paths = write_reports([record], tmp_path, bootstrap_samples=20, seed=3)
    with open(paths["csv"], encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))[0]
    assert summary["word_f1"] == "1.0"
    assert summary["input_tokens"] == "100"
    assert summary["cache_write_tokens"] == "10"
    with open(paths["position_csv"], encoding="utf-8", newline="") as stream:
        position = list(csv.DictReader(stream))[0]
    assert position["sequence_position"] == "1" and position["word_f1"] == "1.0"
    with open(paths["markdown"], encoding="utf-8") as stream:
        markdown = stream.read()
    assert "95%" in markdown and "cache_write_tokens" in markdown and "Question position" in markdown
    root = ET.parse(paths["svg"]).getroot()
    assert root.tag.endswith("svg")
    text = "".join(root.itertext())
    assert "a<& N=1" in text and "100" in text and "1.0000" in text


@pytest.mark.parametrize(
    "gold,nonliteral,omitted,out_of_bounds",
    [
        ((Span("made up", 0),), 1, 0, 0),
        ((Span("<omitted>", 0),), 1, 1, 0),
        ((Span("yes", -1),), 1, 0, 1),
        ((Span("yes", 1000),), 1, 0, 1),
    ],
)
def test_invalid_gold_offsets_never_create_character_overlap(
    gold: tuple[Span, ...],
    nonliteral: int,
    omitted: int,
    out_of_bounds: int,
) -> None:
    data = contract("yes source text <omitted>", gold)
    result = score(data, data.questions, answers("yes"))
    assert result["gold_nonliteral_spans"] == nonliteral
    assert result["gold_omitted_spans"] == omitted
    assert result["gold_out_of_bounds"] == out_of_bounds
    assert result["citation_f1"] is None
    assert result["citation_precision"] is result["citation_recall"] is None
    assert result["citation_exact_match"] == 1
    assert result["citation_scored_questions"] == 0
    assert result["invalid_gold_questions"] == 1
    assert not result["per_question"][0]["citation_scoring_valid"]
    assert result["per_question"][0]["literal_gold_chars"] == 0


def test_partial_invalid_gold_retains_only_literal_intervals_but_unknown_char_score(tmp_path) -> None:
    data = contract("yes source text", (Span("yes", 0), Span("different annotation", 0)))
    result = score(data, data.questions, answers("yes", "different annotation"))
    assert result["word_f1"] == 1
    assert result["citation_f1"] is None
    assert result["per_question"][0]["literal_gold_chars"] == 3
    record = row()
    record["metrics"] = result
    paths = write_reports([record], tmp_path, bootstrap_samples=10)
    with open(paths["csv"], encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))[0]
    assert summary["gold_nonliteral_spans"] == "1"
    assert summary["citation_f1"] == ""
    with open(paths["markdown"], encoding="utf-8") as stream:
        assert "excluded from character averages" in stream.read()


def test_tool_agent_late_sequence_degradation_is_visible_and_missing_keeps_position(tmp_path) -> None:
    questions = tuple(Question(str(i), "Clause?", (Span("yes", 0),)) for i in range(1, 4))
    data = Contract("c", "yes", questions)
    predicted = Answers(
        answers=[Answer(question_id="1", spans=[Citation(quote="yes", block_id="c")]), Answer(question_id="2")]
    )
    record = row(method="tool_agent", n=3)
    record["metrics"] = score(data, questions, predicted)
    details = record["metrics"]["per_question"]
    assert [item["sequence_position"] for item in details] == [1, 2, 3]
    assert [item["position_bucket"] for item in details] == ["start", "start", "start"]
    assert details[2]["missing_answer"]
    paths = write_reports([record], tmp_path, bootstrap_samples=10)
    with open(paths["position_csv"], encoding="utf-8", newline="") as stream:
        positions = list(csv.DictReader(stream))
    assert [item["sequence_position"] for item in positions] == ["1", "2", "3"]
    assert [item["word_f1"] for item in positions] == ["1.0", "0.0", "0.0"]
    assert all(item["n"] == "3" and item["method"] == "tool_agent" for item in positions)
    svg_text = "".join(ET.parse(paths["svg"]).getroot().itertext())
    assert "tool_agent N=3 question #3" in svg_text


def test_empty_reports_still_emit_valid_artifacts(tmp_path) -> None:
    paths = write_reports([], tmp_path, bootstrap_samples=10)
    assert ET.parse(paths["svg"]).getroot().tag.endswith("svg")
    with open(paths["csv"], encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
