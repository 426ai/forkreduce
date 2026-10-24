"""Synthetic accounting tests: byte-token fixture avoids even tokenizer downloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from benchmarks import estimate
from benchmarks.data import CATEGORIES
from benchmarks.schema import Contract, MethodConfig, Question


class _ByteEncoding:
    def encode(self, text: str, *, disallowed_special: tuple) -> list[int]:
        assert disallowed_special == ()
        return list(text.encode())


@pytest.fixture(autouse=True)
def offline_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(estimate, "_encoding", lambda: _ByteEncoding())


def _contracts(count: int = 1, text: str = "A commercial contract. " * 5) -> list[Contract]:
    return [
        Contract(
            f"contract-{index:03}",
            text + str(index),
            tuple(Question(category, f"Find {category}") for category in CATEGORIES),
        )
        for index in range(count)
    ]


def _report(**kwargs: object) -> dict:
    kwargs.setdefault("config", estimate.EstimateConfig(repair_rounds=0))
    return estimate.estimate_grid(_contracts(), question_counts=(5,), **kwargs)


def test_token_count_uses_encoding_without_special_token_rejection() -> None:
    assert estimate.count_tokens("<|endoftext|> café") == len("<|endoftext|> café".encode())


def test_distribution_linear_percentiles_and_mean() -> None:
    contracts = [Contract(str(n), "x" * n, ()) for n in (8, 2, 6, 4)]
    distribution = estimate.token_length_distribution(contracts)
    assert distribution == {
        "tokenizer": "cl100k_base",
        "count": 4,
        "min": 2,
        "p25": 3.5,
        "median": 5.0,
        "p75": 6.5,
        "p95": pytest.approx(7.7),
        "max": 8,
        "mean": 5,
    }
    single = estimate.token_length_distribution([contracts[0]])
    assert all(single[key] == 8 for key in ("min", "p25", "median", "p75", "p95", "max", "mean"))
    with pytest.raises(ValueError, match="At least one"):
        estimate.token_length_distribution([])


def test_full_180_grid_nested_scenarios_and_json_serialization() -> None:
    report = estimate.estimate_grid(_contracts(20), config=estimate.EstimateConfig(repair_rounds=0))
    assert report["workload_count"] == len(report["rows"]) == 180
    assert report["dataset"]["contract_count"] == report["distribution"]["count"] == 20
    assert len(report["totals"]) == 3 * 3 * 2 * 2
    assert len({(row["contract_id"], row["n"], row["method"]) for row in report["rows"]}) == 180
    assert len(report["dataset"]["input_snapshot_sha256"]) == 64
    for row in report["rows"]:
        assert len(row["question_ids"]) == row["n"]
        assert set(row["scenarios"]) == {"deepseek-flash", "claude-sonnet-4-6"}
        for scenarios in row["scenarios"].values():
            assert set(scenarios) == {"cold", "warm"}
    assert json.loads(json.dumps(report)) == report


def test_inputs_are_not_sampled_or_capped() -> None:
    report = estimate.estimate_grid(_contracts(21), question_counts=(5,), methods=("one_shot",))
    assert report["workload_count"] == 21
    assert [item["id"] for item in report["dataset"]["contracts"]] == [c.id for c in _contracts(21)]


def test_nested_questions_are_shared_by_contracts_and_methods() -> None:
    report = estimate.estimate_grid(_contracts(2), config=estimate.EstimateConfig(repair_rounds=0))
    ids_by_n = {}
    for row in report["rows"]:
        assert row["question_ids"] == ids_by_n.setdefault(row["n"], row["question_ids"])
    assert ids_by_n[5] == ids_by_n[20][:5] == ids_by_n[41][:5]
    assert ids_by_n[20] == ids_by_n[41][:20]


def test_cold_disables_all_cache_and_warm_includes_first_write() -> None:
    report = _report()
    for row in report["rows"]:
        for model, scenarios in row["scenarios"].items():
            cold, warm = scenarios["cold"], scenarios["warm"]
            assert cold["cache_hit_tokens"] == cold["cache_write_tokens"] == 0
            assert cold["cache_miss_tokens"] == cold["input_tokens"]
            assert warm["input_tokens"] == cold["input_tokens"]
            assert (
                warm["cache_hit_tokens"] + warm["cache_miss_tokens"] + warm["cache_write_tokens"]
                == warm["input_tokens"]
            )
            if model == "claude-sonnet-4-6":
                assert warm["cache_write_tokens"] > 0
            else:
                assert warm["cache_write_tokens"] == 0
                assert warm["cache_miss_tokens"] > 0
            if row["method"] == "one_shot":
                assert warm["cache_hit_tokens"] == 0
            else:
                assert warm["cache_hit_tokens"] > 0


def test_fork_n_plus_reduce_pays_double_output() -> None:
    report = _report(config=estimate.EstimateConfig(output_tokens_per_question=100, repair_rounds=0))
    by_method = {row["method"]: row["scenarios"]["deepseek-flash"]["cold"] for row in report["rows"]}
    assert by_method["forkreduce"]["requests"] == 6
    assert by_method["forkreduce"]["output_tokens"] == 2 * 5 * 100
    assert by_method["one_shot"]["requests"] == 1
    assert by_method["one_shot"]["output_tokens"] == 5 * 100
    assert by_method["tool_agent"]["requests"] == 5 * 3
    assert by_method["tool_agent"]["output_tokens"] == 5 * (100 + 2 * 96)


def test_agent_history_grows_and_does_not_include_full_document() -> None:
    questions = _contracts()[0].questions[:3]
    config = estimate.EstimateConfig(repair_rounds=0, tool_result_tokens=50)
    short = estimate._workload(100, questions, "tool_agent", config, 3, (20, 30))
    long = estimate._workload(10000, questions, "tool_agent", config, 3, (20, 30))
    assert short == long
    assert len(short) == 9
    assert all(right.input_tokens > left.input_tokens for left, right in zip(short, short[1:], strict=False))
    assert short[1].reuse_tokens == short[0].input_tokens
    assert sum(request.input_tokens for request in short) > len(short) * short[0].input_tokens
    larger_results = estimate._workload(
        100,
        questions,
        "tool_agent",
        estimate.EstimateConfig(
            repair_rounds=0,
            tool_result_tokens=500,
        ),
        3,
        (20, 30),
    )
    assert sum(request.input_tokens for request in larger_results) > sum(request.input_tokens for request in short)


def test_actual_chunk_rendering_and_retrieval_bounds_are_accounted() -> None:
    contracts = _contracts(text="alpha beta gamma " * 30)
    method_config = MethodConfig(chunk_chars=60, chunk_overlap=20, read_chars=30, search_results=2)
    report = estimate.estimate_grid(contracts, question_counts=(5,), method_config=method_config)
    snapshot = report["dataset"]["contracts"][0]
    assert snapshot["rendered_document_tokens"] > snapshot["document_tokens"]
    assert snapshot["section_count"] > 1
    assert len(snapshot["modeled_tool_result_tokens"]) == 2
    assert all(value > 0 for value in snapshot["modeled_tool_result_tokens"])
    assert report["method_config"]["chunk_overlap"] == 20


def test_repairs_are_bounded_expected_contingency() -> None:
    config = estimate.EstimateConfig(repair_probability=0.2, repair_rounds=2, output_tokens_per_question=100)
    report = _report(config=config)
    repairable = {"forkreduce": 6, "tool_agent": 5, "one_shot": 1}
    for row in report["rows"]:
        usage = row["scenarios"]["deepseek-flash"]["cold"]
        responses = repairable[row["method"]]
        assert usage["expected_repair_requests"] == pytest.approx(responses * 0.24)
        assert usage["requests"] == pytest.approx(usage["base_requests"] + responses * 0.24)
        assert usage["maximum_requests"] == usage["base_requests"] + responses * 2
    fork = next(row for row in report["rows"] if row["method"] == "forkreduce")
    assert fork["scenarios"]["deepseek-flash"]["cold"]["output_tokens"] == pytest.approx(1000 * 1.24)


def test_write_pricing_is_separate_not_double_billed() -> None:
    rates = estimate.load_prices()["models"]["claude-sonnet-4-6"]
    usage = {"cache_miss_tokens": 0, "cache_hit_tokens": 0, "cache_write_tokens": 1_000_000, "output_tokens": 0}
    assert estimate.price_usage(usage, rates) == 3.75
    assert estimate.price_usage({**usage, "cache_miss_tokens": 1_000_000}, rates) == 6.75
    assert estimate.price_usage({**usage, "cache_hit_tokens": None}, rates) is None
    assert estimate.price_usage({}, rates) is None


def test_custom_prices_and_report_totals(tmp_path: Path) -> None:
    custom = {
        "currency": "USD",
        "unit": "per_million_tokens",
        "models": {
            "local-price": {
                "input_miss": 1,
                "input_hit": 0.1,
                "cache_write": 1.25,
                "output": 2,
            }
        },
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(custom))
    assert estimate.load_prices(path) == custom
    report = estimate.estimate_grid(_contracts(2), estimate.load_prices(path), question_counts=(5,))
    for total in report["totals"]:
        rows = [row for row in report["rows"] if row["n"] == total["n"] and row["method"] == total["method"]]
        usages = [row["scenarios"][total["model"]][total["cache_mode"]] for row in rows]
        assert total["usd"] == sum(usage["usd"] for usage in usages)
        assert total["requests"] == sum(usage["requests"] for usage in usages)
        assert total["contracts"] == 2


def test_snapshot_changes_with_text_questions_and_rendering() -> None:
    original = _report()["dataset"]["input_snapshot_sha256"]
    changed = estimate.estimate_grid(_contracts(text="Changed source"), question_counts=(5,))
    assert changed["dataset"]["input_snapshot_sha256"] != original
    config_changed = _report(method_config=MethodConfig(chunk_chars=50, chunk_overlap=10))
    assert config_changed["dataset"]["input_snapshot_sha256"] != original


def test_markdown_contains_complete_aggregate_table_and_caveats() -> None:
    report = _report()
    markdown = estimate.render_markdown(report)
    assert "tokenizer proxy" in markdown
    assert "Cold and warm are alternative" in markdown
    assert "input miss" in markdown and "write" in markdown
    assert "2026-10-24" in markdown
    assert "https://api-docs.deepseek.com/quick_start/pricing" in markdown
    assert "https://platform.claude.com/docs/en/about-claude/pricing" in markdown
    for total in report["totals"]:
        assert f"| {total['n']} | {total['method']} | {total['model']} | {total['cache_mode']} |" in markdown


@pytest.mark.parametrize(
    "config",
    [
        {"repair_probability": -1},
        {"repair_probability": float("nan")},
        {"repair_rounds": -1},
        {"tool_loops_per_question": 1.5},
        {"tool_result_tokens": -1},
    ],
)
def test_invalid_assumptions_are_rejected(config: dict) -> None:
    with pytest.raises(ValueError):
        estimate.EstimateConfig(**config)


def test_invalid_prices_and_grid_are_rejected() -> None:
    prices = copy.deepcopy(estimate.load_prices())
    prices["models"]["deepseek-flash"]["input_hit"] = -1
    with pytest.raises(ValueError, match="input_hit"):
        estimate.estimate_grid(_contracts(), prices, question_counts=(5,))
    with pytest.raises(ValueError, match="Cache modes"):
        _report(cache_modes=("free",))
    with pytest.raises(ValueError, match="positive"):
        estimate.estimate_grid(_contracts(), question_counts=(0,))
    with pytest.raises(ValueError, match="Methods"):
        _report(methods=("unknown",))
    with pytest.raises(ValueError, match="price entry"):
        _report(models=("unknown",))
