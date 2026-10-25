"""Synthetic CUAD fixtures: no network and no model calls."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from benchmarks import data
from benchmarks.schema import Contract, Question


def _qa(contract: str, category: str, answers: list[dict] | None = None) -> dict:
    return {"id": f"{contract}__{category}", "question": f"Find {category}", "answers": answers or []}


def test_aggregate_contract_categories_and_all_gold_spans() -> None:
    payload = {
        "data": [
            {
                "title": "b",
                "paragraphs": [
                    {
                        "context": "alpha beta alpha",
                        "qas": [
                            _qa("b", "Parties", [{"text": "alpha", "answer_start": 0}]),
                            _qa("b", "Empty"),
                        ],
                    }
                ],
            },
            {
                "title": "b",
                "paragraphs": [
                    {
                        "context": "alpha beta alpha",
                        "qas": [
                            _qa(
                                "b",
                                "Parties",
                                [{"text": "alpha", "answer_start": 11}, {"text": "alpha", "answer_start": 0}],
                            ),
                        ],
                    },
                    {
                        "context": "gamma",
                        "qas": [
                            _qa("b", "Parties", [{"text": "gamma", "answer_start": 0}]),
                        ],
                    },
                ],
            },
            {"title": "a", "paragraphs": [{"context": "none", "qas": [_qa("a", "Empty")]}]},
        ]
    }
    contracts = data.parse_contracts(payload)
    assert [contract.id for contract in contracts] == ["a", "b"]
    contract = contracts[1]
    assert contract.text == "alpha beta alpha\n\ngamma"
    questions = {question.id: question for question in contract.questions}
    assert questions["Empty"].answers == ()
    assert [(span.text, span.start) for span in questions["Parties"].answers] == [
        ("alpha", 0),
        ("alpha", 11),
        ("gamma", 18),
    ]
    for span in questions["Parties"].answers:
        assert contract.text[span.start : span.start + len(span.text)] == span.text


def test_reject_inconsistent_question_text_and_invalid_offsets() -> None:
    first = {"title": "a", "paragraphs": [{"context": "alpha", "qas": [_qa("a", "Parties")]}]}
    second = {"title": "b", "paragraphs": [{"context": "alpha", "qas": [_qa("b", "Parties")]}]}
    second["paragraphs"][0]["qas"][0]["question"] = "Different canonical text"
    with pytest.raises(ValueError, match="canonical question"):
        data.parse_contracts({"data": [first, second]})
    first["paragraphs"][0]["qas"][0]["answers"] = [{"text": "alpha", "answer_start": 1}]
    with pytest.raises(ValueError, match="gold span"):
        data.parse_contracts({"data": [first]})


def _contracts(count: int = 30) -> list[Contract]:
    return [
        Contract(
            str(index).zfill(3),
            "text",
            tuple(Question(category, f"Find {category}") for category in reversed(data.CATEGORIES)),
        )
        for index in range(count)
    ]


def test_sampling_uses_sorted_ids_and_stable_seed() -> None:
    contracts = _contracts()
    selected = data.sample_contracts(contracts)
    assert len(selected) == 20
    assert selected == data.sample_contracts(list(reversed(contracts)))
    assert [contract.id for contract in selected] == sorted(contract.id for contract in selected)
    assert selected != data.sample_contracts(contracts, seed=5)
    assert data.sample_contracts(contracts, count=0) == []
    with pytest.raises(ValueError, match="count"):
        data.sample_contracts(contracts, count=31)
    with pytest.raises(ValueError, match="unique"):
        data.sample_contracts([contracts[0], contracts[0]], count=1)


def test_question_selection_is_shared_nested_category_permutation() -> None:
    first, second = _contracts(2)
    second = Contract(second.id, second.text, tuple(reversed(second.questions)))
    five = data.select_questions(first, 5)
    twenty = data.select_questions(first, 20)
    all_questions = data.select_questions(first, 41)
    assert five == twenty[:5] == all_questions[:5]
    assert twenty == all_questions[:20]
    assert [q.id for q in all_questions] == [q.id for q in data.select_questions(second, 41)]
    assert len(set(q.id for q in all_questions)) == 41
    with pytest.raises(ValueError, match="count"):
        data.select_questions(first, 42)
    with pytest.raises(ValueError, match="unique"):
        data.select_questions(Contract("duplicate", "", (five[0], five[0])), 1)


@pytest.fixture
def synthetic_release(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes]:
    raw = json.dumps(
        {
            "data": [
                {
                    "title": f"contract-{index:03}",
                    "paragraphs": [
                        {"context": "text", "qas": [_qa(str(index), category) for category in data.CATEGORIES]}
                    ],
                }
                for index in range(510)
            ]
        }
    ).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(data.CUAD_MEMBER, raw)
    archive_raw = buffer.getvalue()
    monkeypatch.setattr(data, "CUAD_JSON_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(data, "CUAD_ARCHIVE_SHA256", hashlib.sha256(archive_raw).hexdigest())
    return raw, archive_raw


def test_pinned_download_and_cached_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_release: tuple[bytes, bytes],
) -> None:
    raw, archive_raw = synthetic_release
    requests = []

    def download(url: str, timeout: int) -> io.BytesIO:
        requests.append((url, timeout))
        return io.BytesIO(archive_raw)

    monkeypatch.setattr(data.urllib.request, "urlopen", download)
    contracts = data.load_contracts(tmp_path)
    assert len(contracts) == 510
    assert all(len(contract.questions) == 41 for contract in contracts)
    assert requests == [(data.CUAD_URL, 120)]
    assert data.CUAD_REVISION in data.CUAD_URL
    assert (tmp_path / data.CUAD_MEMBER).read_bytes() == raw
    assert (tmp_path / "data.zip").read_bytes() == archive_raw
    assert data.load_contracts(tmp_path) == contracts
    assert len(requests) == 1
    assert not list(tmp_path.glob("*.part"))


def test_existing_archive_is_verified_and_extracted_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_release: tuple[bytes, bytes],
) -> None:
    _, archive_raw = synthetic_release
    (tmp_path / "data.zip").write_bytes(archive_raw)

    def no_network(*args: object, **kwargs: object) -> None:
        pytest.fail("Cached dataset attempted a network call")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    assert len(data.load_contracts(tmp_path)) == 510
    (tmp_path / data.CUAD_MEMBER).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data.load_contracts(tmp_path)


def test_corrupt_archive_is_not_used_or_redownloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data.zip").write_bytes(b"tampered")
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No redownload on corruption"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data.load_contracts(tmp_path)
    assert not (tmp_path / data.CUAD_MEMBER).exists()
