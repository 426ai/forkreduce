"""Approval boundaries must fail before dataset access or provider construction."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from benchmarks.__main__ import _approval, main, parser, run
from pydantic_ai import models


def test_unapproved_run_refuses_before_dataset_access(monkeypatch, tmp_path):
    monkeypatch.delenv("FORKREDUCE_BENCH_ALLOW_PAID", raising=False)
    args = parser().parse_args(["run", "--cache-dir", str(tmp_path / "never-created")])
    with models.override_allow_model_requests(False), pytest.raises(ValueError, match="paid_execution_disabled"):
        asyncio.run(run(args))
    assert not args.cache_dir.exists()


def test_approval_is_bound_to_exact_estimate_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("FORKREDUCE_BENCH_ALLOW_PAID", "1")
    path = tmp_path / "estimate.json"
    content = json.dumps({"workload": {"seed": 20261024}}).encode()
    path.write_bytes(content)
    args = parser().parse_args(
        [
            "run",
            "--estimate-file",
            str(path),
            "--approve-estimate",
            hashlib.sha256(content).hexdigest(),
            "--budget-usd",
            "1",
        ]
    )
    assert _approval(args)["workload"]["seed"] == 20261024
    path.write_bytes(content + b"\n")
    with pytest.raises(ValueError, match="approved_estimate_digest_mismatch"):
        _approval(args)


@pytest.mark.parametrize("budget", ["nan", "inf", "-1", "0"])
def test_nonfinite_or_nonpositive_budget_cannot_authorize(monkeypatch, budget):
    monkeypatch.setenv("FORKREDUCE_BENCH_ALLOW_PAID", "1")
    args = parser().parse_args(
        ["run", "--estimate-file", "unused.json", "--approve-estimate", "digest", "--budget-usd", budget]
    )
    with pytest.raises(ValueError, match="positive_budget_required"):
        _approval(args)


def test_cli_does_not_echo_credentials_in_exception(monkeypatch, capsys):
    import benchmarks.__main__ as cli

    secret = "syntheticCredentialMustNeverBePrinted"

    def fail(_args):
        raise ValueError(secret)

    monkeypatch.setattr(cli, "estimate", fail)
    monkeypatch.setattr("sys.argv", ["benchmarks", "estimate"])
    assert main() == 2
    assert secret not in capsys.readouterr().err
