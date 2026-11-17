"""Tests for ledger.py: spend totals, budget cap, resume, concurrent JSONL writes (PLAN §3 cost, §10)."""

import json
import threading

import pytest

from p2c_task_re.ledger import BudgetExceeded, CallRecord, Ledger


def call(role="generator", tag=None, cost=0.5, **extra) -> CallRecord:
    return CallRecord(
        role=role, tag=tag, provider="anthropic", model_requested="claude-sonnet-5-5", cost_usd=cost, **extra
    )


def test_spent_by_role_and_tag(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl")
    ledger.record(call("generator", "a", 1.0))
    ledger.record(call("critic", "a", 0.25))
    ledger.record(call("critic", "b", 0.5))
    ledger.record(call("panel", "b", None))  # unknown price counts as 0
    assert ledger.spent() == pytest.approx(1.75)
    assert ledger.spent(role="critic") == pytest.approx(0.75)
    assert ledger.spent(tag="b") == pytest.approx(0.5)
    assert ledger.spent(role="critic", tag="a") == pytest.approx(0.25)
    assert ledger.spent(role="panel") == 0.0


def test_budget_check(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl", budget_usd=1.0)
    ledger.check_budget()
    ledger.record(call(cost=0.6))
    ledger.check_budget()
    ledger.record(call(cost=0.4))
    with pytest.raises(BudgetExceeded, match=r"\$1\.0000 of the \$1\.00 budget"):
        ledger.check_budget()


def test_no_budget_never_raises():
    ledger = Ledger(None)
    ledger.record(call(cost=1e6))
    ledger.check_budget()
    assert ledger.spent() == 1e6


def test_resume_reloads_spend_and_skips_truncated_line(tmp_path):
    path = tmp_path / "run" / "calls.jsonl"
    first = Ledger(path)
    first.record(call(cost=0.3))
    first.record(call(cost=0.2, error="APIConnectionError: boom"))
    with path.open("a") as f:
        f.write('{"role": "gener')  # a crash mid-write
    resumed = Ledger(path, budget_usd=0.5)
    assert resumed.spent() == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        resumed.check_budget()


def test_lines_are_metadata_fields(tmp_path):
    path = tmp_path / "calls.jsonl"
    Ledger(path).record(call(tag="t", cost=0.1, input_tokens=10, output_tokens=5, latency_s=1.5))
    line = json.loads(path.read_text())
    assert set(line) == {
        "ts",
        "role",
        "tag",
        "provider",
        "model_requested",
        "model_used",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "refusal",
        "refusal_category",
        "stop_reason",
        "fallback_requested",
        "fallback_used",
        "latency_s",
        "error",
    }
    assert line["ts"].endswith("+00:00")


def test_concurrent_writes_produce_valid_jsonl(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = Ledger(path)
    n_threads, per_thread = 16, 100

    def work(i: int) -> None:
        for j in range(per_thread):
            ledger.record(call(role=f"r{i}", tag=f"t{j}", cost=0.001, error="x" * 300))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text().splitlines()
    assert len(lines) == n_threads * per_thread
    records = [CallRecord.model_validate_json(line) for line in lines]
    assert {r.role for r in records} == {f"r{i}" for i in range(n_threads)}
    assert ledger.spent() == pytest.approx(n_threads * per_thread * 0.001)
    assert Ledger(path).spent() == pytest.approx(ledger.spent())
