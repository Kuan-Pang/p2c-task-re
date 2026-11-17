"""Offline tests for llm.py: request shapes, refusal/fallback reporting, cost, roster loading (PLAN §3, §10).

The fakes return real SDK response objects (validated from dicts), so attribute access matches the SDKs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from omegaconf import OmegaConf
from openai.types.responses import Response
from pydantic import ValidationError

from p2c_task_re import llm
from p2c_task_re.ledger import BudgetExceeded, CallRecord, Ledger
from p2c_task_re.llm import LLMError, Price, RoleSpec, complete, cost_usd

SONNET = RoleSpec(
    provider="anthropic", model="claude-sonnet-5-5", price_per_mtok=Price(input=2.0, output=10.0)
)
GPT = RoleSpec(provider="openai", model="gpt-6.1-sol")
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# ── fakes ─────────────────────────────────────────────────────────────────────────────────────────


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, message=None, error=None):
        self.message, self.error, self.calls = message, error, []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeStream(self.message)


class FakeAnthropic:
    def __init__(self, message=None, error=None):
        self.messages = FakeMessages(message, error)
        self.beta = SimpleNamespace(messages=FakeMessages(message, error))


class FakeResponses:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def anthropic_message(
    text="", *, stop_reason="end_turn", category=None, model="claude-sonnet-5-5", iterations=None, beta=False
):
    data = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_details": {"type": "refusal", "category": category} if stop_reason == "refusal" else None,
        "usage": {"input_tokens": 1000, "output_tokens": 500},
    }
    if beta:
        data["usage"]["iterations"] = iterations
        return BetaMessage.model_validate(data)
    return Message.model_validate(data)


def openai_response(content, *, status="completed", incomplete=None, model="gpt-6.1-sol-2026-10-01"):
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": status,
            "incomplete_details": {"reason": incomplete} if incomplete else None,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": content,
                },
            ],
            "usage": {
                "input_tokens": 200,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 300,
                "output_tokens_details": {"reasoning_tokens": 250},
                "total_tokens": 500,
            },
        }
    )


def use(monkeypatch, provider, client):
    monkeypatch.setattr(llm, "get_client", lambda p: client if p == provider else pytest.fail(p))
    return client


# ── anthropic ─────────────────────────────────────────────────────────────────────────────────────


def test_anthropic_plain_request_shape(monkeypatch):
    client = use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message("hello")))
    out = complete(SONNET, "hi", role="generator", system="be brief")

    (kwargs,) = client.messages.calls
    assert kwargs == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "system": "be brief",
    }
    assert not client.beta.messages.calls
    assert out.text == "hello" and out.parsed is None and not out.refusal and not out.fallback_used
    assert out.model == "claude-sonnet-5-5" and out.stop_reason == "end_turn"
    assert (out.input_tokens, out.output_tokens) == (1000, 500)
    assert out.cost_usd == pytest.approx(1000 * 2e-6 + 500 * 10e-6)


def test_anthropic_schema_uses_structured_output(monkeypatch):
    client = use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message('{"answer": "KO"}')))
    messages = [{"role": "user", "content": "q"}]
    out = complete(SONNET, messages, role="critic", schema=SCHEMA)

    (kwargs,) = client.messages.calls
    assert kwargs["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kwargs["messages"] == messages and "system" not in kwargs
    assert all(m["role"] != "assistant" for m in kwargs["messages"])  # no prefill
    assert out.parsed == {"answer": "KO"}


def test_anthropic_truncated_json_parses_to_none(monkeypatch):
    use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message('{"answer": "K', stop_reason="max_tokens")))
    out = complete(SONNET, "q", role="critic", schema=SCHEMA)
    assert out.parsed is None and out.stop_reason == "max_tokens" and not out.refusal


def test_anthropic_refusal(monkeypatch):
    use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message("", stop_reason="refusal", category="bio")))
    out = complete(SONNET, "q", role="panel", schema=SCHEMA)
    assert out.refusal and out.refusal_category == "bio" and out.parsed is None
    assert out.stop_reason == "refusal"


def test_anthropic_fallback_request_and_reporting(monkeypatch):
    served = anthropic_message(
        "ok",
        model="claude-sonnet-5",
        beta=True,
        iterations=[
            {
                "type": "message",
                "input_tokens": 1000,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            {
                "type": "fallback_message",
                "model": "claude-sonnet-5",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ],
    )
    client = use(monkeypatch, "anthropic", FakeAnthropic(served))
    out = complete(SONNET, "q", role="generator", allow_fallback=True)

    assert not client.messages.calls
    (kwargs,) = client.beta.messages.calls
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"] and kwargs["fallbacks"] == "default"
    assert out.fallback_used and out.model == "claude-sonnet-5" and not out.refusal


def test_anthropic_fallback_requested_but_not_needed(monkeypatch):
    plain = anthropic_message(
        "ok",
        beta=True,
        iterations=[
            {
                "type": "message",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ],
    )
    use(monkeypatch, "anthropic", FakeAnthropic(plain))
    out = complete(SONNET, "q", role="generator", allow_fallback=True)
    assert not out.fallback_used and out.model == "claude-sonnet-5-5"


# ── openai ────────────────────────────────────────────────────────────────────────────────────────


def test_openai_plain_request_shape(monkeypatch):
    client = use(
        monkeypatch,
        "openai",
        FakeOpenAI(openai_response([{"type": "output_text", "text": "hello", "annotations": []}])),
    )
    out = complete(GPT, "hi", role="panel", system="be brief")

    (kwargs,) = client.responses.calls
    assert kwargs == {
        "model": "gpt-6.1-sol",
        "input": "hi",
        "reasoning": {"effort": "high"},
        "max_output_tokens": 32000,
        "instructions": "be brief",
    }
    assert out.text == "hello" and not out.refusal and out.parsed is None
    assert out.model == "gpt-6.1-sol-2026-10-01" and out.stop_reason == "completed"
    assert (out.input_tokens, out.output_tokens) == (200, 300)
    assert out.cost_usd is None  # price unknown


def test_openai_schema_uses_strict_json_schema(monkeypatch):
    content = [{"type": "output_text", "text": '{"answer": "KO"}', "annotations": []}]
    client = use(monkeypatch, "openai", FakeOpenAI(openai_response(content)))
    out = complete(GPT, [{"role": "user", "content": "q"}], role="checker", schema=SCHEMA)

    (kwargs,) = client.responses.calls
    assert kwargs["text"] == {
        "format": {"type": "json_schema", "name": "answer", "schema": SCHEMA, "strict": True}
    }
    assert "temperature" not in kwargs and "top_p" not in kwargs
    assert out.parsed == {"answer": "KO"}


def test_openai_refusal_content_item(monkeypatch):
    use(monkeypatch, "openai", FakeOpenAI(openai_response([{"type": "refusal", "refusal": "I can't help."}])))
    out = complete(GPT, "q", role="panel", schema=SCHEMA)
    assert out.refusal and out.parsed is None and out.text == "I can't help."
    assert out.refusal_category is None


def test_openai_content_filter_counts_as_refusal(monkeypatch):
    use(
        monkeypatch,
        "openai",
        FakeOpenAI(openai_response([], status="incomplete", incomplete="content_filter")),
    )
    out = complete(GPT, "q", role="panel")
    assert out.refusal and out.refusal_category == "content_filter" and out.stop_reason == "content_filter"


def test_openai_fallback_is_recorded_not_faked(monkeypatch, tmp_path):
    client = use(
        monkeypatch,
        "openai",
        FakeOpenAI(openai_response([{"type": "output_text", "text": "x", "annotations": []}])),
    )
    ledger = Ledger(tmp_path / "calls.jsonl")
    with pytest.warns(UserWarning, match="no server-side refusal fallback"):
        out = complete(GPT, "q", role="generator", allow_fallback=True, ledger=ledger)
    (kwargs,) = client.responses.calls
    assert "fallbacks" not in kwargs and not out.fallback_used
    (line,) = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert json.loads(line)["fallback_requested"] is True and json.loads(line)["fallback_used"] is False


# ── cost, ledger integration, errors ──────────────────────────────────────────────────────────────


def test_cost_usd():
    assert cost_usd(Price(input=2.0, output=10.0), 1_000_000, 100_000) == pytest.approx(3.0)
    assert cost_usd(Price(input=2.0, output=None), 10, 10) is None
    assert cost_usd(Price(), 10, 10) is None


def test_complete_records_metadata_only(monkeypatch, tmp_path):
    use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message("SECRET ANSWER")))
    ledger = Ledger(tmp_path / "calls.jsonl")
    complete(SONNET, "SECRET PROMPT", role="judge", system="SECRET SYSTEM", ledger=ledger, tag="t-01")

    raw = (tmp_path / "calls.jsonl").read_text()
    assert "SECRET" not in raw
    record = CallRecord.model_validate_json(raw)
    assert (record.role, record.tag, record.model_requested, record.model_used) == (
        "judge",
        "t-01",
        "claude-sonnet-5-5",
        "claude-sonnet-5-5",
    )
    assert record.cost_usd == pytest.approx(0.007) and record.error is None
    assert ledger.spent(role="judge") == pytest.approx(0.007)


def test_complete_records_refusals(monkeypatch, tmp_path):
    use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message("", stop_reason="refusal", category="bio")))
    ledger = Ledger(tmp_path / "calls.jsonl")
    complete(SONNET, "q", role="panel", ledger=ledger)
    record = CallRecord.model_validate_json((tmp_path / "calls.jsonl").read_text())
    assert record.refusal and record.refusal_category == "bio" and record.stop_reason == "refusal"


def test_complete_failure_is_recorded_and_raised(monkeypatch, tmp_path):
    use(monkeypatch, "anthropic", FakeAnthropic(error=ConnectionError("network down")))
    ledger = Ledger(tmp_path / "calls.jsonl")
    with pytest.raises(LLMError, match="network down") as info:
        complete(SONNET, "q", role="critic", ledger=ledger)
    assert isinstance(info.value.__cause__, ConnectionError)
    record = CallRecord.model_validate_json((tmp_path / "calls.jsonl").read_text())
    assert record.error == "ConnectionError: network down" and record.model_used is None


def test_budget_exceeded_blocks_the_call(monkeypatch):
    client = use(monkeypatch, "anthropic", FakeAnthropic(anthropic_message("ok")))
    ledger = Ledger(None, budget_usd=0.01)
    complete(SONNET, "q", role="generator", ledger=ledger)  # $0.007, under budget
    complete(SONNET, "q", role="generator", ledger=ledger)  # starts under budget, ends at $0.014
    with pytest.raises(BudgetExceeded):
        complete(SONNET, "q", role="generator", ledger=ledger)
    assert len(client.messages.calls) == 2


def test_get_client_sets_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    llm.get_client.cache_clear()
    try:
        assert llm.get_client("anthropic").max_retries == llm.MAX_RETRIES
        assert llm.get_client("openai").max_retries == llm.MAX_RETRIES
        assert llm.get_client("openai") is llm.get_client("openai")
    finally:
        llm.get_client.cache_clear()


# ── roster ────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg_models(repo_root: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(repo_root / "configs/models/default.yaml"), resolve=True)


def test_load_roster_from_default_yaml(cfg_models):
    roster = llm.load_roster(cfg_models)
    assert roster["generator"] == RoleSpec(
        provider="anthropic",
        model="claude-sonnet-5-5",
        effort="high",
        max_tokens=32000,
        price_per_mtok=Price(input=2.0, output=10.0),
    )
    assert [s.model for s in roster["panel"]] == ["gpt-6.1-sol", "claude-sonnet-5-5"]
    assert [s.provider for s in roster["checker"]] == ["openai", "anthropic"]
    assert roster["judge_regrade"].model == "gpt-6.1-sol"
    assert roster["judge_regrade"].price_per_mtok == Price(input=2.0, output=10.0)
    assert roster["heldout"] is None


def test_refusal_fallback_roles(cfg_models):
    assert all(llm.allows_refusal_fallback(cfg_models, r) for r in ("generator", "critic", "adversary"))
    measured = ("panel", "checker", "judge", "judge_regrade", "heldout")
    assert not any(llm.allows_refusal_fallback(cfg_models, r) for r in measured)


def test_refusal_fallback_rejects_measured_roles(cfg_models):
    cfg_models["refusal_fallback"] = ["generator", "panel"]
    with pytest.raises(ValueError, match="panel"):
        llm.load_roster(cfg_models)


def test_role_spec_rejects_sampling_params():
    with pytest.raises(ValidationError):
        RoleSpec(provider="anthropic", model="claude-sonnet-5-5", temperature=0.0)
