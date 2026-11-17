"""Live smoke test of llm.py against both providers with the default roster (PLAN §3, §10). Costs < $0.05.

Run: `source ~/.api_keys && uv run pytest -m live tests/live/test_llm_live.py -q -s`
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from p2c_task_re.ledger import Ledger
from p2c_task_re.llm import Completion, RoleSpec, complete, load_roster

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER = load_roster(
    OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs/models/default.yaml"), resolve=True)
)
SONNET: RoleSpec = ROSTER["generator"].model_copy(update={"max_tokens": 4000})
GPT: RoleSpec = ROSTER["judge_regrade"].model_copy(update={"max_tokens": 4000})
PROMPT = "Reply with the single word OK."
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def needs_key(name: str):
    return pytest.mark.skipif(not os.environ.get(name), reason=f"{name} not set")


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "calls.jsonl")


def show(label: str, out: Completion) -> None:
    print(
        f"\n{label}: model={out.model} stop={out.stop_reason} refusal={out.refusal}"
        f" fallback={out.fallback_used}"
        f" tokens={out.input_tokens}/{out.output_tokens} cost={out.cost_usd} latency={out.latency_s}s"
        f" text={out.text[:60]!r}"
    )


def check(out: Completion, model_prefix: str) -> None:
    assert not out.refusal
    assert out.model.startswith(model_prefix)
    assert out.input_tokens > 0 and out.output_tokens > 0
    assert "OK" in out.text.upper()


@needs_key("ANTHROPIC_API_KEY")
def test_anthropic_plain(ledger):
    out = complete(SONNET, PROMPT, role="generator", system="You are terse.", ledger=ledger, tag="live")
    show("anthropic plain", out)
    check(out, "claude-sonnet-5-5")
    assert out.cost_usd and ledger.spent() == out.cost_usd


@needs_key("ANTHROPIC_API_KEY")
def test_anthropic_schema(ledger):
    out = complete(SONNET, PROMPT, role="critic", schema=SCHEMA, ledger=ledger)
    show("anthropic schema", out)
    check(out, "claude-sonnet-5-5")
    assert out.parsed is not None and out.parsed["answer"].strip().upper().startswith("OK")


@needs_key("ANTHROPIC_API_KEY")
def test_anthropic_fallback_accepted(ledger):
    out = complete(SONNET, PROMPT, role="generator", allow_fallback=True, ledger=ledger)
    show("anthropic fallback", out)
    # Sonnet 5.5's default fallback is Sonnet 5; a benign prompt should stay on Sonnet 5.5.
    check(out, "claude-sonnet-5")
    assert out.fallback_used == (out.model != "claude-sonnet-5-5")


@needs_key("OPENAI_API_KEY")
def test_openai_plain(ledger):
    out = complete(GPT, PROMPT, role="judge_regrade", system="You are terse.", ledger=ledger)
    show("openai plain", out)
    check(out, "gpt-6.1-sol")
    assert out.cost_usd is not None and out.cost_usd > 0  # the roster prices gpt-6.1-sol


@needs_key("OPENAI_API_KEY")
def test_openai_schema(ledger):
    out = complete(GPT, PROMPT, role="judge_regrade", schema=SCHEMA, ledger=ledger)
    show("openai schema", out)
    check(out, "gpt-6.1-sol")
    assert out.parsed is not None and out.parsed["answer"].strip().upper().startswith("OK")
