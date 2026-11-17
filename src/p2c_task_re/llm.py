"""Provider calls for every LLM role: Anthropic Messages and OpenAI Responses (PLAN §3 roster, §10 `llm.py`).

`complete()` is the one entry point. It sends no temperature or top_p (Sonnet 5.5 rejects non-default
sampling, so repeated samples are repeated calls), requests JSON through each provider's structured-output
mode when a schema is given, reports refusals instead of retrying them, and records every call in the ledger.
Transient errors (429, 5xx, connection) are retried by the SDK clients themselves (`MAX_RETRIES`).
"""

import functools
import json
import time
import warnings
from typing import Literal

import anthropic
import openai
from pydantic import BaseModel, ConfigDict

from p2c_task_re.ledger import CallRecord, Ledger

Provider = Literal["anthropic", "openai"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

MAX_RETRIES = 4
TIMEOUT_S = 900.0
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# PLAN §10: the refusal fallback may change the answering model only where its identity does not matter.
FALLBACK_ELIGIBLE_ROLES = frozenset({"generator", "critic", "adversary"})


class Price(BaseModel):
    """USD per million tokens; None = unknown."""

    model_config = ConfigDict(extra="forbid")

    input: float | None = None
    output: float | None = None


class RoleSpec(BaseModel):
    """One role entry of `configs/models/*.yaml`."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str
    effort: Effort = "high"
    max_tokens: int = 32000
    price_per_mtok: Price = Price()


class Completion(BaseModel):
    """One model answer. On a refusal, `text` is whatever was emitted (maybe partial) and `parsed` is None."""

    text: str
    parsed: dict | None
    refusal: bool
    refusal_category: str | None
    model: str  # the model that actually answered; differs from the request when a fallback fired
    fallback_used: bool
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float


class LLMError(RuntimeError):
    """A call that failed after the SDK's own retries (or with a non-retryable error such as a 400)."""


# ── roster ────────────────────────────────────────────────────────────────────────────────────────


def load_roster(cfg_models: dict) -> dict[str, RoleSpec | list[RoleSpec] | None]:
    """Turn the resolved `models` config (its `roles:` block) into RoleSpecs; list roles stay lists."""
    _fallback_roles(cfg_models)  # fail at startup on a forbidden `refusal_fallback` entry
    roster: dict[str, RoleSpec | list[RoleSpec] | None] = {}
    for role, value in cfg_models["roles"].items():
        if value is None:
            roster[role] = None
        elif isinstance(value, list):
            roster[role] = [RoleSpec.model_validate(v) for v in value]
        else:
            roster[role] = RoleSpec.model_validate(value)
    return roster


def allows_refusal_fallback(cfg_models: dict, role: str) -> bool:
    """Whether `role` is listed under `refusal_fallback:` and may therefore pass `allow_fallback=True`."""
    return role in _fallback_roles(cfg_models)


def _fallback_roles(cfg_models: dict) -> set[str]:
    listed = set(cfg_models.get("refusal_fallback") or [])
    if forbidden := listed - FALLBACK_ELIGIBLE_ROLES:
        raise ValueError(
            f"refusal_fallback may list only {sorted(FALLBACK_ELIGIBLE_ROLES)}, got {sorted(forbidden)}"
        )
    return listed


# ── calls ─────────────────────────────────────────────────────────────────────────────────────────


@functools.cache
def get_client(provider: Provider) -> anthropic.Anthropic | openai.OpenAI:
    """One SDK client per provider, created on first use; the key comes from the environment."""
    if provider == "anthropic":
        return anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    return openai.OpenAI(max_retries=MAX_RETRIES, timeout=TIMEOUT_S)


def cost_usd(price: Price, input_tokens: int, output_tokens: int) -> float | None:
    if price.input is None or price.output is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1e6


def complete(
    spec: RoleSpec,
    messages: str | list[dict],
    *,
    role: str,
    system: str | None = None,
    schema: dict | None = None,
    ledger: Ledger | None = None,
    allow_fallback: bool = False,
    tag: str | None = None,
) -> Completion:
    """Call `spec`'s model once and return its answer.

    `messages` is a user prompt or a provider-format message list. `schema` is a JSON schema for the answer,
    written in the strict subset both providers accept (every object has `additionalProperties: false` and
    lists all its properties as `required`); `parsed` is None if the text is not a JSON object (a refusal,
    or output cut off at `max_tokens`). `allow_fallback` turns on Anthropic's server-side refusal fallback;
    OpenAI has none, so there it is recorded as requested-but-unused. Raises `BudgetExceeded` before the
    call if the ledger's budget is spent, and `LLMError` if the call fails.
    """
    if ledger is not None:
        ledger.check_budget()
    if allow_fallback and spec.provider == "openai":
        warnings.warn("OpenAI has no server-side refusal fallback; allow_fallback is ignored", stacklevel=2)

    start = time.monotonic()
    try:
        if spec.provider == "anthropic":
            completion = _complete_anthropic(spec, messages, system, schema, allow_fallback)
        else:
            completion = _complete_openai(spec, messages, system, schema)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if ledger is not None:
            ledger.record(
                CallRecord(
                    role=role,
                    tag=tag,
                    provider=spec.provider,
                    model_requested=spec.model,
                    fallback_requested=allow_fallback,
                    latency_s=round(time.monotonic() - start, 3),
                    error=error[:500],
                )
            )
        raise LLMError(f"{spec.provider} {spec.model} call for role {role!r} failed: {error}") from exc

    completion.latency_s = round(time.monotonic() - start, 3)
    if ledger is not None:
        ledger.record(
            CallRecord(
                role=role,
                tag=tag,
                provider=spec.provider,
                model_requested=spec.model,
                model_used=completion.model,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=completion.cost_usd,
                refusal=completion.refusal,
                refusal_category=completion.refusal_category,
                stop_reason=completion.stop_reason,
                fallback_requested=allow_fallback,
                fallback_used=completion.fallback_used,
                latency_s=completion.latency_s,
            )
        )
    return completion


def _complete_anthropic(
    spec: RoleSpec, messages: str | list[dict], system: str | None, schema: dict | None, allow_fallback: bool
) -> Completion:
    output_config: dict = {"effort": spec.effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs: dict = {
        "model": spec.model,
        "max_tokens": spec.max_tokens,
        "messages": [{"role": "user", "content": messages}] if isinstance(messages, str) else messages,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
    }
    if system is not None:
        kwargs["system"] = system

    client = get_client("anthropic")
    api = client.messages
    if allow_fallback:  # the fallback lives on the beta surface only
        api = client.beta.messages
        kwargs |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    with api.stream(**kwargs) as stream:
        message = stream.get_final_message()

    text = "".join(block.text for block in message.content if block.type == "text")
    refusal = message.stop_reason == "refusal"
    # A fallback-served answer carries a `fallback_message` entry in usage.iterations (beta responses only).
    iterations = getattr(message.usage, "iterations", None) or []
    return Completion(
        text=text,
        parsed=_parse_json(text) if schema is not None and not refusal else None,
        refusal=refusal,
        refusal_category=message.stop_details.category if refusal and message.stop_details else None,
        model=message.model,
        fallback_used=any(it.type == "fallback_message" for it in iterations),
        stop_reason=message.stop_reason,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        # A fallback answer is priced at the requested model's rate (Sonnet 5.5 falls back to Sonnet 5:
        # same price).
        cost_usd=cost_usd(spec.price_per_mtok, message.usage.input_tokens, message.usage.output_tokens),
        latency_s=0.0,
    )


def _complete_openai(
    spec: RoleSpec, messages: str | list[dict], system: str | None, schema: dict | None
) -> Completion:
    kwargs: dict = {
        "model": spec.model,
        "input": messages,
        "reasoning": {"effort": spec.effort},  # the SDK accepts all five of our levels as-is
        "max_output_tokens": spec.max_tokens,
    }
    if system is not None:
        kwargs["instructions"] = system
    if schema is not None:
        json_schema = {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}
        kwargs["text"] = {"format": json_schema}

    response = get_client("openai").responses.create(**kwargs)
    if response.status == "failed":
        raise RuntimeError(f"response failed: {response.error}")

    texts, refusals = [], []
    for item in response.output:
        # Commentary-phase messages are not the answer (same rule as the SDK's own parser).
        if item.type != "message" or item.phase not in (None, "final_answer"):
            continue
        for part in item.content:
            if part.type == "output_text":
                texts.append(part.text)
            elif part.type == "refusal":
                refusals.append(part.refusal)
    incomplete = response.incomplete_details.reason if response.incomplete_details else None
    refusal = bool(refusals) or incomplete == "content_filter"
    text = "".join(texts) or "".join(refusals)
    usage = response.usage
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    return Completion(
        text=text,
        parsed=_parse_json(text) if schema is not None and not refusal else None,
        refusal=refusal,
        refusal_category="content_filter" if incomplete == "content_filter" else None,
        model=response.model,
        fallback_used=False,
        stop_reason=incomplete or response.status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(spec.price_per_mtok, input_tokens, output_tokens),
        latency_s=0.0,
    )


def _parse_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
