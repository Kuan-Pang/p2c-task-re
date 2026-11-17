"""Scripted judges and fixture loaders for the grader tests.

The scripted judge answers by criterion id from a table, so a test can state exactly which criteria
a given answer satisfies and then check the arithmetic the grader does with those verdicts. Its
evidence quote is pulled out of the response it was shown, so it passes the verbatim check the way a
well-behaved real judge would; tests that want the check to fail pass their own quote.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "grading"
FAMILIES = ("f1", "f2", "f3", "f4", "f5", "f6", "f7")

_RESPONSE = re.compile(r"<response>\n(.*?)\n</response>", re.DOTALL)
_CRITERION = re.compile(r"^# Criterion (\S+)\s*$", re.MULTILINE)
_CONSISTENCY = re.compile(r"^# Check: consistency\s*$", re.MULTILINE)


def task_dir(family: str) -> Path:
    return FIXTURES / family


def expected_outcomes(family: str) -> dict[str, dict[str, Any]]:
    return json.loads((FIXTURES / family / "expected.json").read_text(encoding="utf-8"))


def answer_text(family: str, name: str) -> str:
    return (FIXTURES / family / "answers" / name).read_text(encoding="utf-8")


def response_shown(user_message: str) -> str:
    """The response text the judge prompt put in front of the judge."""
    match = _RESPONSE.search(user_message)
    return match.group(1) if match else ""


def rationale_shown(user_message: str) -> str:
    return response_shown(user_message).split("\n\nRationale:\n", 1)[-1].strip()


def first_sentence(text: str) -> str:
    """A quote any judge could copy: the first sentence of the rationale it was shown."""
    text = text.strip()
    match = re.search(r"^.*?[.!?](?=\s|$)", text, re.DOTALL)
    return match.group(0) if match else text


class ScriptedJudge:
    """Answers `yes`/`no` per criterion id from a table, quoting the response it was shown.

    With `on_usage` it also reports `tokens` after every call, the way a real judge reports the token
    usage of its API response, so a test can check what the grader and the ledger do with it.
    """

    def __init__(
        self,
        verdicts: dict[str, str],
        *,
        consistency: str = "no",
        evidence: str | None = None,
        name: str = "scripted",
        jitter: float = 0.0,
        on_usage: Any | None = None,
        tokens: tuple[int, int] = (1000, 40),
    ) -> None:
        self.name = name
        self.verdicts = verdicts
        self.consistency = consistency
        self.evidence = evidence
        self.jitter = jitter
        self.on_usage = on_usage
        self.tokens = tokens
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._random = random.Random(17)

    def complete(self, system: str, user: str) -> str | None:
        with self._lock:
            self.calls.append(user)
            delay = self._random.random() * self.jitter if self.jitter else 0.0
        if self.on_usage is not None:
            self.on_usage(self.name, self.tokens[0], self.tokens[1], False)
        if delay:
            time.sleep(delay)
        criterion = _CRITERION.search(user)
        if criterion:
            verdict = self.verdicts[criterion.group(1)]
        elif _CONSISTENCY.search(user):
            verdict = self.consistency
        else:
            raise AssertionError("the judge prompt named neither a criterion nor the consistency check")
        quote = self.evidence if self.evidence is not None else first_sentence(rationale_shown(user))
        return json.dumps({"verdict": verdict, "evidence": quote if verdict == "yes" else ""})

    @property
    def criteria_seen(self) -> list[str]:
        return [m.group(1) for m in (_CRITERION.search(call) for call in self.calls) if m]


class RefusingJudge:
    """Always refuses, the way a real judge signals a safety decline."""

    def __init__(self, name: str = "refuser") -> None:
        self.name = name
        self.calls = 0

    def complete(self, system: str, user: str) -> str | None:
        self.calls += 1
        return None


class BrokenJudge:
    """Always fails, either by raising or by answering with something that is not a verdict."""

    def __init__(self, name: str = "broken", mode: str = "error") -> None:
        self.name = name
        self.mode = mode
        self.calls = 0

    def complete(self, system: str, user: str) -> str | None:
        self.calls += 1
        if self.mode == "error":
            raise RuntimeError("judge backend is down")
        return "I am afraid I cannot put that in JSON."


class FlakyJudge:
    """Fails its first `failures` calls, then behaves like the judge it wraps."""

    def __init__(self, inner: Any, failures: int, name: str | None = None) -> None:
        self.inner = inner
        self.failures = failures
        self.name = name or inner.name
        self.calls = 0

    def complete(self, system: str, user: str) -> str | None:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient judge error")
        return self.inner.complete(system, user)


def judge_for(outcome: dict[str, Any], **overrides: Any) -> ScriptedJudge:
    """The scripted judge an expected-outcome entry describes."""
    settings: dict[str, Any] = {
        "consistency": outcome.get("consistency", "no"),
        "evidence": outcome.get("evidence"),
    }
    settings.update(overrides)
    return ScriptedJudge(outcome["verdicts"], **settings)
