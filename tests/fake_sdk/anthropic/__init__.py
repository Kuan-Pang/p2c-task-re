"""Stand-in for the anthropic SDK, for a grade.py run in a subprocess (test.sh, `harbor run`).

Put tests/fake_sdk and tests/ on PYTHONPATH. Every judge prompt is answered "yes" with a quote from the
response, as grade_helpers.ScriptedJudge answers, at a fixed token usage per call. With P2C_FAKE_LOG set,
each call appends a line to that file; with P2C_FAKE_DOWN set, each call raises instead.
"""

import os
from types import SimpleNamespace

from grade_helpers import ScriptedJudge


class _Every(dict):
    def __missing__(self, criterion_id):
        return "yes"


class _Messages:
    def create(self, *, system, messages, **_):
        if os.environ.get("P2C_FAKE_LOG"):
            with open(os.environ["P2C_FAKE_LOG"], "a", encoding="utf-8") as log:
                log.write("call\n")
        if os.environ.get("P2C_FAKE_DOWN"):
            raise RuntimeError("judge backend is down")
        text = ScriptedJudge(_Every()).complete(system, messages[0]["content"])
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=40),
        )


class Anthropic:
    def __init__(self, **_):
        self.messages = _Messages()
