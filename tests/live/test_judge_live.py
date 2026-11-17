"""The real judges, against the F2 fixture. Opt in with `-m live`.

One sample per criterion rather than the contract's three: the protocol (majority, retries,
fallback) is covered offline, and what these tests buy is the real models' behaviour on a real
rubric. Run:

    source ~/.api_keys && uv run pytest -m live tests/live/test_judge_live.py -q -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from grade_helpers import RefusingJudge, answer_text, task_dir

from p2c_task_re.grading import grade as g

pytestmark = pytest.mark.live

EFFORT = os.environ.get("JUDGE_EFFORT", g.DEFAULT_JUDGE_EFFORT)


def report(result: dict[str, Any], label: str) -> None:
    """Print what the judge actually said, so a disagreement is visible in the test output."""
    print(f"\n--- {label}: status={result['status']} answer={result['answer_format']} "
          f"s={result['s']:.3f} core={result['s_core']:.3f} judged={result['s_judge']:.3f} "
          f"passed={result['passed']} spend={json.dumps(result['judge_usage'])}")
    for criterion in result["criteria"]:
        votes = criterion["votes"]
        detail = "; ".join(
            f"{v['model']} {v['raw_verdict'] or v['reason']}"
            + (f" [{v['reason']}]" if v["reason"] and v["raw_verdict"] else "")
            for v in votes
        )
        print(f"    {criterion['id']} ({criterion['points']}p) -> {criterion['verdict']}  {detail}")
        for vote in votes:
            if vote["evidence"]:
                print(f"        evidence: {vote['evidence'][:160]}")
    print(f"    consistency -> {result['consistency']['contradiction']}")


def graded_by_claude(answer: str) -> dict[str, Any]:
    """Grade one F2 answer with a Claude judge of its own, so each result carries its own spend."""
    usage = g.JudgeUsage()
    judge = g.AnthropicJudge("claude-sonnet-5-5", EFFORT, usage.add)
    return g.grade(answer, task_dir("f2"), judge, samples=1, concurrency=4, usage=usage)


def test_claude_judge_grades_the_f2_gold_and_an_ablation() -> None:
    gold = graded_by_claude(answer_text("f2", "gold.json"))
    report(gold, "gold")
    ablation = graded_by_claude(answer_text("f2", "wrong_critical.json"))
    report(ablation, "wrong_critical")

    assert gold["status"] == "ok"
    assert gold["s_core"] == 1.0
    assert gold["passed"] is True, "the gold answer must pass with the real judge"
    assert gold["judge_usage"]["claude-sonnet-5-5"]["calls"] == len(gold["criteria"]) + 1
    assert ablation["s_core"] == 0.0, "a wrong critical arm zeroes the core"
    assert ablation["passed"] is False


def test_the_gpt_regrade_judge_answers_when_the_primary_refuses() -> None:
    usage = g.JudgeUsage()
    regrade = g.OpenAIJudge("gpt-6.1-sol", EFFORT, usage.add)
    result = g.grade(
        answer_text("f2", "gold.json"),
        task_dir("f2"),
        RefusingJudge(name="always-refuses"),
        regrade,
        samples=1,
        concurrency=4,
        usage=usage,
    )
    report(result, "gold, regraded by gpt-6.1-sol")

    assert result["judge_models"] == ["always-refuses", "gpt-6.1-sol"]
    for criterion in result["criteria"]:
        models = [vote["model"] for vote in criterion["votes"]]
        assert models[0] == "always-refuses" and criterion["votes"][0]["reason"] == "refusal"
        assert "gpt-6.1-sol" in models
        assert [v for v in criterion["votes"] if v["valid"]], "the regrade judge produced no verdict"
    assert result["s_core"] == 1.0
    # Only the regrade judge calls an API here: the refusing one is local.
    assert list(result["judge_usage"]) == ["gpt-6.1-sol"]
