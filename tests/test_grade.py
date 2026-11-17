"""Tests for the grader: docs/task_format.md §4–§7.

The golden files under tests/fixtures/grading hold one synthetic task per family with its gold, its
ablations, the best constant policy and a seeded random answer, each paired with the outcome the
contract prescribes. The rest of the file tests the machinery around them: rubric validation, the
pre-judge pass, the judge protocol and the CLI.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from grade_helpers import (
    FAMILIES,
    BrokenJudge,
    FlakyJudge,
    RefusingJudge,
    ScriptedJudge,
    answer_text,
    expected_outcomes,
    judge_for,
    response_shown,
    task_dir,
)

from p2c_task_re import families
from p2c_task_re.grading import grade as g

GRADER_DIR = Path(g.__file__).resolve().parent


def rubric_of(family: str) -> dict[str, Any]:
    return json.loads((task_dir(family) / "rubric.json").read_text(encoding="utf-8"))


def gold_of(family: str) -> dict[str, Any]:
    return json.loads((task_dir(family) / "gold.json").read_text(encoding="utf-8"))


def silent_judge(family: str, **overrides: Any) -> ScriptedJudge:
    """A judge that grants nothing — what an answer with no derivation earns."""
    return ScriptedJudge({c["id"]: "no" for c in rubric_of(family)["criteria"]}, **overrides)


# ------------------------------------------------------------------- golden files, all seven families

GOLDEN = [(family, name) for family in FAMILIES for name in expected_outcomes(family)]


@pytest.mark.parametrize(
    ("family", "name"), GOLDEN, ids=[f"{family}-{name[:-5]}" for family, name in GOLDEN]
)
def test_golden_outcome(family: str, name: str) -> None:
    outcome = expected_outcomes(family)[name]
    judge = judge_for(outcome)
    result = g.grade(answer_text(family, name), task_dir(family), judge)

    assert result["status"] == outcome["status"]
    assert result["s_core"] == pytest.approx(outcome["s_core"], abs=1e-6)
    assert result["s_judge"] == pytest.approx(outcome["s_judge"], abs=1e-6)
    assert result["s"] == pytest.approx(outcome["s"], abs=1e-6)
    assert result["passed"] is outcome["pass"]
    assert result["task_id"] == rubric_of(family)["task_id"]
    if outcome["status"] in ("invalid_answer", "no_primary_mechanism"):
        assert judge.calls == [], "a gate that fires before the judge must call no judge"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_gold_answer_passes(family: str) -> None:
    outcome = expected_outcomes(family)["gold.json"]
    result = g.grade(answer_text(family, "gold.json"), task_dir(family), judge_for(outcome))
    assert result["passed"] is True
    assert result["s"] == pytest.approx(1.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_gold_rationales_do_not_echo_the_question(family: str) -> None:
    question = (task_dir(family) / "question.md").read_text(encoding="utf-8")
    assert g.masked_spans(gold_of(family)["rationale"], question) == []


@pytest.mark.parametrize("family", FAMILIES)
def test_every_criterion_is_judged_three_times(family: str) -> None:
    outcome = expected_outcomes(family)["gold.json"]
    judge = judge_for(outcome)
    result = g.grade(answer_text(family, "gold.json"), task_dir(family), judge, samples=3)
    for criterion in result["criteria"]:
        assert len([v for v in criterion["votes"] if v["valid"]]) == 3
    assert len(result["consistency"]["votes"]) == 3
    # One criterion per call: no call shows the judge a second criterion's text.
    texts = {c["text"] for c in rubric_of(family)["criteria"]}
    for call in judge.calls:
        assert sum(text in call for text in texts) <= 1
    assert sorted(judge.criteria_seen) == sorted(
        [c["id"] for c in rubric_of(family)["criteria"] for _ in range(3)]
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_the_judge_never_sees_the_gold(family: str) -> None:
    """The gold answer and its derivation stay out of every judge prompt (§6)."""
    name = "hedge.json" if "hedge.json" in expected_outcomes(family) else "constant.json"
    judge = judge_for(expected_outcomes(family)[name])
    g.grade(answer_text(family, name), task_dir(family), judge)
    gold = gold_of(family)
    for call in judge.calls:
        # The answer fields are the solver's own, so a correct answer may match the gold's; the
        # gold derivation is what must never reach the judge.
        assert gold["rationale"] not in call
        assert gold["task_id"] not in call


# -------------------------------------------------------------------------- chance floors and policies


def _constant_answers(core: dict[str, Any]) -> list[dict[str, Any]]:
    """Every constant policy open to an answer for this family (PLAN §6)."""
    if core["type"] == "marker_directions":
        return [{"markers": dict.fromkeys(core["markers"], value)} for value in core["classes"]]
    if core["type"] == "arm_enum":
        return [{"arms": dict.fromkeys(core["arms"], value)} for value in core["values"]]
    if core["type"] == "ranking":
        arms = list(core["arms"])
        return [
            {"ranking": [[arm] for arm in arms]},
            {"ranking": [[arm] for arm in reversed(arms)]},
            {"ranking": [arms]},
        ]
    if core["type"] == "choice":
        return [{"choice": option} for option in core["options"]]
    return []


@pytest.mark.parametrize("family", ["f1", "f2", "f3", "f6"])
def test_the_best_constant_policy_fails(family: str) -> None:
    """The best constant answer must stay at or under s = 0.35 and must never pass."""
    rubric = rubric_of(family)
    scores = []
    for answer in _constant_answers(rubric["core"]):
        result = g.grade(
            json.dumps({"answer": answer, "rationale": ""}), task_dir(family), silent_judge(family)
        )
        assert result["passed"] is False
        scores.append(result["s"])
    assert max(scores) <= 0.35, f"{family}: best constant policy scores {max(scores)}"


def test_the_mean_constant_choice_is_at_chance_for_f4() -> None:
    """F4 answers one field, so the only within-task constants are the options themselves: the
    policy "always answer option k" is a suite-level policy, and its expected score is the mean."""
    scores = []
    for answer in _constant_answers(rubric_of("f4")["core"]):
        result = g.grade(
            json.dumps({"answer": answer, "rationale": ""}), task_dir("f4"), silent_judge("f4")
        )
        assert result["passed"] is False
        scores.append(result["s"])
    assert sum(scores) / len(scores) <= 0.35


@pytest.mark.parametrize("family", ["f5", "f7"])
def test_the_canned_constant_answer_fails(family: str) -> None:
    outcome = expected_outcomes(family)["constant.json"]
    result = g.grade(answer_text(family, "constant.json"), task_dir(family), judge_for(outcome))
    assert result["passed"] is False
    assert result["s"] <= 0.35


@pytest.mark.parametrize("family", FAMILIES)
def test_the_seeded_random_answer_fails(family: str) -> None:
    outcome = expected_outcomes(family)["random_seeded.json"]
    result = g.grade(answer_text(family, "random_seeded.json"), task_dir(family), judge_for(outcome))
    assert result["passed"] is False


# The analytic floors of §5, measured on critical-field-free core specs: a critical field multiplies
# the floor by the chance of getting that one field right, which is a property of the task, not of
# the family's s_core function.
FLOOR_SAMPLES = 500


def test_f1_chance_floor_is_one_third_on_a_balanced_panel() -> None:
    core = {
        "type": "marker_directions",
        "classes": ["up", "down", "unchanged"],
        "markers": ["m1", "m2", "m3", "m4", "m5", "m6"],
    }
    gold = {"markers": {"m1": "up", "m2": "up", "m3": "down", "m4": "down",
                        "m5": "unchanged", "m6": "unchanged"}}
    rng = random.Random(11)
    scores = [
        g.s_core_marker_directions(
            core, gold, {"markers": {m: rng.choice(core["classes"]) for m in core["markers"]}}
        ).score
        for _ in range(FLOOR_SAMPLES)
    ]
    assert sum(scores) / FLOOR_SAMPLES == pytest.approx(1 / 3, abs=0.04)


def test_f2_chance_floor_is_one_over_the_enum_size() -> None:
    core = {"type": "arm_enum", "values": ["increased", "lost", "unchanged"],
            "arms": ["a1", "a2", "a3", "a4"]}
    gold = {"arms": {"a1": "increased", "a2": "lost", "a3": "unchanged", "a4": "lost"}}
    rng = random.Random(12)
    scores = [
        g.s_core_arm_enum(core, gold, {"arms": {a: rng.choice(core["values"]) for a in core["arms"]}}).score
        for _ in range(FLOOR_SAMPLES)
    ]
    assert sum(scores) / FLOOR_SAMPLES == pytest.approx(1 / 3, abs=0.04)


def test_f3_chance_floor_is_one_half() -> None:
    core = {"type": "ranking", "arms": ["a1", "a2", "a3", "a4"]}
    gold = {"ranking": [["a1", "a2"], ["a3"], ["a4"]]}
    rng = random.Random(13)
    scores = []
    for _ in range(FLOOR_SAMPLES):
        order = list(core["arms"])
        rng.shuffle(order)
        scores.append(g.s_core_ranking(core, gold, {"ranking": [[arm] for arm in order]}).score)
    assert sum(scores) / FLOOR_SAMPLES == pytest.approx(0.5, abs=0.04)


def test_f4_chance_floor_is_one_over_the_option_count() -> None:
    core = {"type": "choice", "options": ["o1", "o2", "o3", "o4"]}
    gold = {"choice": "o2"}
    rng = random.Random(14)
    scores = [
        g.s_core_choice(core, gold, {"choice": rng.choice(core["options"])}).score
        for _ in range(FLOOR_SAMPLES)
    ]
    assert sum(scores) / FLOOR_SAMPLES == pytest.approx(0.25, abs=0.05)


# ------------------------------------------------------------------------------ the pre-judge pass


@pytest.mark.parametrize(
    "answer_text_in",
    [None, "", "not json at all", "[1, 2, 3]", '{"rationale": "no answer object"}',
     '{"answer": "a string, not an object"}'],
)
def test_an_unusable_answer_is_a_valid_zero(answer_text_in: str | None) -> None:
    judge = silent_judge("f1")
    result = g.grade(answer_text_in, task_dir("f1"), judge)
    assert result["status"] == "invalid_answer"
    assert result["answer_format"] == "invalid"
    assert (result["s"], result["s_core"], result["s_judge"]) == (0.0, 0.0, 0.0)
    assert result["passed"] is False
    assert judge.calls == []
    assert all(criterion["votes"] == [] for criterion in result["criteria"])


# ---------------------------------------------------------------------------- the answer's formatting


def fenced(text: str, info: str = "json") -> str:
    return f"```{info}\n{text}\n```"


def test_a_bare_answer_object_is_recorded_as_bare() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), judge_for(outcome))
    assert result["answer_format"] == "bare"
    # Only the envelope's own fence counts: one quoted inside the rationale is just rationale text.
    envelope = json.loads(answer_text("f1", "gold.json"))
    envelope["rationale"] = "```json\n{}\n```"
    assert g.parse_answer(json.dumps(envelope)).answer_format == "bare"


@pytest.mark.parametrize("info", ["json", "", "JSON"])
def test_one_enclosing_fence_is_stripped_and_recorded_as_fenced(info: str) -> None:
    """A fenced answer object is a formatting slip, graded exactly like the same object unfenced."""
    bare = answer_text("f1", "gold.json")
    outcome = expected_outcomes("f1")["gold.json"]
    plain = g.grade(bare, task_dir("f1"), judge_for(outcome))
    result = g.grade(fenced(bare.strip(), info), task_dir("f1"), judge_for(outcome))

    assert result["answer_format"] == "fenced"
    assert result["status"] == "ok"
    assert result["passed"] is True
    scores = ("s", "s_core", "s_judge")
    assert [result[key] for key in scores] == [plain[key] for key in scores]


def test_whitespace_around_the_answer_does_not_make_it_invalid() -> None:
    bare = answer_text("f1", "gold.json").strip()
    outcome = expected_outcomes("f1")["gold.json"]
    for text in (f"\n\n  {bare}\n\n", f"\n{fenced(bare)}\n  "):
        result = g.grade(text, task_dir("f1"), judge_for(outcome))
        assert result["passed"] is True


def test_nothing_beyond_one_enclosing_fence_is_extracted() -> None:
    """Prose around the object, a second object, or a fence over part of the reply stay invalid."""
    bare = answer_text("f1", "gold.json").strip()
    replies = {
        "prose before the fence": f"Here is my answer:\n{fenced(bare)}",
        "prose after the fence": f"{fenced(bare)}\nI hope this helps.",
        "prose around a bare object": f"Here it is: {bare} done",
        "two fenced objects": f"{fenced(bare)}\n{fenced(bare)}",
        "two bare objects": f"{bare}\n{bare}",
        "an unclosed fence": f"```json\n{bare}",
        # A code fence opens on a line of its own, so this is not one.
        "a one-line fence": f"```json {bare}```",
    }
    for label, reply in replies.items():
        judge = silent_judge("f1")
        result = g.grade(reply, task_dir("f1"), judge)
        assert result["status"] == "invalid_answer", label
        assert result["answer_format"] == "invalid", label
        assert judge.calls == [], label


def test_parse_answer_reports_how_the_answer_arrived() -> None:
    bare = '{"answer": {"choice": "a"}, "rationale": "r"}'
    assert g.parse_answer(bare) == g.ParsedAnswer({"choice": "a"}, "r", "bare")
    assert g.parse_answer(fenced(bare)) == g.ParsedAnswer({"choice": "a"}, "r", "fenced")
    assert g.parse_answer(fenced(bare, "")) == g.ParsedAnswer({"choice": "a"}, "r", "fenced")
    assert g.parse_answer('{"answer": {"choice": "a"}}') == g.ParsedAnswer({"choice": "a"}, "", "bare")
    # A non-string rationale is not an unusable answer: it is graded as no rationale at all.
    assert g.parse_answer('{"answer": {"choice": "a"}, "rationale": 7}').rationale == ""
    for unusable in (None, "", fenced("not json"), f"text {fenced(bare)}"):
        assert g.parse_answer(unusable) is None


def test_the_rationale_is_truncated_at_the_cap() -> None:
    gold = gold_of("f1")["answer"]
    rationale = " ".join(["filler"] * 260) + " SENTINEL"
    judge = silent_judge("f1")
    result = g.grade(
        json.dumps({"answer": gold, "rationale": rationale}), task_dir("f1"), judge
    )
    assert result["truncated"] is True
    assert judge.calls and all("SENTINEL" not in call for call in judge.calls)
    assert all(call.count("filler") == 250 for call in judge.calls)


def test_a_short_rationale_is_not_marked_truncated() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), judge_for(outcome))
    assert result["truncated"] is False


def test_truncate_words_keeps_the_text_when_it_fits() -> None:
    assert g.truncate_words("one two three", 5) == ("one two three", False)
    assert g.truncate_words("one two three", 2) == ("one two", True)


def test_build_response_text_is_the_contract_format() -> None:
    text = g.build_response_text({"choice": "x"}, "because")
    assert text == '{\n  "choice": "x"\n}\n\nRationale:\nbecause'
    assert g.build_response_text({"k": "é"}, "").endswith('"é"\n}\n\nRationale:\n')


def test_masked_spans_need_ten_shared_tokens() -> None:
    question = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    nine = "beta gamma delta epsilon zeta eta theta iota kappa"
    ten = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    assert g.masked_spans(nine, question) == []
    assert g.masked_spans(ten, question) == [ten]
    assert g.masked_spans(f"first {ten} last", question) == [ten]


def test_a_byte_order_mark_does_not_make_an_answer_invalid() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    result = g.grade("\ufeff" + answer_text("f1", "gold.json"), task_dir("f1"), judge_for(outcome))
    assert result["answer_format"] == "bare"
    assert result["s"] == pytest.approx(1.0)


@pytest.mark.parametrize("family", ["f5", "f7"])
def test_an_extra_answer_field_zeroes_a_judged_core_too(family: str) -> None:
    gold = json.loads(answer_text(family, "gold.json"))
    gold["answer"]["notes"] = "more text for the judge"
    result = g.grade(json.dumps(gold), task_dir(family), judge_for(expected_outcomes(family)["gold.json"]))
    assert result["s_core"] == 0.0
    assert result["passed"] is False
    assert "notes" in result["core_detail"][-1]["note"]


def test_answer_keys_that_fold_together_are_off_contract() -> None:
    gold = gold_of("f2")["answer"]
    arm, value = next(iter(gold["arms"].items()))
    answer = {"arms": {**gold["arms"], arm.upper(): value}}
    assert g.score_core(rubric_of("f2"), gold, answer).score == 0.0


def test_the_f5_gate_blocks_the_judge_without_a_primary_mechanism() -> None:
    judge = silent_judge("f5")
    result = g.grade(answer_text("f5", "empty_mechanism.json"), task_dir("f5"), judge)
    assert result["status"] == "no_primary_mechanism"
    assert result["s"] == 0.0
    assert judge.calls == []


# ------------------------------------------------------------------------------- the judge protocol


def test_evidence_absent_from_the_response_is_not_credited() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    judge = judge_for(outcome, evidence="The response never contained this sentence.")
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), judge)
    assert result["s_judge"] == 0.0
    assert all(criterion["verdict"] == "no" for criterion in result["criteria"])
    reasons = {v["reason"] for c in result["criteria"] for v in c["votes"]}
    assert reasons == {"evidence is not in the response"}
    assert {v["raw_verdict"] for c in result["criteria"] for v in c["votes"]} == {"yes"}


def test_evidence_inside_a_masked_span_is_not_credited() -> None:
    outcome = expected_outcomes("f2")["echo.json"]
    judge = judge_for(outcome)
    result = g.grade(answer_text("f2", "echo.json"), task_dir("f2"), judge)
    assert result["masked_spans"], "a rationale copied from the question must be masked"
    assert result["s_judge"] == 0.0
    reasons = {v["reason"] for c in result["criteria"] for v in c["votes"]}
    assert reasons == {"evidence only echoes the question"}


def test_the_closing_period_of_an_echoed_sentence_does_not_lift_it_out_of_the_mask() -> None:
    quote = "Polymyxin B binds the lipid A moiety of LPS and keeps it away from any receptor."
    judge = judge_for(expected_outcomes("f2")["echo.json"], evidence=quote)
    result = g.grade(answer_text("f2", "echo.json"), task_dir("f2"), judge)
    assert result["s_judge"] == 0.0
    reasons = {v["reason"] for c in result["criteria"] for v in c["votes"]}
    assert reasons == {"evidence only echoes the question"}


def test_a_quote_the_solver_also_wrote_in_its_own_words_counts() -> None:
    echo = json.loads(answer_text("f2", "echo.json"))
    echo["rationale"] += " Polymyxin B binds the lipid A moiety, so that arm is lost."
    judge = judge_for(expected_outcomes("f2")["echo.json"], evidence="Polymyxin B binds the lipid A moiety")
    result = g.grade(json.dumps(echo), task_dir("f2"), judge)
    assert result["masked_spans"], "the first two sentences still echo the question"
    assert result["s_judge"] == pytest.approx(1.0)


def test_free_text_answer_fields_are_capped_and_masked() -> None:
    """An F7 `experiment` is shown to the judge like the rationale: cut to the cap, echoes masked."""
    echoed = "the blocking antibody reduces disease in Rag1-KO mice, which have ILC3s but no T cells"
    gold = json.loads(answer_text("f7", "gold.json"))
    gold["answer"]["experiment"] = f"{echoed}. " + " ".join(["filler"] * 500)
    judge = ScriptedJudge(
        dict.fromkeys(expected_outcomes("f7")["gold.json"]["verdicts"], "yes"), evidence=echoed
    )
    result = g.grade(json.dumps(gold), task_dir("f7"), judge)
    assert result["truncated"] is True
    assert all(response_shown(call).count("filler") < 400 for call in judge.calls)
    assert any(echoed in span for span in result["masked_spans"])
    assert result["s_core"] == result["s_judge"] == 0.0
    reasons = {v["reason"] for c in result["criteria"] for v in c["votes"]}
    assert reasons == {"evidence only echoes the question"}


def test_a_placeholder_in_the_answer_stays_literal() -> None:
    gold = json.loads(answer_text("f1", "gold.json"))
    gold["rationale"] = "{{CRITERION_TEXT}} {{QUESTION}}"
    judge = silent_judge("f1")
    g.grade(json.dumps(gold), task_dir("f1"), judge)
    texts = [criterion["text"] for criterion in rubric_of("f1")["criteria"]]
    for call in judge.calls:
        assert response_shown(call).endswith("Rationale:\n{{CRITERION_TEXT}} {{QUESTION}}")
        assert not any(text in response_shown(call) for text in texts)


def test_evidence_is_matched_up_to_case_and_whitespace() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    quote = "LOSING   prmt7\n opens the Tcf7 locus"
    judge = judge_for(outcome, evidence=quote)
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), judge)
    assert result["s_judge"] == pytest.approx(1.0)


@pytest.mark.parametrize("family", ["f1", "f5"])
def test_a_contradiction_zeroes_the_judge_section(family: str) -> None:
    outcome = expected_outcomes(family)["gold.json"]
    result = g.grade(
        answer_text(family, "gold.json"), task_dir(family), judge_for(outcome, consistency="yes")
    )
    assert result["status"] == "contradiction"
    assert result["consistency"]["contradiction"] is True
    assert result["s_judge"] == 0.0
    assert all(criterion["verdict"] == "no" for criterion in result["criteria"])
    assert result["passed"] is False
    # F5 scores its core through the same criteria, so a contradiction takes the core with it.
    assert result["s_core"] == (0.0 if family == "f5" else 1.0)


def test_the_majority_of_three_samples_decides() -> None:
    class TwoNoOneYes:
        name = "split"

        def __init__(self) -> None:
            self.seen = 0

        def complete(self, system: str, user: str) -> str:
            self.seen += 1
            verdict = "yes" if self.seen % 3 == 0 else "no"
            return json.dumps({"verdict": verdict, "evidence": "Losing Prmt7 opens the Tcf7 locus"})

    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), TwoNoOneYes(), concurrency=1)
    assert all(criterion["verdict"] == "no" for criterion in result["criteria"])


def test_a_refusal_goes_to_the_regrade_judge_and_is_recorded() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    primary, regrade = RefusingJudge(), judge_for(outcome, name="regrade-model")
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), primary, regrade)

    assert result["passed"] is True
    assert result["judge_models"] == ["refuser", "regrade-model"]
    votes = result["criteria"][0]["votes"]
    assert [v["model"] for v in votes[:2]] == ["refuser", "regrade-model"]
    assert votes[0]["valid"] is False and votes[0]["reason"] == "refusal"
    assert votes[1]["valid"] is True
    # A refusal is not transient, so the primary judge is asked exactly once per sample.
    assert primary.calls == 3 * (len(rubric_of("f1")["criteria"]) + 1)


def test_a_failed_sample_is_retried_before_the_regrade_judge() -> None:
    outcome = expected_outcomes("f1")["gold.json"]
    flaky = FlakyJudge(judge_for(outcome), failures=1, name="flaky")
    result = g.grade(answer_text("f1", "gold.json"), task_dir("f1"), flaky, concurrency=1)

    invalid = [v for c in result["criteria"] for v in c["votes"] if not v["valid"]]
    assert len(invalid) == 1
    assert invalid[0]["reason"].startswith("error: RuntimeError")
    assert result["passed"] is True


@pytest.mark.parametrize("mode", ["error", "unparseable"])
def test_too_few_valid_votes_is_a_grading_error(mode: str) -> None:
    cause = "error: RuntimeError" if mode == "error" else "unparseable response"
    with pytest.raises(g.GradingError, match=f"valid judge votes; last failure: {cause}"):
        g.grade(
            answer_text("f1", "gold.json"),
            task_dir("f1"),
            BrokenJudge(mode=mode),
            BrokenJudge(name="broken-regrade", mode=mode),
            retries=1,
        )


def test_results_are_deterministic_under_concurrency() -> None:
    outcome = expected_outcomes("f1")["wrong_mechanism.json"]
    runs = [
        g.grade(
            answer_text("f1", "wrong_mechanism.json"),
            task_dir("f1"),
            judge_for(outcome, jitter=0.01),
            concurrency=width,
        )
        for width in (1, 8, 8)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_parse_verdict_tolerates_a_fenced_block_and_rejects_junk() -> None:
    assert g.parse_verdict('```json\n{"verdict": "YES", "evidence": "q"}\n```') == ("yes", "q")
    assert g.parse_verdict('here it is: {"verdict": "no", "evidence": ""} done') == ("no", "")
    assert g.parse_verdict('{"verdict": "maybe", "evidence": ""}') is None
    assert g.parse_verdict("no json here") is None


def test_the_judge_prompt_has_its_three_sections() -> None:
    prompts = g.load_prompts()
    assert set(prompts) == {"system", "criterion_check", "consistency_check"}
    for placeholder in ("{{QUESTION}}", "{{RESPONSE}}", "{{CRITERION_ID}}", "{{CRITERION_TEXT}}"):
        assert placeholder in prompts["criterion_check"]
    assert "{{RESPONSE}}" in prompts["consistency_check"]
    # No placeholder may survive into a rendered prompt.
    rendered = g.load_prompts()["criterion_check"]
    for key, value in {"question": "Q", "response": "R", "criterion_id": "c1", "criterion_text": "T"}.items():
        rendered = rendered.replace("{{" + key.upper() + "}}", value)
    assert "{{" not in rendered


def test_the_judge_prompt_falls_back_to_the_copy_beside_grade_py(tmp_path: Path) -> None:
    assert g.prompt_path(task_dir("f1")) == GRADER_DIR / "judge_prompt.md"
    packaged = tmp_path / "judge_prompt.md"
    packaged.write_text((GRADER_DIR / "judge_prompt.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert g.prompt_path(tmp_path) == packaged


# --------------------------------------------------------------------------------------- judge spend

VERDICT_JSON = '{"verdict": "no", "evidence": ""}'


class FakeAnthropicClient:
    """What `AnthropicJudge.client()` returns, with the two fields the judge reads: content and usage."""

    def __init__(self, *, refusal: bool = False, tokens: tuple[int, int] = (1200, 55)) -> None:
        self.refusal = refusal
        self.tokens = tokens
        self.kwargs: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs.append(kwargs)
        return SimpleNamespace(
            stop_reason="refusal" if self.refusal else "end_turn",
            content=[SimpleNamespace(type="text", text=VERDICT_JSON)],
            usage=SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]),
        )


class FakeOpenAIClient:
    """What `OpenAIJudge.client()` returns: a Responses object with output, output_text and usage."""

    def __init__(self, *, refusal: bool = False, tokens: tuple[int, int] = (900, 30)) -> None:
        self.refusal = refusal
        self.tokens = tokens
        self.responses = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        part_type = "refusal" if self.refusal else "output_text"
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type=part_type)])],
            output_text=VERDICT_JSON,
            usage=SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]),
        )


def test_judge_usage_totals_every_call_per_model() -> None:
    usage = g.JudgeUsage()
    usage.add("claude-sonnet-5-5", 1000, 40)
    usage.add("claude-sonnet-5-5", 1200, 60, refusal=True)
    usage.add("gpt-6.1-sol", 800, 20)
    assert usage.per_model() == {
        "claude-sonnet-5-5": {"calls": 2, "input_tokens": 2200, "output_tokens": 100, "refusals": 1},
        "gpt-6.1-sol": {"calls": 1, "input_tokens": 800, "output_tokens": 20, "refusals": 0},
    }
    assert usage.calls() == 3
    assert g.JudgeUsage().per_model() == {}


@pytest.mark.parametrize("refusal", [False, True])
def test_a_real_judge_reports_the_usage_of_every_call_refusals_included(refusal: bool) -> None:
    """No network: the judges are given a stand-in client, which is all they touch."""
    usage = g.JudgeUsage()
    claude = g.make_judge("claude-sonnet-5-5", "medium", usage.add)
    gpt = g.make_judge("gpt-6.1-sol", "medium", usage.add)
    claude._client = FakeAnthropicClient(refusal=refusal)
    gpt._client = FakeOpenAIClient(refusal=refusal)

    answers = [judge.complete("system", "user") for judge in (claude, gpt)]

    assert answers == ([None, None] if refusal else [VERDICT_JSON, VERDICT_JSON])
    assert usage.per_model() == {
        "claude-sonnet-5-5": {
            "calls": 1, "input_tokens": 1200, "output_tokens": 55, "refusals": int(refusal)
        },
        "gpt-6.1-sol": {"calls": 1, "input_tokens": 900, "output_tokens": 30, "refusals": int(refusal)},
    }


def test_a_judge_built_without_a_callback_reports_nothing() -> None:
    judge = g.make_judge("claude-sonnet-5-5", "medium")
    judge._client = FakeAnthropicClient()
    assert judge.on_usage is None
    assert judge.complete("system", "user") == VERDICT_JSON  # the call still works


def test_grade_reports_the_judge_usage_it_was_given() -> None:
    usage = g.JudgeUsage()
    outcome = expected_outcomes("f2")["gold.json"]
    judge = ScriptedJudge(outcome["verdicts"], on_usage=usage.add, tokens=(1000, 40))
    result = g.grade(answer_text("f2", "gold.json"), task_dir("f2"), judge, usage=usage)

    calls = 3 * (len(rubric_of("f2")["criteria"]) + 1)  # samples x (criteria + the consistency check)
    assert result["judge_usage"] == {
        "scripted": {"calls": calls, "input_tokens": 1000 * calls, "output_tokens": 40 * calls,
                     "refusals": 0}
    }


def test_an_answer_graded_without_a_judge_call_reports_no_usage() -> None:
    usage = g.JudgeUsage()
    result = g.grade(None, task_dir("f2"), silent_judge("f2"), usage=usage)
    assert result["status"] == "invalid_answer"
    assert result["judge_usage"] == {}


# -------------------------------------------------------------------------------------- the rubric


@pytest.mark.parametrize("family", FAMILIES)
def test_check_accepts_every_fixture(family: str) -> None:
    assert g.check_tests_dir(task_dir(family)) == []


def _f1_rubric() -> dict[str, Any]:
    return rubric_of("f1")


def _drop_points(rubric: dict[str, Any]) -> None:
    for criterion in rubric["criteria"]:
        criterion["points"] = 0


RUBRIC_MUTATIONS = [
    (lambda r: r.update(w_core=0.80), "w_core"),
    (lambda r: r.update(w_core=0.25), "w_core"),
    (lambda r: r.update(tau=0.0), "tau"),
    (lambda r: r.update(rationale_word_cap=0), "rationale_word_cap"),
    (lambda r: r.update(direction="sideways"), "direction"),
    (lambda r: r.update(direction=None), "may not be null"),
    (lambda r: r.pop("direction"), "missing key `direction`"),
    (lambda r: r["criteria"][1].update(id=r["criteria"][0]["id"]), "duplicate criterion id"),
    (lambda r: r["criteria"][1].update(points=11), "points"),
    (lambda r: r["criteria"][1].update(points=2.5), "points"),
    (lambda r: r["criteria"][1].update(capability="vibes"), "capability"),
    (lambda r: r["criteria"][1].pop("grounding"), "grounding"),
    (lambda r: r["criteria"][1].update(text="  "), "has no text"),
    (lambda r: r["criteria"][1].update(core=True), "no criterion may be"),
    (_drop_points, "carry no points"),
    (lambda r: r["core"].update(critical=["not a marker"]), "is not a marker"),
    (lambda r: r["core"]["markers"].append(r["core"]["markers"][0].upper()), "lists an entry twice"),
    (lambda r: r.update(tau=True), "tau"),
    (lambda r: r["criteria"][1].update(core="false"), "is not true or false"),
    (lambda r: r["core"].update(type="choice"), "is not 'marker_directions'"),
    (lambda r: r.update(family="F9"), "family"),
    (lambda r: r.pop("core"), "missing key `core`"),
]


@pytest.mark.parametrize(
    ("mutate", "fragment"), RUBRIC_MUTATIONS, ids=[fragment for _, fragment in RUBRIC_MUTATIONS]
)
def test_validate_rubric_rejects(mutate: Any, fragment: str) -> None:
    rubric = _f1_rubric()
    mutate(rubric)
    problems = g.validate_rubric(rubric)
    assert any(fragment in problem for problem in problems), problems


def test_f5_needs_a_core_criterion_and_a_null_direction() -> None:
    rubric = rubric_of("f5")
    for criterion in rubric["criteria"]:
        criterion["core"] = False
    assert any("at least one `core: true`" in p for p in g.validate_rubric(rubric))

    rubric = rubric_of("f5")
    rubric["direction"] = "forward"
    assert any("direction must be null" in p for p in g.validate_rubric(rubric))


def test_f3_must_declare_its_top_tie_set_critical() -> None:
    rubric = rubric_of("f3")
    rubric["core"].pop("critical_top")
    assert any("critical_top must be true" in p for p in g.validate_rubric(rubric))


def test_the_core_block_may_only_name_what_the_instruction_defines() -> None:
    question = (task_dir("f1") / "question.md").read_text(encoding="utf-8")
    rubric = _f1_rubric()
    assert g.validate_rubric(rubric, question) == []
    rubric["core"]["markers"].append("CD103")
    rubric["core"]["critical"] = ["CD103"]
    assert any("never defines" in p for p in g.validate_rubric(rubric, question))


def test_check_reports_a_gold_that_does_not_score_one(tmp_path: Path) -> None:
    shutil.copytree(task_dir("f2"), tmp_path / "tests")
    gold = json.loads((tmp_path / "tests" / "gold.json").read_text(encoding="utf-8"))
    gold["answer"]["arms"]["LPS + anti-TLR4"] = "abolished"  # not one of the declared values
    (tmp_path / "tests" / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    problems = g.check_tests_dir(tmp_path / "tests")
    assert any("scores s_core" in problem for problem in problems)


def test_check_reports_a_gold_that_leaves_out_a_declared_marker(tmp_path: Path) -> None:
    """Otherwise --check passes, and every complete answer names a marker the gold lacks (s_core 0)."""
    shutil.copytree(task_dir("f1"), tmp_path / "tests")
    gold = json.loads((tmp_path / "tests" / "gold.json").read_text(encoding="utf-8"))
    critical = {g.fold(m) for m in rubric_of("f1")["core"]["critical"]}
    dropped = next(m for m in gold["answer"]["markers"] if g.fold(m) not in critical)
    del gold["answer"]["markers"][dropped]
    (tmp_path / "tests" / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    assert any("are not exactly core.markers" in p for p in g.check_tests_dir(tmp_path / "tests"))
    # and a complete answer is judged against the declared markers, not the gold's keys
    full = gold_of("f1")["answer"]
    assert g.score_core(rubric_of("f1"), gold["answer"], full).score == pytest.approx(1.0)


def test_check_reports_a_gold_a_constant_policy_would_match(tmp_path: Path) -> None:
    shutil.copytree(task_dir("f2"), tmp_path / "tests")
    gold = json.loads((tmp_path / "tests" / "gold.json").read_text(encoding="utf-8"))
    gold["answer"]["arms"] = dict.fromkeys(gold["answer"]["arms"], "lost")
    (tmp_path / "tests" / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    assert any("the same value" in p for p in g.check_tests_dir(tmp_path / "tests"))


def test_check_reports_a_missing_file(tmp_path: Path) -> None:
    shutil.copytree(task_dir("f2"), tmp_path / "tests")
    (tmp_path / "tests" / "rubric.json").unlink()
    assert g.check_tests_dir(tmp_path / "tests") == ["missing rubric.json"]


def test_grade_refuses_an_invalid_rubric(tmp_path: Path) -> None:
    shutil.copytree(task_dir("f1"), tmp_path / "tests")
    rubric = rubric_of("f1")
    rubric["w_core"] = 0.95
    (tmp_path / "tests" / "rubric.json").write_text(json.dumps(rubric), encoding="utf-8")
    with pytest.raises(g.TestsDirError, match="w_core"):
        g.grade(answer_text("f1", "gold.json"), tmp_path / "tests", silent_judge("f1"))


# ------------------------------------------------------------------------------------------ the CLI


def test_cli_check_returns_zero() -> None:
    assert g.main(["--check", "--tests-dir", str(task_dir("f3"))]) == 0


def test_cli_check_returns_one_on_a_broken_tests_dir(tmp_path: Path) -> None:
    shutil.copytree(task_dir("f1"), tmp_path / "tests")
    (tmp_path / "tests" / "question.md").unlink()
    assert g.main(["--check", "--tests-dir", str(tmp_path / "tests")]) == 1


def test_cli_writes_both_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = expected_outcomes("f2")["gold.json"]
    monkeypatch.setattr(
        g,
        "make_judge",
        lambda model, effort=None, on_usage=None: judge_for(outcome, name=model, on_usage=on_usage),
    )
    code = g.main(
        [
            "--tests-dir", str(task_dir("f2")),
            "--answer", str(task_dir("f2") / "answers" / "gold.json"),
            "--out-dir", str(tmp_path),
            "--judge-model", "claude-sonnet-5-5",
            "--regrade-model", "gpt-6.1-sol",
        ]
    )
    assert code == 0
    reward = json.loads((tmp_path / "reward.json").read_text(encoding="utf-8"))
    assert reward == {"reward": 1.0, "core": 1.0, "judged": 1.0}
    assert "pass" not in reward
    grading = json.loads((tmp_path / "grading.json").read_text(encoding="utf-8"))
    assert grading["judge_models"] == ["claude-sonnet-5-5", "gpt-6.1-sol"]
    assert grading["grade_version"] == g.GRADE_VERSION
    assert grading["answer_format"] == "bare"
    assert set(grading) == {
        "task_id", "status", "answer_format", "passed", "s", "s_core", "s_judge", "core_detail",
        "criteria", "consistency", "truncated", "masked_spans", "judge_models", "judge_usage",
        "grade_version",
    }


def test_cli_records_the_judge_spend_of_the_judges_it_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A Harbor run grades through this CLI, so its grading.json has to carry the judge spend."""
    outcome = expected_outcomes("f2")["gold.json"]
    monkeypatch.setattr(
        g,
        "make_judge",
        lambda model, effort=None, on_usage=None: judge_for(
            outcome, name=model, on_usage=on_usage, tokens=(1500, 60)
        ),
    )
    code = g.main(
        [
            "--tests-dir", str(task_dir("f2")),
            "--answer", str(task_dir("f2") / "answers" / "gold.json"),
            "--out-dir", str(tmp_path),
            "--judge-model", "claude-sonnet-5-5",
            "--regrade-model", "gpt-6.1-sol",
        ]
    )
    assert code == 0
    calls = 3 * (len(rubric_of("f2")["criteria"]) + 1)
    grading = json.loads((tmp_path / "grading.json").read_text(encoding="utf-8"))
    # Only the primary judge answers: the regrade judge is built but never reached.
    assert grading["judge_usage"] == {
        "claude-sonnet-5-5": {
            "calls": calls, "input_tokens": 1500 * calls, "output_tokens": 60 * calls, "refusals": 0
        }
    }
    assert f"judge_calls={calls}" in capsys.readouterr().out


def test_cli_exits_two_and_writes_no_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        g, "make_judge", lambda model, effort=None, on_usage=None: BrokenJudge(name=model)
    )
    monkeypatch.setenv("JUDGE_RETRIES", "0")
    code = g.main(
        [
            "--tests-dir", str(task_dir("f1")),
            "--answer", str(task_dir("f1") / "answers" / "gold.json"),
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 2
    assert not (tmp_path / "reward.json").exists()
    assert not (tmp_path / "grading.json").exists()


def test_cli_exits_one_on_a_bad_tests_dir(tmp_path: Path) -> None:
    assert g.main(["--tests-dir", str(tmp_path / "nowhere"), "--answer", "a", "--out-dir", "b"]) == 1


def test_make_judge_picks_the_provider() -> None:
    assert isinstance(g.make_judge("claude-sonnet-5-5"), g.AnthropicJudge)
    assert isinstance(g.make_judge("gpt-6.1-sol"), g.OpenAIJudge)
    assert isinstance(g.make_judge("o4-mini"), g.OpenAIJudge)
    with pytest.raises(g.TestsDirError):
        g.make_judge("llama-9")
    # Building a judge must not import an SDK or need a key.
    assert g.make_judge("claude-sonnet-5-5")._client is None


def test_provider_of_reads_the_model_name() -> None:
    assert g.provider_of("claude-sonnet-5-5") == "anthropic"
    assert g.provider_of("gpt-6.1-sol") == g.provider_of("o3") == "openai"
    assert g.provider_of("gemini-2") is None


def test_make_judge_passes_the_usage_callback_to_either_provider() -> None:
    usage = g.JudgeUsage()
    assert g.make_judge("claude-sonnet-5-5", "medium", usage.add).on_usage == usage.add
    assert g.make_judge("gpt-6.1-sol", "medium", usage.add).on_usage == usage.add


def test_the_copied_grader_runs_on_a_missing_answer_without_an_api_key(tmp_path: Path) -> None:
    """End to end, the way Harbor runs it: grade.py alone in a tests dir, no answer, no keys."""
    packaged = tmp_path / "tests"
    shutil.copytree(task_dir("f2"), packaged)
    for name in ("grade.py", "judge_prompt.md"):
        shutil.copy(GRADER_DIR / name, packaged / name)
    source = (packaged / "grade.py").read_text(encoding="utf-8")
    assert "import p2c_task_re" not in source and "from p2c_task_re" not in source

    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
    out_dir = tmp_path / "verifier"
    proc = subprocess.run(
        [sys.executable, str(packaged / "grade.py"), "--tests-dir", str(packaged),
         "--answer", str(tmp_path / "no-such-answer.json"), "--out-dir", str(out_dir)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads((out_dir / "reward.json").read_text(encoding="utf-8")) == {
        "reward": 0.0, "core": 0.0, "judged": 0.0
    }
    grading = json.loads((out_dir / "grading.json").read_text(encoding="utf-8"))
    assert grading["status"] == "invalid_answer"
    assert all(criterion["votes"] == [] for criterion in grading["criteria"])


# ------------------------------------------------------------------------- the package-side contract


@pytest.mark.parametrize("family", FAMILIES)
def test_families_schema_and_example_match_the_rubric(family: str) -> None:
    rubric = rubric_of(family)
    schema = families.answer_schema(rubric["family"], rubric["core"])
    assert schema["required"] == ["answer", "rationale"]
    example = families.example_answer(rubric["family"], rubric["core"])
    required = g.REQUIRED_ANSWER_FIELDS[rubric["family"]]
    assert set(required) <= set(example["answer"])
    assert g.extra_answer_fields(rubric["family"], example["answer"]) == []
    if rubric["direction"] is None:
        assert families.DIRECTIONS[rubric["family"]] == ()
    else:
        assert rubric["direction"] in families.DIRECTIONS[rubric["family"]]

