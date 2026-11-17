"""Tests for the in-process runner (PLAN §3 "One grading module", §9 sanity runs).

Judges are scripted. The in-process path takes `grade_helpers.ScriptedJudge` directly. The test.sh path
runs grade.py in a subprocess, which builds its own judges from the JUDGE_* variables, so it gets a
stand-in `anthropic` package on PYTHONPATH that answers like a ScriptedJudge; loading the same stand-in
in-process lets the parity test run both paths through the very same judge code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from grade_helpers import BrokenJudge, RefusingJudge, ScriptedJudge
from omegaconf import OmegaConf

from p2c_task_re import evaluate as ev
from p2c_task_re import llm
from p2c_task_re.grading import grade as g
from p2c_task_re.ledger import BudgetExceeded, CallRecord, Ledger
from p2c_task_re.llm import Completion, RoleSpec

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "tasks"
ABLATIONS = REPO / "tests" / "fixtures" / "ablations"
TASKS = ev.find_tasks(EXAMPLES)
TASK_IDS = [task.name for task in TASKS]
SOLVER = RoleSpec(provider="anthropic", model="claude-sonnet-5-5")
CONSISTENCY = {"no_contradiction": "no", "contradiction": "yes"}

FAKE_SDK = REPO / "tests" / "fake_sdk"  # an anthropic stand-in for grade.py run by test.sh


def yes_judge() -> ScriptedJudge:
    return ScriptedJudge(defaultdict(lambda: "yes"))


def expected_of(task_id: str) -> dict[str, Any]:
    return json.loads((ABLATIONS / task_id / "expected.json").read_text(encoding="utf-8"))


def criteria_count(task_dir: Path) -> int:
    return len(json.loads((task_dir / "tests" / "rubric.json").read_text(encoding="utf-8"))["criteria"])


def rows_by_task(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in summary["rows"]}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeJudgeSDK:
    """Where the stand-in SDK logs its calls."""

    def __init__(self, log: Path) -> None:
        self.log = log

    @property
    def calls(self) -> int:
        return len(self.log.read_text(encoding="utf-8").splitlines()) if self.log.exists() else 0


@pytest.fixture
def fake_judge_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeJudgeSDK:
    monkeypatch.setenv("PYTHONPATH", f"{FAKE_SDK}{os.pathsep}{REPO / 'tests'}")
    monkeypatch.setenv("P2C_FAKE_LOG", str(tmp_path / "judge_calls.log"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    spec = importlib.util.spec_from_file_location("anthropic", FAKE_SDK / "anthropic" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "anthropic", module)  # what AnthropicJudge imports in-process
    return FakeJudgeSDK(tmp_path / "judge_calls.log")


def fake_complete(text: str, *, refusal: bool = False, calls: list | None = None):
    """A stand-in for llm.complete that answers `text` and records how it was called."""

    def complete(spec: RoleSpec, messages: Any, **kwargs: Any) -> Completion:
        if calls is not None:
            calls.append({"spec": spec, "messages": messages, **kwargs})
        return Completion(
            text=text,
            parsed=None,
            refusal=refusal,
            refusal_category="cyber" if refusal else None,
            model=spec.model,
            fallback_used=False,
            stop_reason="refusal" if refusal else "end_turn",
            input_tokens=1200,
            output_tokens=300,
            cost_usd=0.0054,
            latency_s=1.5,
        )

    return complete


# ------------------------------------------------------------------------------------------- oracle


def test_oracle_passes_every_example_in_process(tmp_path: Path) -> None:
    judge = yes_judge()
    summary = ev.evaluate(TASKS, ["oracle"], tmp_path, judges=(judge, None))
    for task_id, row in rows_by_task(summary).items():
        assert row == {
            "task_id": task_id,
            "agent": "oracle",
            "n": 1,
            "n_valid": 1,
            "n_invalid": 0,
            "n_resumed": 0,
            "n_refusals": 0,
            "mean_s": 1.0,
            "mean_s_core": 1.0,
            "mean_s_judge": 1.0,
            "pass_rate": 1.0,
        }
        trial = tmp_path / task_id / "oracle" / "01"
        assert sorted(p.name for p in trial.iterdir()) == ["answer.json", "grading.json", "reward.json"]
        assert read_json(trial / "reward.json") == {"reward": 1.0, "core": 1.0, "judged": 1.0}
    assert read_json(tmp_path / "summary.json") == summary
    assert summary["judge_models"] == ["scripted"]
    assert len(judge.calls) == sum(3 * (criteria_count(task) + 1) for task in TASKS)


def test_oracle_passes_every_example_via_test_sh(tmp_path: Path, fake_judge_sdk: FakeJudgeSDK) -> None:
    summary = ev.evaluate(TASKS, ["oracle"], tmp_path / "out", via="test_sh")
    assert summary["judge_models"] == ["claude-sonnet-5-5", "gpt-6.1-sol"]
    for row in summary["rows"]:
        assert (row["n_valid"], row["mean_s"], row["mean_s_core"], row["pass_rate"]) == (1, 1.0, 1.0, 1.0)
    assert fake_judge_sdk.calls == sum(3 * (criteria_count(task) + 1) for task in TASKS)


def test_k_samples_get_their_own_numbered_folders(tmp_path: Path) -> None:
    summary = ev.evaluate(TASKS[:1], ["oracle"], tmp_path, k=2, judges=(yes_judge(), None))
    assert summary["k"] == 2 and summary["rows"][0]["n"] == 2
    assert sorted(p.name for p in (tmp_path / TASK_IDS[0] / "oracle").iterdir()) == ["01", "02"]


# ---------------------------------------------------------------------------------------------- nop


def assert_valid_zero(out: Path, summary: dict[str, Any]) -> None:
    for task_id, row in rows_by_task(summary).items():
        assert (row["n_valid"], row["n_invalid"], row["mean_s"], row["pass_rate"]) == (1, 0, 0.0, 0.0)
        trial = out / task_id / "nop" / "01"
        assert not (trial / "answer.json").exists()
        assert read_json(trial / "reward.json") == {"reward": 0.0, "core": 0.0, "judged": 0.0}
        assert read_json(trial / "grading.json")["status"] == "invalid_answer"


def test_nop_is_a_valid_zero_in_process_with_no_judge_call(tmp_path: Path) -> None:
    judge = yes_judge()
    summary = ev.evaluate(TASKS, ["nop"], tmp_path, judges=(judge, None))
    assert_valid_zero(tmp_path, summary)
    assert judge.calls == []


def test_nop_is_a_valid_zero_via_test_sh_with_no_judge_call(
    tmp_path: Path, fake_judge_sdk: FakeJudgeSDK
) -> None:
    summary = ev.evaluate(TASKS, ["nop"], tmp_path / "out", via="test_sh")
    assert_valid_zero(tmp_path / "out", summary)
    assert fake_judge_sdk.calls == 0


# ---------------------------------------------------------------------------------------- ablations


ABLATION_CASES = [
    (task_id, name) for task_id in TASK_IDS for name in ["gold", *expected_of(task_id)["ablations"]]
]


@pytest.mark.parametrize(
    ("task_id", "name"), ABLATION_CASES, ids=[f"{t[-12:]}-{n}" for t, n in ABLATION_CASES]
)
def test_each_ablation_reproduces_its_expected_outcome(task_id: str, name: str, tmp_path: Path) -> None:
    """A scripted judge returns exactly the verdicts expected.json lists; the grade must follow from them."""
    task_dir = EXAMPLES / task_id
    expected_all = expected_of(task_id)
    if name == "gold":
        expected = expected_all["gold"]
        ev.solve("oracle", task_dir, tmp_path, ledger=None, tag="gold")
    else:
        expected = expected_all["ablations"][name]
        shutil.copy(ABLATIONS / task_id / name, tmp_path / "answer.json")
    # The scripted judge quotes the rationale's first sentence; with no rationale (F5's core criteria
    # are met by the answer fields alone) it quotes the answer object, which the response shows verbatim.
    envelope = read_json(tmp_path / "answer.json")
    evidence = None if envelope["rationale"] else json.dumps(envelope["answer"], indent=2, ensure_ascii=False)
    judge = ScriptedJudge(
        expected["criteria"], consistency=CONSISTENCY[expected["consistency"]], evidence=evidence
    )

    result = ev.grade_answer(task_dir, tmp_path, judges=(judge, None))

    assert result["passed"] is expected["pass"]
    assert result["s_core"] == pytest.approx(expected["s_core"], abs=0.005)
    assert result["s_judge"] == pytest.approx(expected["s_judge"], abs=0.005)
    assert result["s"] == pytest.approx(expected["s"], abs=0.005)
    verdicts = {c["id"]: c["verdict"] for c in result["criteria"]}
    assert verdicts == expected["criteria"], "a scripted yes was not credited (evidence check or masking)"


# --------------------------------------------------------------------------------------- model agent


def test_a_model_agent_writes_its_text_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gold = read_json(EXAMPLES / TASK_IDS[1] / "tests" / "gold.json")
    text = "  " + json.dumps({"answer": gold["answer"], "rationale": gold["rationale"]}, indent=3) + "\n\n"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(llm, "complete", fake_complete(text, calls=calls))
    ledger = Ledger(None)

    summary = ev.evaluate(TASKS[1:2], [SOLVER], tmp_path, judges=(yes_judge(), None), ledger=ledger)

    trial = tmp_path / TASK_IDS[1] / "claude-sonnet-5-5" / "01"
    assert (trial / "answer.json").read_bytes() == text.encode("utf-8")
    assert summary["rows"][0]["pass_rate"] == 1.0
    [call] = calls
    assert call["messages"] == ev.solver_prompt(EXAMPLES / TASK_IDS[1])
    assert call["role"] == "solver" and call["ledger"] is ledger
    assert call["tag"] == f"{TASK_IDS[1]}/claude-sonnet-5-5/01"
    assert "system" not in call and "schema" not in call
    solver = read_json(trial / "solver.json")
    assert solver["refusal"] is False and solver["input_tokens"] == 1200
    assert "text" not in solver


def test_the_instruction_is_sent_without_its_canary() -> None:
    for task in TASKS:
        prompt = ev.solver_prompt(task)
        assert "p2c-canary" not in prompt
        assert prompt.startswith("# Background")
        assert (task / "instruction.md").read_text(encoding="utf-8").endswith(prompt)


def test_the_instruction_matches_what_harbor_shows_an_agent() -> None:
    task_module = pytest.importorskip("harbor.models.task.task")
    for task in TASKS:
        assert ev.solver_prompt(task) == task_module.Task(task).instruction


@pytest.mark.parametrize("text", ["I can't help with that request.", ""])
def test_a_refusal_is_graded_zero_not_dropped(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "complete", fake_complete(text, refusal=True))
    judge = yes_judge()
    summary = ev.evaluate(TASKS[:1], [SOLVER], tmp_path, judges=(judge, None))

    trial = tmp_path / TASK_IDS[0] / "claude-sonnet-5-5" / "01"
    assert (trial / "answer.json").read_text(encoding="utf-8") == text
    assert read_json(trial / "grading.json")["status"] == "invalid_answer"
    assert read_json(trial / "solver.json")["refusal"] is True
    row = summary["rows"][0]
    assert (row["n_valid"], row["n_invalid"], row["n_refusals"], row["mean_s"]) == (1, 0, 1, 0.0)
    assert judge.calls == []


def test_a_failed_model_call_is_an_invalid_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> Completion:
        raise llm.LLMError("anthropic claude-sonnet-5-5 call for role 'solver' failed: overloaded")

    monkeypatch.setattr(llm, "complete", failing)
    summary = ev.evaluate(TASKS[:1], [SOLVER, "oracle"], tmp_path, judges=(yes_judge(), None))

    model_row, oracle_row = summary["rows"]
    assert (model_row["n_valid"], model_row["n_invalid"], model_row["mean_s"]) == (0, 1, None)
    assert oracle_row["n_valid"] == 1
    trial = tmp_path / TASK_IDS[0] / "claude-sonnet-5-5" / "01"
    assert "overloaded" in (trial / "error.txt").read_text(encoding="utf-8")
    assert not (trial / "reward.json").exists()


def test_a_judge_failure_is_an_invalid_trial(tmp_path: Path) -> None:
    summary = ev.evaluate(
        TASKS[:1], ["oracle"], tmp_path, judging=ev.JudgeConfig(retries=0), judges=(BrokenJudge(), None)
    )
    assert (summary["rows"][0]["n_valid"], summary["rows"][0]["n_invalid"]) == (0, 1)
    trial = tmp_path / TASK_IDS[0] / "oracle" / "01"
    assert "GradingError" in (trial / "error.txt").read_text(encoding="utf-8")
    assert not (trial / "reward.json").exists()


def test_a_judge_failure_via_test_sh_is_an_invalid_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_judge_sdk: FakeJudgeSDK
) -> None:
    monkeypatch.setenv("P2C_FAKE_DOWN", "1")  # every judge call raises, so grade.py exits 2
    judging = ev.JudgeConfig(regrade_model=None, retries=0, samples=1)
    summary = ev.evaluate(TASKS[:1], ["oracle"], tmp_path / "out", judging=judging, via="test_sh")
    assert summary["rows"][0]["n_invalid"] == 1
    assert not (tmp_path / "out" / TASK_IDS[0] / "oracle" / "01" / "reward.json").exists()
    error = (tmp_path / "out" / TASK_IDS[0] / "oracle" / "01" / "error.txt").read_text(encoding="utf-8")
    assert "GradingError" in error and "judge failure" in error


# -------------------------------------------------------------------------------------- judge spend


def metered_judge_factory(judge: ScriptedJudge) -> Any:
    """A stand-in for `grade.make_judge` that hands `judge` the on_usage callback it is built with."""

    def make_judge(model: str, effort: str | None = None, on_usage: Any | None = None) -> ScriptedJudge:
        return ScriptedJudge(
            judge.verdicts, name=model, on_usage=on_usage, tokens=judge.tokens, evidence=judge.evidence
        )

    return make_judge


SONNET = RoleSpec(
    provider="anthropic", model="claude-sonnet-5-5", price_per_mtok=llm.Price(input=2.0, output=10.0)
)
GPT = RoleSpec(provider="openai", model="gpt-6.1-sol", price_per_mtok=llm.Price(input=2.0, output=10.0))
PRICED_JUDGING = ev.JudgeConfig(
    model=SONNET.model,
    regrade_model=GPT.model,
    specs={ev.JUDGE_ROLE: SONNET, ev.JUDGE_REGRADE_ROLE: GPT},
)


def test_every_judge_call_reaches_the_ledger_and_grading_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judging is most of the cost, so a run's ledger has to carry it as well as its solver calls."""
    monkeypatch.setattr(g, "make_judge", metered_judge_factory(yes_judge()))
    ledger = Ledger(tmp_path / "calls.jsonl")

    summary = ev.evaluate(TASKS[:1], ["oracle"], tmp_path / "out", judging=PRICED_JUDGING, ledger=ledger)

    assert summary["rows"][0]["mean_s"] == 1.0
    calls = 3 * (criteria_count(TASKS[0]) + 1)
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(records) == calls
    assert {record["role"] for record in records} == {"judge"}  # the regrade judge is never reached
    assert {record["model_requested"] for record in records} == {"claude-sonnet-5-5"}
    assert {record["tag"] for record in records} == {f"{TASK_IDS[0]}/oracle/01"}
    one = (1000 * 2.0 + 40 * 10.0) / 1e6  # ScriptedJudge's default tokens at the roster's prices
    assert ledger.spent(role="judge") == pytest.approx(calls * one)
    usage = read_json(tmp_path / "out" / TASK_IDS[0] / "oracle" / "01" / "grading.json")["judge_usage"]
    assert usage == {
        "claude-sonnet-5-5": {
            "calls": calls, "input_tokens": 1000 * calls, "output_tokens": 40 * calls, "refusals": 0
        }
    }


def test_a_regrade_judge_call_is_recorded_under_its_own_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_judge(model: str, effort: str | None = None, on_usage: Any | None = None) -> Any:
        if model == GPT.model:
            return ScriptedJudge(defaultdict(lambda: "yes"), name=model, on_usage=on_usage)
        return RefusingJudge(name=model)  # refuses without reporting usage, as a failed call does

    monkeypatch.setattr(g, "make_judge", make_judge)
    ledger = Ledger(tmp_path / "calls.jsonl")
    ev.evaluate(TASKS[:1], ["oracle"], tmp_path / "out", judging=PRICED_JUDGING, ledger=ledger)

    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert {record["role"] for record in records} == {"judge_regrade"}
    assert {record["provider"] for record in records} == {"openai"}
    assert ledger.spent(role="judge") == 0.0 and ledger.spent(role="judge_regrade") > 0.0


def test_a_judge_model_the_roster_does_not_price_is_still_recorded() -> None:
    record = ev.JudgeConfig(model="claude-sonnet-5-5", regrade_model=None).call_record(
        "claude-sonnet-5-5", 1000, 40, tag="t/oracle/01"
    )
    assert (record.role, record.provider, record.cost_usd) == ("judge", "anthropic", None)
    assert PRICED_JUDGING.call_record(GPT.model, 1000, 40, True).refusal is True
    assert PRICED_JUDGING.role_of(GPT.model) == "judge_regrade"
    assert PRICED_JUDGING.role_of("some-other-model") == "judge"


def test_the_budget_is_checked_before_a_trial_is_graded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap has to stop the run before the judging, which is where the money goes."""
    monkeypatch.setattr(g, "make_judge", metered_judge_factory(yes_judge()))
    ledger = Ledger(tmp_path / "calls.jsonl", budget_usd=0.001)
    ledger.record(ledger_record_worth(0.002))

    with pytest.raises(BudgetExceeded):
        ev.evaluate(TASKS[:1], ["oracle"], tmp_path / "out", judging=PRICED_JUDGING, ledger=ledger)

    assert not (tmp_path / "out" / TASK_IDS[0] / "oracle" / "01" / "reward.json").exists()
    assert ledger.spent(role="judge") == 0.0


def ledger_record_worth(cost_usd: float) -> CallRecord:
    return CallRecord(
        role="solver", provider="anthropic", model_requested="claude-sonnet-5-5", cost_usd=cost_usd
    )


# ------------------------------------------------------------------------------------------- resume


def test_a_finished_trial_is_not_solved_or_graded_again(tmp_path: Path) -> None:
    first = ev.evaluate(TASKS[:1], ["oracle"], tmp_path, judges=(yes_judge(), None))
    trial = tmp_path / TASK_IDS[0] / "oracle" / "01"
    answer = (trial / "answer.json").read_bytes()
    (trial / "answer.json").write_bytes(answer + b"\n")  # a re-solve would overwrite this

    judge = BrokenJudge()  # grading again would make the trial invalid
    second = ev.evaluate(TASKS[:1], ["oracle"], tmp_path, judges=(judge, None))

    row = second["rows"][0]
    assert (row["n_resumed"], row["n_valid"], row["n_invalid"]) == (1, 1, 0)
    assert row["mean_s"] == first["rows"][0]["mean_s"] == 1.0
    assert judge.calls == 0
    assert (trial / "answer.json").read_bytes() == answer + b"\n"
    assert first["rows"][0]["n_resumed"] == 0


def test_a_half_written_trial_is_run_again(tmp_path: Path) -> None:
    trial = tmp_path / TASK_IDS[0] / "oracle" / "01"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text('{"reward": 0.5}\n', encoding="utf-8")  # no grading.json beside it

    summary = ev.evaluate(TASKS[:1], ["oracle"], tmp_path, judges=(yes_judge(), None))

    assert summary["rows"][0]["n_resumed"] == 0
    assert read_json(trial / "reward.json") == {"reward": 1.0, "core": 1.0, "judged": 1.0}


def test_a_solved_but_ungraded_model_trial_is_graded_without_a_second_solver_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = read_json(EXAMPLES / TASK_IDS[0] / "tests" / "gold.json")
    text = json.dumps({"answer": gold["answer"], "rationale": gold["rationale"]})
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(llm, "complete", fake_complete(text, refusal=True, calls=calls))
    broken = (BrokenJudge(), None)
    ev.evaluate(TASKS[:1], [SOLVER], tmp_path, judging=ev.JudgeConfig(retries=0), judges=broken)
    trial = tmp_path / TASK_IDS[0] / "claude-sonnet-5-5" / "01"
    assert (trial / "error.txt").exists() and not (trial / "reward.json").exists()

    summary = ev.evaluate(TASKS[:1], [SOLVER], tmp_path, judges=(yes_judge(), None))

    assert len(calls) == 1, "the stored answer is graded; the solver is not paid twice"
    row = summary["rows"][0]
    assert (row["n_valid"], row["n_resumed"], row["n_refusals"]) == (1, 0, 1)
    assert row["mean_s"] == 1.0


def test_relative_task_and_output_folders_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The oracle's solve.sh runs inside its trial folder, so the runner resolves every path first."""
    shutil.copytree(EXAMPLES / TASK_IDS[0], tmp_path / "tasks" / TASK_IDS[0])
    monkeypatch.chdir(tmp_path)
    summary = ev.evaluate([Path("tasks") / TASK_IDS[0]], ["oracle"], Path("out"), judges=(yes_judge(), None))
    assert summary["rows"][0]["mean_s"] == 1.0
    grading = ev.grade_answer(
        Path("tasks") / TASK_IDS[0], Path("out") / TASK_IDS[0] / "oracle" / "01", judges=(yes_judge(), None)
    )
    assert grading["s"] == 1.0


def test_a_rerun_trial_drops_the_previous_attempts_error(tmp_path: Path) -> None:
    summary = ev.evaluate(
        TASKS[:1], ["oracle"], tmp_path, judging=ev.JudgeConfig(retries=0), judges=(BrokenJudge(), None)
    )
    trial = tmp_path / TASK_IDS[0] / "oracle" / "01"
    assert (summary["rows"][0]["n_invalid"], (trial / "error.txt").exists()) == (1, True)

    summary = ev.evaluate(TASKS[:1], ["oracle"], tmp_path, judges=(yes_judge(), None))

    assert (summary["rows"][0]["n_valid"], summary["rows"][0]["n_resumed"]) == (1, 0)
    assert not (trial / "error.txt").exists(), "a stale error must not outlive the trial it describes"
    assert read_json(trial / "reward.json") == {"reward": 1.0, "core": 1.0, "judged": 1.0}


# ------------------------------------------------------------------------------------------- parity


PARITY_CASES = [
    (task_id, answer)
    for task_id in TASK_IDS
    for answer in ("gold_paraphrase.json", "textbook_prior.json", None)
]


@pytest.mark.parametrize(
    ("task_id", "answer"), PARITY_CASES, ids=[f"{t[-12:]}-{a or 'no-answer'}" for t, a in PARITY_CASES]
)
def test_in_process_and_test_sh_grade_a_stored_answer_identically(
    task_id: str, answer: str | None, tmp_path: Path, fake_judge_sdk: FakeJudgeSDK
) -> None:
    trials = {via: tmp_path / via for via in ("in_process", "test_sh")}
    for via, trial in trials.items():
        trial.mkdir()
        if answer is not None:
            shutil.copy(ABLATIONS / task_id / answer, trial / "answer.json")
        ev.grade_answer(EXAMPLES / task_id, trial, via=via)

    in_process, test_sh = trials["in_process"], trials["test_sh"]
    assert (in_process / "reward.json").read_bytes() == (test_sh / "reward.json").read_bytes()
    assert read_json(in_process / "grading.json") == read_json(test_sh / "grading.json")


# ------------------------------------------------------------------------------------------ settings


def test_judge_config_comes_from_the_roster_and_the_grading_stage() -> None:
    models = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "models" / "default.yaml"), resolve=True
    )
    stages = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "stages" / "default.yaml"), resolve=True
    )
    roster = llm.load_roster(models)
    judging = ev.JudgeConfig.from_config(roster, stages["grading"])
    assert judging == ev.JudgeConfig(
        model="claude-sonnet-5-5",
        regrade_model="gpt-6.1-sol",
        effort="medium",
        samples=3,
        retries=2,
        concurrency=8,
        specs={"judge": roster["judge"], "judge_regrade": roster["judge_regrade"]},
    )
    # The roster entries are what a judge call is costed at in the ledger.
    assert judging.specs["judge"].price_per_mtok.input == 2.0
    assert judging.env() == {
        "JUDGE_MODEL": "claude-sonnet-5-5",
        "JUDGE_REGRADE_MODEL": "gpt-6.1-sol",
        "JUDGE_EFFORT": "medium",
        "JUDGE_SAMPLES": "3",
        "JUDGE_RETRIES": "2",
        "JUDGE_CONCURRENCY": "8",
    }


def test_the_default_judges_are_the_real_ones_unbuilt() -> None:
    judge, regrade = ev.JudgeConfig().build()
    assert isinstance(judge, g.AnthropicJudge) and isinstance(regrade, g.OpenAIJudge)
    assert judge._client is None  # building a judge calls nothing
    assert ev.JudgeConfig(regrade_model=None).build()[1] is None


def test_agents_need_distinct_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="share a trial folder"):
        ev.evaluate(TASKS, [SOLVER, SOLVER.model_copy(update={"effort": "low"})], tmp_path)
    with pytest.raises(ValueError, match="neither"):
        ev.evaluate(TASKS, ["random"], tmp_path)


def test_injected_judges_cannot_reach_test_sh(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test.sh"):
        ev.grade_answer(TASKS[0], tmp_path, via="test_sh", judges=(yes_judge(), None))
