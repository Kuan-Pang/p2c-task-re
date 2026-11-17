"""M0 live check: the three hand-built tasks through the real grader (PLAN §11 M0, §9 sanity runs).

For each example task, three answers are graded in-process with the real judges — claude-sonnet-5-5 at
3 samples per criterion, gpt-6.1-sol configured as the regrade judge: the oracle (solution/solve.sh, run
through `evaluate`), and the gold_paraphrase and correct_core_wrong_mechanism ablations. Each result is
compared with tests/fixtures/ablations/<task_id>/expected.json; every criterion whose majority verdict
differs is printed with each vote's verdict and evidence, and all of it lands in live_report.json.

Spend is counted twice over, and the two have to agree: the runner records every judge call in the run's
ledger at the roster's prices, and a `Meter` patched onto both SDKs' create methods counts the same calls
from the responses themselves. BUDGET_USD is the ledger's cap, and a phase is skipped if it would take the
run past it. Run:

    source ~/.api_keys && uv run pytest -m live tests/live/test_evaluate_live.py -q -s
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from p2c_task_re import evaluate as ev
from p2c_task_re import llm
from p2c_task_re.ledger import Ledger

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "tasks"
ABLATIONS = REPO / "tests" / "fixtures" / "ablations"
TASKS = ev.find_tasks(EXAMPLES)
MODELS = OmegaConf.to_container(OmegaConf.load(REPO / "configs" / "models" / "default.yaml"), resolve=True)
ROSTER = llm.load_roster(MODELS)
PRICES = {MODELS[name]["model"]: MODELS[name]["price_per_mtok"] for name in ("sonnet", "gpt")}
# The roster's own judge roles, so a judge call is costed in the ledger exactly as a solver call is.
JUDGING = ev.JudgeConfig(
    model=ROSTER["judge"].model,
    regrade_model=ROSTER["judge_regrade"].model,
    samples=3,
    specs={ev.JUDGE_ROLE: ROSTER["judge"], ev.JUDGE_REGRADE_ROLE: ROSTER["judge_regrade"]},
)
BUDGET_USD = 2.0
PHASES = ("oracle", "gold_paraphrase.json", "correct_core_wrong_mechanism.json")


class Meter:
    """Token usage per model, read off every judge response; cost at the roster's per-Mtok prices."""

    def __init__(self, prices: dict[str, dict[str, float | None]]) -> None:
        self.prices = prices
        self.tokens: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            counts = self.tokens.setdefault(model, [0, 0, 0])
            counts[0] += 1
            counts[1] += input_tokens
            counts[2] += output_tokens

    def cost(self) -> float:
        """Known cost; a model without a price counts as 0 and shows up in `report()`."""
        total = 0.0
        for model, (_, tokens_in, tokens_out) in self.tokens.items():
            price = self.prices.get(model, {})
            if price.get("input") is not None and price.get("output") is not None:
                total += (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6
        return total

    def report(self) -> dict[str, Any]:
        return {
            "cost_usd": round(self.cost(), 4),
            "calls": {
                m: {"calls": c, "input_tokens": i, "output_tokens": o} for m, (c, i, o) in self.tokens.items()
            },
        }


@pytest.fixture(scope="module")
def meter() -> Iterator[Meter]:
    from anthropic.resources.messages import Messages
    from openai.resources.responses import Responses

    meter = Meter(PRICES)
    messages_create, responses_create = Messages.create, Responses.create

    def metered_messages(self: Any, *args: Any, **kwargs: Any) -> Any:
        message = messages_create(self, *args, **kwargs)
        meter.add(kwargs["model"], message.usage.input_tokens, message.usage.output_tokens)
        return message

    def metered_responses(self: Any, *args: Any, **kwargs: Any) -> Any:
        response = responses_create(self, *args, **kwargs)
        usage = response.usage
        meter.add(kwargs["model"], usage.input_tokens if usage else 0, usage.output_tokens if usage else 0)
        return response

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Messages, "create", metered_messages)
        patch.setattr(Responses, "create", metered_responses)
        yield meter


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("evaluate_live")


@pytest.fixture(scope="module")
def ledger(out: Path) -> Ledger:
    """The run's ledger: every judge call lands here, and BUDGET_USD is its hard stop."""
    return Ledger(out / "calls.jsonl", budget_usd=BUDGET_USD)


def ledger_spend(ledger: Ledger) -> dict[str, Any]:
    """What the ledger recorded, per model: role, calls, tokens and cost."""
    per_model: dict[str, dict[str, Any]] = {}
    lines = ledger.path.read_text(encoding="utf-8").splitlines() if ledger.path.exists() else []
    for line in lines:
        call = json.loads(line)
        counts = per_model.setdefault(
            call["model_requested"],
            {"role": call["role"], "calls": 0, "input_tokens": 0, "output_tokens": 0, "refusals": 0},
        )
        counts["calls"] += 1
        counts["input_tokens"] += call["input_tokens"]
        counts["output_tokens"] += call["output_tokens"]
        counts["refusals"] += int(call["refusal"])
    return {
        "cost_usd": round(ledger.spent(), 4),
        "judge_cost_usd": round(ledger.spent(role=ev.JUDGE_ROLE), 4),
        "judge_regrade_cost_usd": round(ledger.spent(role=ev.JUDGE_REGRADE_ROLE), 4),
        "per_model": per_model,
    }


@pytest.fixture(scope="module")
def report(out: Path, meter: Meter, ledger: Ledger) -> Iterator[dict[str, Any]]:
    collected: dict[str, Any] = {"out_dir": str(out), "phases": {}}
    yield collected
    collected["spend"] = {"sdk_metered": meter.report(), "ledger": ledger_spend(ledger)}
    (out / "live_report.json").write_text(json.dumps(collected, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nspend, metered off the SDK responses: {json.dumps(meter.report())}"
        f"\nspend, as the ledger recorded it:      {json.dumps(ledger_spend(ledger))}"
        f"\nreport: {out / 'live_report.json'}"
    )


def compare(task_id: str, answer: str, result: dict[str, Any]) -> dict[str, Any]:
    """The real grade next to expected.json, with every criterion the real judge decided differently."""
    expected_all = json.loads((ABLATIONS / task_id / "expected.json").read_text(encoding="utf-8"))
    expected = expected_all["gold"] if answer == "oracle" else expected_all["ablations"][answer]
    disagreements = []
    for criterion in result["criteria"]:
        want = expected["criteria"][criterion["id"]]
        if criterion["verdict"] != want:
            votes = [
                {key: vote[key] for key in ("model", "raw_verdict", "verdict", "reason", "evidence")}
                for vote in criterion["votes"]
            ]
            disagreements.append(
                {"id": criterion["id"], "expected": want, "got": criterion["verdict"], "votes": votes}
            )
    want_contradiction = expected["consistency"] == "contradiction"
    return {
        "task_id": task_id,
        "answer": answer,
        "status": result["status"],
        "answer_format": result["answer_format"],
        "judge_usage": result["judge_usage"],
        "real": {key: result[key] for key in ("s", "s_core", "s_judge", "passed")},
        "expected": {
            "s": expected["s"],
            "s_core": expected["s_core"],
            "s_judge": expected["s_judge"],
            "passed": expected["pass"],
        },
        "consistency_agrees": result["consistency"]["contradiction"] == want_contradiction,
        "disagreements": disagreements,
    }


def print_row(row: dict[str, Any]) -> None:
    real, want = row["real"], row["expected"]
    print(
        f"\n{row['task_id']} {row['answer']}: status={row['status']} answer={row['answer_format']} "
        f"s={real['s']:.3f} (exp {want['s']:.2f}) core={real['s_core']:.3f} (exp {want['s_core']:.2f}) "
        f"judged={real['s_judge']:.3f} (exp {want['s_judge']:.2f}) "
        f"pass={real['passed']} (exp {want['passed']})"
        f"\n  judge usage: {json.dumps(row['judge_usage'])}"
    )
    for item in row["disagreements"]:
        print(f"  {item['id']}: expected {item['expected']}, real judge {item['got']}")
        for vote in item["votes"]:
            print(
                f"    {vote['model']} raw={vote['raw_verdict']} counted={vote['verdict']} "
                f"reason={vote['reason']}"
            )
            if vote["evidence"]:
                print(f"      evidence: {vote['evidence']}")
    if not row["consistency_agrees"]:
        print("  consistency check disagrees with expected.json")


@pytest.mark.parametrize("answer", PHASES)
def test_the_examples_grade_as_expected_with_the_real_judge(
    answer: str, meter: Meter, ledger: Ledger, report: dict[str, Any]
) -> None:
    phase_costs = [phase["cost_usd"] for phase in report["phases"].values()]
    projected = max(phase_costs, default=0.0)
    if meter.cost() + projected > BUDGET_USD:
        pytest.skip(f"spent ${meter.cost():.2f}; another phase (~${projected:.2f}) would pass ${BUDGET_USD}")
    out = Path(report["out_dir"]) / answer.removesuffix(".json")
    spent_before = meter.cost()

    if answer == "oracle":
        ev.evaluate(TASKS, ["oracle"], out, judging=JUDGING, ledger=ledger, concurrency=3)
        trials = {task.name: out / task.name / "oracle" / "01" for task in TASKS}
    else:
        trials = {}
        for task in TASKS:
            trial = out / task.name
            trial.mkdir(parents=True)
            shutil.copy(ABLATIONS / task.name / answer, trial / "answer.json")
            ev.grade_answer(task, trial, judging=JUDGING, ledger=ledger, tag=f"{task.name}/{answer}")
            trials[task.name] = trial

    rows = []
    for task_id, trial in trials.items():
        result = json.loads((trial / "grading.json").read_text(encoding="utf-8"))
        rows.append(compare(task_id, answer, result))
        print_row(rows[-1])
    report["phases"][answer] = {"cost_usd": round(meter.cost() - spent_before, 4), "rows": rows}
    print(
        f"phase {answer}: ${meter.cost() - spent_before:.3f} metered "
        f"(total ${meter.cost():.3f} metered, ${ledger.spent():.3f} in the ledger)"
    )

    if answer == "oracle":  # PLAN §9: the oracle must give core = 1.0 and reward >= 0.9
        assert all(row["real"]["s_core"] == 1.0 and row["real"]["s"] >= 0.9 for row in rows), rows
    assert [row["real"]["passed"] for row in rows] == [row["expected"]["passed"] for row in rows]
    # Every judge call the SDKs served is a ledger line, priced the same way.
    assert ledger.spent() == pytest.approx(meter.cost(), abs=0.005), ledger_spend(ledger)
    assert all(row["judge_usage"] for row in rows), "a graded answer must report its judge spend"
