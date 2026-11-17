"""M0 live check: the three examples through the real `harbor` CLI with LocalEnvironment (PLAN §9).

One run, in this order, on examples/tasks itself:

    harbor run -a oracle             every task: core = 1.0 and reward >= 0.9
    harbor run -a nop                every task: a valid 0
    harbor job regrade               the nop job (still 0) and the oracle job (Harbor vs Harbor)
    harbor run -a SingleShotAgent    claude-sonnet-5-5 on the F2 task, k = 1
    in-process parity                the same stored answers graded by evaluate.grade_answer:
                                     core exact, |Δreward| <= 0.02 (PLAN §3)

The judge is claude-sonnet-5-5 at JUDGE_SAMPLES=1 throughout, to stay inside BUDGET_USD; gpt-6.1-sol is
the regrade judge but only fires on a refusal or a failed sample. Judge spend is metered by a sitecustomize
hook on PYTHONPATH that logs every SDK response's token usage, in Harbor's verifier subprocesses and in this
process alike, and the `judge_usage` every grading.json reports (Harbor's and the in-process runner's)
must add up to exactly those calls. The solver's cost is the agent's own record in result.json; its
streamed call does not pass through the hooked `create`. Everything lands in <tmp>/live_report.json and is
printed. Run:

    source ~/.api_keys && uv run pytest -m live tests/live/test_harbor_live.py -q -s
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anthropic.resources.messages import Messages
from omegaconf import OmegaConf
from openai.resources.responses import Responses

pytest.importorskip("harbor")  # before importing harbor_env, so a plain `uv sync` still collects the suite

from p2c_task_re import evaluate as ev  # noqa: E402
from p2c_task_re.harbor_env import VENV_BIN  # noqa: E402

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "tasks"
TASKS = ev.find_tasks(EXAMPLES)
F2 = next(task for task in TASKS if "-f2-" in task.name)
LOCAL = "p2c_task_re.harbor_env:LocalEnvironment"
AGENT = "p2c_task_re.harbor_agent:SingleShotAgent"
SOLVER = "claude-sonnet-5-5"
SAMPLES = ["--ve", "JUDGE_SAMPLES=1"]
JUDGING = ev.JudgeConfig(model="claude-sonnet-5-5", regrade_model="gpt-6.1-sol", samples=1)
BUDGET_USD = 1.0
CALL_USD = 0.008  # one judge call, measured in the in-process live run; for the budget check only

USAGE_HOOK = '''\
"""Log the token usage of every judge response to $P2C_USAGE_LOG (one JSON line per call)."""
import functools
import json
import os


def _logged(create, provider):
    @functools.wraps(create)
    def wrapper(self, *args, **kwargs):
        response = create(self, *args, **kwargs)
        usage = response.usage
        line = {"provider": provider, "model": response.model,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
        with open(os.environ["P2C_USAGE_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps(line) + "\\n")
        return response
    return wrapper


if os.environ.get("P2C_USAGE_LOG"):
    from anthropic.resources.messages import Messages
    from openai.resources.responses import Responses

    Messages.create = _logged(Messages.create, "anthropic")
    Responses.create = _logged(Responses.create, "openai")
'''


def prices() -> dict[str, dict[str, float | None]]:
    models = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "models" / "default.yaml"), resolve=True
    )
    return {entry["model"]: entry["price_per_mtok"] for entry in (models["sonnet"], models["gpt"])}


def judge_spend(usage_log: Path) -> dict[str, Any]:
    """Calls, tokens and cost per model from the usage log; a model without a price adds no cost."""
    per_model: dict[str, dict[str, int]] = {}
    lines = usage_log.read_text(encoding="utf-8").splitlines() if usage_log.exists() else []
    for line in lines:
        call = json.loads(line)
        counts = per_model.setdefault(call["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        counts["calls"] += 1
        counts["input_tokens"] += call["input_tokens"]
        counts["output_tokens"] += call["output_tokens"]
    return {"cost_usd": cost_usd(per_model), "per_model": per_model}


def cost_usd(per_model: dict[str, dict[str, int]]) -> float:
    cost = 0.0
    for model, counts in per_model.items():
        price = next((p for name, p in prices().items() if model.startswith(name)), {})
        if price.get("input") is not None and price.get("output") is not None:
            cost += (
                counts["input_tokens"] * price["input"] + counts["output_tokens"] * price["output"]
            ) / 1e6
    return round(cost, 4)


def add_usage(total: dict[str, dict[str, int]], judge_usage: dict[str, dict[str, int]]) -> None:
    """Add one grading.json's `judge_usage` into `total` (calls and tokens per model)."""
    for model, counts in judge_usage.items():
        into = total.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        for key in into:
            into[key] += counts[key]


def trials_of(job_dir: Path) -> dict[str, dict[str, Any]]:
    """task_id -> {rewards, exception, answer path, agent result, grading summary} for every trial."""
    trials = {}
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        grading_path = result_path.parent / "verifier" / "grading.json"
        grading = json.loads(grading_path.read_text(encoding="utf-8")) if grading_path.exists() else {}
        trials[result["task_name"].removeprefix("p2c/")] = {
            "rewards": (result["verifier_result"] or {}).get("rewards"),
            "exception": result["exception_info"],
            "answer": result_path.parent / "artifacts" / "app" / "output" / "answer.json",
            "agent_result": result["agent_result"],
            "status": grading.get("status"),
            "answer_format": grading.get("answer_format"),
            "judge_usage": grading.get("judge_usage"),
        }
    return trials


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    if not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY")):
        pytest.skip("source ~/.api_keys first")
    root = tmp_path_factory.mktemp("harbor_live")
    hook = root / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(USAGE_HOOK, encoding="utf-8")
    usage_log = root / "judge_usage.jsonl"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("P2C_USAGE_LOG", str(usage_log))
        # Meter this process's in-process grading too; re-setting the originals first makes the context
        # put them back afterwards.
        patch.setattr(Messages, "create", Messages.create)
        patch.setattr(Responses, "create", Responses.create)
        runpy.run_path(str(hook / "sitecustomize.py"))
        yield run_everything(root, usage_log, {**os.environ, "PYTHONPATH": str(hook)})


def run_everything(root: Path, usage_log: Path, env: dict[str, str]) -> dict[str, Any]:
    dataset, jobs = EXAMPLES, root / "jobs"
    commands: list[str] = []

    def harbor(*args: str | Path, cost_estimate: float = 0.0) -> None:
        spent = judge_spend(usage_log)["cost_usd"]
        if spent + cost_estimate > BUDGET_USD:
            pytest.fail(f"stopping before `harbor {args[0]}`: ${spent:.2f} spent, ~${cost_estimate:.2f} more")
        command = ["harbor", *map(str, args)]
        commands.append(" ".join(command))
        proc = subprocess.run(
            [str(VENV_BIN / "harbor"), *command[1:]], env=env, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    criteria = sum(len(json.loads((t / "tests" / "rubric.json").read_text())["criteria"]) for t in TASKS)
    oracle_usd = (criteria + len(TASKS)) * CALL_USD  # one sample per criterion, one consistency check
    common = ["-e", LOCAL, *SAMPLES, "-o", jobs, "-y"]
    regrade = ["-p", dataset, "-e", LOCAL, *SAMPLES, "-o", jobs]
    single = ["-p", dataset / F2.name, "-a", AGENT, "-m", SOLVER, "-k", "1", "-r", "3", *common]
    harbor("run", "-p", dataset, "-a", "oracle", *common, "--job-name", "oracle", cost_estimate=oracle_usd)
    harbor("run", "-p", dataset, "-a", "nop", *common, "--job-name", "nop")
    harbor("job", "regrade", jobs / "nop", *regrade, "--job-name", "nop-regrade")
    harbor("job", "regrade", jobs / "oracle", *regrade, "--job-name", "oracle-regrade", cost_estimate=0.2)
    harbor("run", *single, "--job-name", "single-shot", cost_estimate=0.15)

    report: dict[str, Any] = {"commands": commands, "jobs": {}}
    for job in ("oracle", "nop", "nop-regrade", "oracle-regrade", "single-shot"):
        report["jobs"][job] = trials_of(jobs / job)

    # Parity (PLAN §3): the very answers Harbor graded, graded again by the in-process runner.
    parity = {}
    graded_usage: dict[str, dict[str, int]] = {}  # every grading.json's judge_usage, added up
    for trials in report["jobs"].values():
        for trial in trials.values():
            add_usage(graded_usage, trial["judge_usage"] or {})
    for job in ("oracle", "single-shot"):
        for task_id, trial in report["jobs"][job].items():
            trial_dir = root / "in_process" / job / task_id
            trial_dir.mkdir(parents=True)
            shutil.copy(trial["answer"], trial_dir / "answer.json")
            result = ev.grade_answer(dataset / task_id, trial_dir, judging=JUDGING, via="in_process")
            add_usage(graded_usage, result["judge_usage"])
            parity[f"{job}/{task_id}"] = {
                "harbor": trial["rewards"],
                "in_process": {"reward": result["s"], "core": result["s_core"], "judged": result["s_judge"]},
                "harbor_judge_usage": trial["judge_usage"],
                "in_process_judge_usage": result["judge_usage"],
            }
    report["parity"] = parity
    report["judge_spend"] = judge_spend(usage_log)
    report["graded_judge_usage"] = {"cost_usd": cost_usd(graded_usage), "per_model": graded_usage}
    solver_cost = report["jobs"]["single-shot"][F2.name]["agent_result"]["cost_usd"] or 0.0
    report["total_cost_usd"] = round(report["judge_spend"]["cost_usd"] + solver_cost, 4)
    (root / "live_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


def test_the_oracle_passes_every_task(live: dict[str, Any]) -> None:
    for trial in live["jobs"]["oracle"].values():
        assert trial["exception"] is None
        assert trial["rewards"]["core"] == 1.0
        assert trial["rewards"]["reward"] >= 0.9


def test_nop_is_a_valid_zero_in_a_run_and_a_regrade(live: dict[str, Any]) -> None:
    zero = {"reward": 0.0, "core": 0.0, "judged": 0.0}
    for job in ("nop", "nop-regrade"):
        assert len(live["jobs"][job]) == len(TASKS)
        for trial in live["jobs"][job].values():
            assert (trial["exception"], trial["rewards"], trial["status"]) == (None, zero, "invalid_answer")


def test_the_oracle_regrade_agrees_with_the_run(live: dict[str, Any]) -> None:
    for task_id, trial in live["jobs"]["oracle-regrade"].items():
        run = live["jobs"]["oracle"][task_id]["rewards"]
        assert trial["rewards"]["core"] == run["core"]
        assert abs(trial["rewards"]["reward"] - run["reward"]) <= 0.02


def test_the_single_shot_agent_answers_and_is_graded(live: dict[str, Any]) -> None:
    trial = live["jobs"]["single-shot"][F2.name]
    assert trial["exception"] is None
    assert trial["answer"].exists()
    assert trial["agent_result"]["metadata"]["model"].startswith(SOLVER)
    assert trial["agent_result"]["n_output_tokens"] > 0


def test_harbor_and_the_in_process_runner_agree_on_the_same_answers(live: dict[str, Any]) -> None:
    for pair in live["parity"].values():
        assert pair["harbor"]["core"] == pair["in_process"]["core"]
        assert abs(pair["harbor"]["reward"] - pair["in_process"]["reward"]) <= 0.02


def test_every_judge_call_is_in_some_grading_json(live: dict[str, Any]) -> None:
    """judge_usage, from Harbor's verifier and the in-process runner alike, adds up to the metered calls."""
    assert live["graded_judge_usage"]["per_model"] == live["judge_spend"]["per_model"]
    for trial in live["jobs"]["oracle"].values():
        assert trial["judge_usage"]["claude-sonnet-5-5"]["calls"] > 0
    for job in ("nop", "nop-regrade"):
        assert all(trial["judge_usage"] == {} for trial in live["jobs"][job].values())


def test_the_run_stayed_inside_its_budget(live: dict[str, Any]) -> None:
    assert live["total_cost_usd"] <= BUDGET_USD
