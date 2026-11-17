"""Tests for the `p2c-pipeline` console script and run.py (PLAN §10): the installed script composes the repo's
configs from any folder, an eval run goes end to end offline, unbuilt stages fail before anything runs, and
a run folder resumes by skipping what exists.

The console-script tests run the real script in a subprocess with the API keys removed from its
environment, so "offline" is checked rather than assumed. The run.py tests call `run_pipeline` directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message
from hydra import compose, initialize_config_dir

from p2c_task_re import llm, run
from p2c_task_re.cli import CONFIG_DIR
from p2c_task_re.config import Config, validate
from p2c_task_re.ledger import BudgetExceeded
from p2c_task_re.run import PipelineError

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "tasks"
PIPELINE = Path(sys.executable).parent / "p2c-pipeline"
TASK_IDS = sorted(path.name for path in EXAMPLES.iterdir() if (path / "tests" / "rubric.json").exists())
M1_STAGES = "select, facts, architect, build, critic, adversary, verify, difficulty, package, review"


def pipeline(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    """Run the installed console script with no API keys in its environment."""
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    return subprocess.run(
        [str(PIPELINE), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def compose_config(*overrides: str) -> Config:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return validate(compose(config_name="pipeline", overrides=list(overrides)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------ console script


def test_cfg_job_prints_the_composed_config_from_any_folder(tmp_path: Path) -> None:
    proc = pipeline("+experiment=eval", "--cfg", "job", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "run_stages:\n- evaluate\n" in proc.stdout
    assert "judge_regrade:" in proc.stdout
    assert list(tmp_path.iterdir()) == []  # printing the config runs nothing and writes nothing


def test_eval_with_nop_runs_end_to_end_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc = pipeline(
        "+experiment=eval", "eval.dataset=examples/tasks", "eval.agents=[nop]", f"hydra.run.dir={run_dir}"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert (run_dir / ".hydra" / "config.yaml").is_file()
    assert "evaluate: 3 tasks" in (run_dir / "pipeline.log").read_text(encoding="utf-8")
    snapshot = read_json(run_dir / "config.json")
    assert snapshot["run_stages"] == ["evaluate"]
    assert snapshot["eval"] == {"dataset": "examples/tasks", "agents": ["nop"], "k": 1}
    assert not (run_dir / "calls.jsonl").exists()  # nop makes no model call

    summary = read_json(run_dir / "eval" / "summary.json")
    assert summary["judge_models"] == ["claude-sonnet-5-5", "gpt-6.1-sol"]
    assert [(row["task_id"], row["agent"]) for row in summary["rows"]] == [(t, "nop") for t in TASK_IDS]
    for row in summary["rows"]:
        assert (row["n_valid"], row["mean_s"], row["pass_rate"]) == (1, 0.0, 0.0)
        trial = run_dir / "eval" / row["task_id"] / "nop" / "01"
        assert read_json(trial / "reward.json") == {"reward": 0.0, "core": 0.0, "judged": 0.0}
        assert read_json(trial / "grading.json")["status"] == "invalid_answer"


def test_the_pilot_fails_fast_naming_the_m1_stages(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc = pipeline("+experiment=pilot", f"hydra.run.dir={run_dir}")
    assert proc.returncode == 1
    assert f"not built yet (M1 adds {M1_STAGES}); built so far: evaluate" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr
    assert sorted(path.name for path in run_dir.iterdir()) == [".hydra", "pipeline.log"]  # Hydra's own


def test_a_yaml_typo_stops_the_script_with_one_line_per_key(tmp_path: Path) -> None:
    configs = shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    stages = configs / "stages" / "default.yaml"
    typo = stages.read_text(encoding="utf-8").replace("judge_samples:", "judge_sampels:")
    stages.write_text(typo, encoding="utf-8")
    proc = pipeline(
        "--config-path", str(configs), "+experiment=eval", "eval.dataset=examples/tasks",
        f"hydra.run.dir={tmp_path / 'run'}",
    )  # fmt: skip
    assert proc.returncode == 1
    assert "p2c-pipeline: invalid config (2 problem(s)):" in proc.stdout
    assert "  stages.grading.judge_samples: required key is missing\n" in proc.stdout
    assert "  stages.grading.judge_sampels: unknown key (misspelled?)\n" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


# ------------------------------------------------------------------------------------- stage order


def test_stages_run_in_pipeline_order_whatever_the_config_order() -> None:
    names = [stage.name for stage in run.STAGES]
    assert names.index("select") == 0 and names[-1] == "evaluate"
    assert [stage.name for stage in run.planned_stages(["evaluate", "evaluate"])] == ["evaluate"]


def test_the_full_run_names_m2_for_the_audit() -> None:
    with pytest.raises(PipelineError, match=r"M1 adds select, .*, package; M2 adds audit\)"):
        run.planned_stages(compose_config("+experiment=full").run_stages)


def test_an_unknown_or_empty_stage_list_is_refused() -> None:
    with pytest.raises(PipelineError, match="unknown stage.*evalute.*The stages, in order: select, facts"):
        run.planned_stages(["evalute"])
    with pytest.raises(PipelineError, match=r"\+experiment=eval"):
        run.planned_stages([])


@pytest.mark.parametrize(
    ("dataset", "message"),
    [("null", "needs eval.dataset"), ("/no/such/folder", "is not a folder"), ("{empty}", "holds no task")],
)
def test_a_bad_dataset_fails_before_anything_is_written(tmp_path: Path, dataset: str, message: str) -> None:
    (tmp_path / "empty").mkdir()
    cfg = compose_config("+experiment=eval", f"eval.dataset={dataset.format(empty=tmp_path / 'empty')}")
    with pytest.raises(PipelineError, match=message):
        run.run_pipeline(cfg, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_agents_expand_roles_to_their_models_once_each() -> None:
    cfg = compose_config("eval.agents=[oracle,panel,judge,nop]")
    agents = run.eval_agents(cfg)
    assert agents[0] == "oracle" and agents[-1] == "nop"
    assert [agent.model for agent in agents[1:-1]] == ["gpt-6.1-sol", "claude-sonnet-5-5"]  # judge = panel's


# ----------------------------------------------------------------------------------- run folder


class FakeAnthropic:
    """Answers every solver call with text that is not JSON (a valid 0 that needs no judge call)."""

    def __init__(self) -> None:
        self.messages = self
        self.calls = 0

    def stream(self, **_: Any) -> FakeAnthropic:
        self.calls += 1
        return self

    def __enter__(self) -> FakeAnthropic:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> Message:
        return Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "I think it goes down."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},  # $0.007 at $2/$10 per Mtok
            }
        )


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> FakeAnthropic:
    client = FakeAnthropic()
    monkeypatch.setattr(llm, "get_client", lambda provider: client)
    return client


def eval_config(*overrides: str) -> Config:
    return compose_config("+experiment=eval", f"eval.dataset={EXAMPLES}", "concurrency=1", *overrides)


def test_a_finished_stage_is_skipped_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = eval_config("eval.agents=[nop]")
    run.run_pipeline(cfg, tmp_path)
    summary = (tmp_path / "eval" / "summary.json").read_bytes()

    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("evaluate ran again")

    monkeypatch.setattr(run, "evaluate", fail)
    run.run_pipeline(cfg.model_copy(update={"budget_usd": 5.0, "concurrency": 4}), tmp_path)
    assert (tmp_path / "eval" / "summary.json").read_bytes() == summary
    assert read_json(tmp_path / "config.json")["budget_usd"] == 5.0


def test_resuming_with_a_different_config_is_refused(tmp_path: Path) -> None:
    run.run_pipeline(eval_config("eval.agents=[nop]"), tmp_path)
    changed = eval_config("eval.agents=[nop]", "eval.k=2", "stages.grading.judge_samples=1")
    refused = r"different config \(changed: eval.k, stages.grading.judge_samples\)"
    with pytest.raises(PipelineError, match=refused):
        run.run_pipeline(changed, tmp_path)


def test_solver_calls_go_to_the_ledger_and_the_budget_stops_the_run(
    tmp_path: Path, fake_anthropic: FakeAnthropic
) -> None:
    with pytest.raises(BudgetExceeded):
        run.run_pipeline(eval_config("eval.agents=[generator]", "budget_usd=0.01"), tmp_path)
    assert fake_anthropic.calls == 2  # $0.007 each: the third call finds $0.014 spent of $0.01
    assert not (tmp_path / "eval" / "summary.json").exists()

    # Raising the budget is the one change a resume may make that matters here; the stage then reruns.
    # The first task's trial finished before the stop, and the second was solved but stopped before its
    # grading, so only the third task's solver call is new.
    run.run_pipeline(eval_config("eval.agents=[generator]", "budget_usd=1.0"), tmp_path)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2 + 1 and {call["role"] for call in calls} == {"solver"}
    assert len({call["tag"] for call in calls}) == 3, "no task was solved twice"
    assert calls[-1]["tag"].endswith("/claude-sonnet-5-5/01")
    rows = read_json(tmp_path / "eval" / "summary.json")["rows"]
    assert {(row["agent"], row["n_valid"], row["mean_s"]) for row in rows} == {("claude-sonnet-5-5", 1, 0.0)}
    assert len(rows) == 3
