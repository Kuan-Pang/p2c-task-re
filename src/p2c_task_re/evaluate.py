"""In-process runner: solve and grade task folders with no container (PLAN §3 "One grading module", §9).

Every number in the plan except the released headline comes from here; the headline comes from Harbor.
Both grade with the same `grade.py`, so the runner can also grade through the task's own `tests/test.sh`,
the script Harbor's verifier runs. Regrading the same stored answers both ways is the parity check
(PLAN §3: `core` exact, |Δreward| ≤ 0.02, since the judge is sampled).

Public API (the Hydra `evaluate` stage calls `evaluate`):

    evaluate(task_dirs, agents, out_dir, *, k=1, judging=JudgeConfig(), via="in_process",
             judges=None, ledger=None, concurrency=4) -> dict          # the summary.json content
    grade_answer(task_dir, trial_dir, *, judging=JudgeConfig(), via="in_process", judges=None,
                 ledger=None, tag=None) -> dict
    JudgeConfig.from_config(roster, grading_cfg)     # roster = llm.load_roster(cfg.models),
                                                     # grading_cfg = cfg.stages.grading
    find_tasks(dataset_dir) -> list[Path]

Agents: "oracle" runs solution/solve.sh with OUTPUT_DIR set to the trial folder; "nop" writes nothing; a
RoleSpec sends the instruction as its single user message through `llm.complete` and writes the completion
text verbatim to answer.json — a refusal too, so it grades as a valid 0 rather than being dropped. The
instruction is sent as Harbor hands it to an agent: instruction.md with its leading canary line removed.

Grading, `via`:
    "in_process"  grade.grade() with judges from grade.make_judge(model, effort), or the injected
                  `judges` pair (primary, regrade) — tests use scripted judges this way.
    "test_sh"     bash tests/test.sh with TESTS_DIR, ANSWER_PATH, VERIFIER_DIR and the JUDGE_* variables
                  set and this interpreter's bin folder first on PATH.

Output, one folder per trial (NN counts from 01):

    <out_dir>/<task_id>/<agent>/<NN>/answer.json   what the agent wrote (absent for nop)
                                     reward.json   {"reward", "core", "judged"}
                                     grading.json  the grader's full record
                                     solver.json   model agents only: model, refusal, tokens, cost (no text)
                                     error.txt     only for an invalid trial
    <out_dir>/summary.json                         one row per task x agent (see `summarize`)

A trial is invalid only when something transient failed: the model call (LLMError), the judge (GradingError,
test.sh exit 2) or the oracle's solve.sh. A missing or unusable answer and a refusal are valid zeros. Means
are over valid trials.

Spend. Solver calls go to the ledger as they are made. Judging is most of the cost, so the in-process path
builds its judges with an `on_usage` callback that records one ledger line per judge API call (role `judge`
or `judge_regrade`, the trial's tag, cost at the roster's prices) and totals the same calls into
grading.json's `judge_usage`; the budget is checked before each trial's grading, so a run stops at its cap
instead of judging past it. The test.sh path builds its judges inside the subprocess, so there the spend
reaches `judge_usage` only.

Resume. A trial whose reward.json and grading.json are already there is not solved or graded again: its
stored grading is read back, and the summary row counts it under `n_resumed`. A model agent's trial that
was solved but not graded (answer.json and solver.json written, then the run stopped) is graded from its
stored answer rather than paying the solver again. A trial that *is* re-run first drops the previous
attempt's error.txt, so a stale error never outlives the failure it describes.

Paths are resolved first, so relative task, trial and output folders work: the oracle's solve.sh and
test.sh run with their trial folder as the working directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, cast

from p2c_task_re import llm
from p2c_task_re.grading import grade as g
from p2c_task_re.ledger import CallRecord, Ledger
from p2c_task_re.llm import Provider, RoleSpec
from p2c_task_re.stages.package import find_tasks, strip_canary

__all__ = ["JudgeConfig", "Trial", "evaluate", "find_tasks", "grade_answer", "solver_prompt", "summarize"]

Agent = str | RoleSpec  # "oracle" | "nop" | a model role
Via = Literal["in_process", "test_sh"]
BUILTIN_AGENTS = ("oracle", "nop")
SOLVER_ROLE = "solver"  # the ledger role of every model-agent call
JUDGE_ROLE = "judge"  # the ledger roles of the grader's calls (PLAN §3 roster)
JUDGE_REGRADE_ROLE = "judge_regrade"
SCRIPT_TIMEOUT_S = 1800  # task.toml's agent and verifier timeouts


@dataclass(frozen=True)
class JudgeConfig:
    """Which judges grade and how. The same values drive both grading paths, so they stay comparable."""

    model: str = g.DEFAULT_JUDGE_MODEL
    regrade_model: str | None = g.DEFAULT_REGRADE_MODEL
    effort: str = g.DEFAULT_JUDGE_EFFORT
    samples: int = g.DEFAULT_SAMPLES
    retries: int = g.DEFAULT_RETRIES
    concurrency: int = g.DEFAULT_CONCURRENCY
    # The roster entries behind the two models, keyed by ledger role: they carry the prices a judge call
    # is costed at. Empty (the default, and what a test's scripted judges need) = recorded with no cost.
    specs: Mapping[str, RoleSpec] = field(default_factory=dict)

    @classmethod
    def from_config(cls, roster: Mapping[str, Any], grading_cfg: Mapping[str, Any]) -> JudgeConfig:
        """Models from the roster's `judge` / `judge_regrade` roles, the rest from `stages.grading`.

        The effort is `grading.judge_effort`, not the role's own: the roster's Claude entry is shared with
        the generator, and task.toml's JUDGE_EFFORT default is the grading one.
        """
        judge: RoleSpec = roster["judge"]
        regrade: RoleSpec | None = roster.get("judge_regrade")
        specs = {JUDGE_ROLE: judge}
        if regrade is not None:
            specs[JUDGE_REGRADE_ROLE] = regrade
        return cls(
            model=judge.model,
            regrade_model=regrade.model if regrade is not None else None,
            effort=grading_cfg["judge_effort"],
            samples=grading_cfg["judge_samples"],
            retries=grading_cfg["judge_retries"],
            concurrency=grading_cfg["judge_concurrency"],
            specs=specs,
        )

    def build(self, on_usage: g.UsageCallback | None = None) -> tuple[g.Judge, g.Judge | None]:
        """The real judges, as grade.py's CLI would build them, reporting their usage to `on_usage`."""
        regrade = g.make_judge(self.regrade_model, self.effort, on_usage) if self.regrade_model else None
        return g.make_judge(self.model, self.effort, on_usage), regrade

    def role_of(self, model: str) -> str:
        """Which ledger role answered: the primary judge unless this is the regrade model."""
        return JUDGE_REGRADE_ROLE if model != self.model and model == self.regrade_model else JUDGE_ROLE

    def call_record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        refusal: bool = False,
        *,
        tag: str | None = None,
    ) -> CallRecord:
        """One ledger line for one judge API call, costed at the roster price of the model that served it."""
        role = self.role_of(model)
        spec = self.specs.get(role)
        if spec is not None and spec.model != model:
            spec = None  # a model the roster does not describe: recorded, with an unknown cost
        # A judge was built for `model`, so grade.py's rule names its provider.
        provider = spec.provider if spec is not None else cast(Provider, g.provider_of(model))
        return CallRecord(
            role=role,
            tag=tag,
            provider=provider,
            model_requested=model,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=llm.cost_usd(spec.price_per_mtok, input_tokens, output_tokens) if spec else None,
            refusal=refusal,
        )

    def env(self) -> dict[str, str]:
        """The JUDGE_* variables grade.py's CLI reads (docs/task_format.md §7); "" means no regrade judge."""
        return {
            "JUDGE_MODEL": self.model,
            "JUDGE_REGRADE_MODEL": self.regrade_model or "",
            "JUDGE_EFFORT": self.effort,
            "JUDGE_SAMPLES": str(self.samples),
            "JUDGE_RETRIES": str(self.retries),
            "JUDGE_CONCURRENCY": str(self.concurrency),
        }


@dataclass
class Trial:
    """One (task, agent, sample). `grading` is the grading.json content, None when the trial is invalid."""

    task_id: str
    agent: str
    sample: int
    grading: dict[str, Any] | None = None
    refusal: bool = False
    error: str | None = None
    resumed: bool = False  # already finished on disk, so neither solved nor graded again


def agent_name(agent: Agent) -> str:
    """The trial folder name: "oracle", "nop", or the model id."""
    if isinstance(agent, RoleSpec):
        return agent.model
    if agent not in BUILTIN_AGENTS:
        raise ValueError(f"agent {agent!r} is neither {' nor '.join(BUILTIN_AGENTS)} nor a RoleSpec")
    return agent


def solver_prompt(task_dir: Path | str) -> str:
    """instruction.md as Harbor hands it to an agent: leading canary lines and blank lines removed."""
    return strip_canary((Path(task_dir) / "instruction.md").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- solve


def solve(agent: Agent, task_dir: Path, trial_dir: Path, *, ledger: Ledger | None, tag: str) -> bool:
    """Let `agent` write trial_dir/answer.json. Returns whether the model refused (always False otherwise)."""
    if agent == "nop":
        return False
    if agent == "oracle":
        env = {**os.environ, "OUTPUT_DIR": str(trial_dir)}
        subprocess.run(
            ["bash", str(task_dir / "solution" / "solve.sh")],
            env=env,
            cwd=trial_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_S,
        )
        return False

    completion = llm.complete(agent, solver_prompt(task_dir), role=SOLVER_ROLE, ledger=ledger, tag=tag)
    (trial_dir / "answer.json").write_text(completion.text, encoding="utf-8")
    record = completion.model_dump(exclude={"text", "parsed"})
    (trial_dir / "solver.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return completion.refusal


# ---------------------------------------------------------------------------------------------- grade


def grade_answer(
    task_dir: Path | str,
    trial_dir: Path | str,
    *,
    judging: JudgeConfig = JudgeConfig(),
    via: Via = "in_process",
    judges: tuple[g.Judge, g.Judge | None] | None = None,
    ledger: Ledger | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Grade trial_dir/answer.json (absent = no answer) and write reward.json and grading.json beside it.

    Returns the grading.json content. Raises GradingError when the judge fails (the trial is invalid),
    TestsDirError when the task folder itself is broken, and BudgetExceeded before judging anything once
    the ledger's budget is spent. With a `ledger`, each judge API call is recorded under `tag`; injected
    `judges` report no usage, since a scripted judge has none.
    """
    task_dir, trial_dir = Path(task_dir).resolve(), Path(trial_dir).resolve()
    if ledger is not None:
        ledger.check_budget()
    answer_path = trial_dir / "answer.json"
    if via == "test_sh":
        if judges is not None:
            raise ValueError(
                "injected judges cannot reach test.sh; it builds its judges from JUDGE_* variables"
            )
        return _grade_with_test_sh(task_dir, answer_path, trial_dir, judging)
    if via != "in_process":
        raise ValueError(f"via must be 'in_process' or 'test_sh', got {via!r}")

    usage = g.JudgeUsage()
    if judges is None:
        judges = judging.build(_usage_sink(usage, judging, ledger, tag))
    judge, regrade = judges
    answer_text = answer_path.read_text(encoding="utf-8") if answer_path.exists() else None
    result = g.grade(
        answer_text,
        task_dir / "tests",
        judge,
        regrade,
        samples=judging.samples,
        retries=judging.retries,
        concurrency=judging.concurrency,
        usage=usage,
    )
    g.write_outputs(result, trial_dir)
    return result


def _usage_sink(
    usage: g.JudgeUsage, judging: JudgeConfig, ledger: Ledger | None, tag: str | None
) -> g.UsageCallback:
    """The judges' `on_usage`: total it for grading.json, and give the ledger one line per judge call."""

    def on_usage(model: str, input_tokens: int, output_tokens: int, refusal: bool) -> None:
        usage.add(model, input_tokens, output_tokens, refusal)
        if ledger is not None:
            ledger.record(judging.call_record(model, input_tokens, output_tokens, refusal, tag=tag))

    return on_usage


def _grade_with_test_sh(
    task_dir: Path, answer_path: Path, out_dir: Path, judging: JudgeConfig
) -> dict[str, Any]:
    """Run the task's shipped test.sh as Harbor's verifier would, with host paths for /tests and /logs."""
    env = {
        **os.environ,
        **judging.env(),
        "TESTS_DIR": str(task_dir / "tests"),
        "ANSWER_PATH": str(answer_path),
        "VERIFIER_DIR": str(out_dir),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    proc = subprocess.run(
        ["bash", str(task_dir / "tests" / "test.sh")],
        env=env,
        cwd=out_dir,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
    )
    if proc.returncode == 2:
        raise g.GradingError(f"test.sh: {proc.stderr.strip()}")
    if proc.returncode != 0:
        raise g.TestsDirError(f"test.sh exited {proc.returncode}: {proc.stderr.strip()}")
    return json.loads((out_dir / "grading.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- run


def run_trial(
    task_dir: Path,
    agent: Agent,
    sample: int,
    out_dir: Path,
    *,
    judging: JudgeConfig,
    via: Via,
    judges: tuple[g.Judge, g.Judge | None] | None,
    ledger: Ledger | None,
) -> Trial:
    """Solve and grade one trial in <out_dir>/<task_id>/<agent>/<NN>/, or resume it if it is finished."""
    trial = Trial(task_id=task_dir.name, agent=agent_name(agent), sample=sample)
    trial_dir = out_dir / trial.task_id / trial.agent / f"{sample:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{trial.task_id}/{trial.agent}/{sample:02d}"
    stored = finished_grading(trial_dir)
    if stored is not None:
        trial.grading, trial.resumed = stored, True
        trial.refusal = _recorded_refusal(trial_dir)
        return trial
    # This trial is being run again, so the last attempt's error must not outlive it.
    (trial_dir / "error.txt").unlink(missing_ok=True)
    try:
        if isinstance(agent, RoleSpec) and _was_solved(trial_dir):
            trial.refusal = _recorded_refusal(trial_dir)  # solved before the run stopped: grade only
        else:
            trial.refusal = solve(agent, task_dir, trial_dir, ledger=ledger, tag=tag)
        trial.grading = grade_answer(
            task_dir, trial_dir, judging=judging, via=via, judges=judges, ledger=ledger, tag=tag
        )
    except subprocess.CalledProcessError as exc:
        trial.error = f"solve.sh exited {exc.returncode}: {exc.stderr.strip()}"
    except (llm.LLMError, g.GradingError) as exc:
        trial.error = f"{type(exc).__name__}: {exc}"
    if trial.error is not None:
        (trial_dir / "error.txt").write_text(trial.error + "\n", encoding="utf-8")
    return trial


def finished_grading(trial_dir: Path) -> dict[str, Any] | None:
    """A finished trial's stored grading, or None when the trial has to be run.

    Finished means both files the grader writes are there and grading.json is readable; a half-written
    trial — reward.json alone, or a grading.json a crash cut short — is run again.
    """
    if not (trial_dir / "reward.json").exists():
        return None
    grading = trial_dir / "grading.json"
    if not grading.exists():
        return None
    try:
        return json.loads(grading.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _was_solved(trial_dir: Path) -> bool:
    """Whether a model agent's call already finished here: `solve` writes solver.json after answer.json."""
    return (trial_dir / "answer.json").exists() and (trial_dir / "solver.json").exists()


def _recorded_refusal(trial_dir: Path) -> bool:
    """Whether the model agent of a resumed trial refused; written by `solve` to solver.json."""
    solver = trial_dir / "solver.json"
    if not solver.exists():
        return False
    try:
        return bool(json.loads(solver.read_text(encoding="utf-8")).get("refusal"))
    except ValueError:
        return False


def evaluate(
    task_dirs: Sequence[Path | str],
    agents: Sequence[Agent],
    out_dir: Path | str,
    *,
    k: int = 1,
    judging: JudgeConfig = JudgeConfig(),
    via: Via = "in_process",
    judges: tuple[g.Judge, g.Judge | None] | None = None,
    ledger: Ledger | None = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Run k samples of every agent on every task, grade them, and write summary.json; returns the summary.

    `concurrency` bounds the trials in flight; each trial's grader runs up to `judging.concurrency` judge
    calls of its own. A BudgetExceeded from the ledger stops the run. Trials already finished in `out_dir`
    are resumed rather than re-run, so a stopped run can be continued into the same folder.

    Each trial builds its own judges, so every judge call is recorded against that trial; pass `judges` to
    grade with judges of your own instead (what the tests do), and they report no usage.
    """
    out_dir = Path(out_dir).resolve()
    names = [agent_name(agent) for agent in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"two agents share a trial folder name: {names}")

    jobs = [
        (Path(task).resolve(), agent, sample)
        for task in task_dirs
        for agent in agents
        for sample in range(1, k + 1)
    ]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(
                run_trial,
                task,
                agent,
                sample,
                out_dir,
                judging=judging,
                via=via,
                judges=judges,
                ledger=ledger,
            )
            for task, agent, sample in jobs
        ]
        trials = [future.result() for future in futures]

    if judges is not None:
        judge_models = [judge.name for judge in judges if judge is not None]
    else:
        judge_models = [model for model in (judging.model, judging.regrade_model) if model]
    summary = {"via": via, "k": k, "judge_models": judge_models, "rows": summarize(trials)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize(trials: Sequence[Trial]) -> list[dict[str, Any]]:
    """One row per (task, agent), in first-seen order. Means are over valid trials, None if none."""
    groups: dict[tuple[str, str], list[Trial]] = {}
    for trial in trials:
        groups.setdefault((trial.task_id, trial.agent), []).append(trial)

    rows = []
    for (task_id, agent), group in groups.items():
        valid = [t.grading for t in group if t.grading is not None]
        rows.append(
            {
                "task_id": task_id,
                "agent": agent,
                "n": len(group),
                "n_valid": len(valid),
                "n_invalid": len(group) - len(valid),
                "n_resumed": sum(t.resumed for t in group),
                "n_refusals": sum(t.refusal for t in group),
                "mean_s": _mean(valid, "s"),
                "mean_s_core": _mean(valid, "s_core"),
                "mean_s_judge": _mean(valid, "s_judge"),
                "pass_rate": _mean(valid, "passed"),
            }
        )
    return rows


def _mean(results: list[dict[str, Any]], key: str) -> float | None:
    return round(fmean(float(result[key]) for result in results), 6) if results else None
