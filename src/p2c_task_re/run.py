"""Stage order and the run folder (PLAN §3 stage list, §10 `run.py`).

`STAGES` is the one ordered list of pipeline stages. `run_stages` in the config picks which of them run;
the order always comes from here. Modules are named for their job (stages/select.py, gates/...), and each
docstring carries its S-number. In M0 only `evaluate` is built: asking for any other stage fails before
anything runs, with the milestone that adds it. The per-candidate loop with its cheapest-first gate order
(S3 lints → V1 → V3 → S6 → S5 → V4, V5 → V2 → D) arrives with those stages in M1.

The run folder is Hydra's run dir (`hydra.run.dir`, default outputs/runs/<run_name>/<timestamp>/):

    .hydra/          Hydra's record of the composed config and the overrides
    pipeline.log     the job log
    config.json      the validated, resolved config of this run
    calls.jsonl      the ledger, one line per model call; `budget_usd` caps the whole run
    eval/            the `evaluate` stage: summary.json and <task_id>/<agent>/<NN>/ trial folders

Resume: pass an existing run folder as `hydra.run.dir`. A stage whose output already exists is skipped
(`evaluate`: eval/summary.json), and the ledger reloads calls.jsonl so the budget covers every attempt.
The config must match config.json, except the settings that change no output (`RESUMABLE`), so one folder
never mixes two configurations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from p2c_task_re.config import Config, roster_warnings
from p2c_task_re.evaluate import BUILTIN_AGENTS, Agent, JudgeConfig, evaluate, find_tasks
from p2c_task_re.ledger import Ledger

log = logging.getLogger(__name__)

# Config keys that may change when a run folder is resumed: they decide what runs and how fast, not what
# any output contains. Raise budget_usd to finish a run that stopped at its cap.
RESUMABLE = ("run_name", "run_stages", "budget_usd", "concurrency")


class PipelineError(Exception):
    """The run cannot start or continue as configured; the message says what to change."""


# ------------------------------------------------------------------------------------------ stages


def evaluate_stage(cfg: Config, run_dir: Path, ledger: Ledger) -> None:
    """Run `eval.agents` over the task folders in `eval.dataset` with the in-process runner, into eval/."""
    out_dir = run_dir / "eval"
    if (out_dir / "summary.json").exists():
        log.info("evaluate: %s exists, skipping", out_dir / "summary.json")
        return
    tasks = eval_tasks(cfg)
    agents = eval_agents(cfg)
    judging = JudgeConfig.from_config(cfg.models.roster(), cfg.stages.grading.model_dump())
    log.info(
        "evaluate: %d tasks from %s x %d agents x k=%d, judge %s (regrade %s)",
        len(tasks),
        cfg.eval.dataset.resolve(),
        len(agents),
        cfg.eval.k,
        judging.model,
        judging.regrade_model,
    )
    summary = evaluate(
        tasks, agents, out_dir, k=cfg.eval.k, judging=judging, ledger=ledger, concurrency=cfg.concurrency
    )
    for row in summary["rows"]:
        log.info(
            "  %-32s %-20s valid %d/%d  refusals %d  mean s %s  pass rate %s",
            row["task_id"],
            row["agent"],
            row["n_valid"],
            row["n"],
            row["n_refusals"],
            _fmt(row["mean_s"]),
            _fmt(row["pass_rate"]),
        )
        if row["n_invalid"]:
            log.warning(
                "  %s %s: %d invalid trial(s); the cause is in %s/<NN>/error.txt",
                row["task_id"],
                row["agent"],
                row["n_invalid"],
                out_dir / row["task_id"] / row["agent"],
            )


@dataclass(frozen=True)
class Stage:
    name: str  # the name run_stages uses
    job: str  # S-number and what it does (PLAN §3)
    milestone: str  # the milestone that builds it (PLAN §11)
    run: Callable[[Config, Path, Ledger], None] | None = None  # None until it is built


STAGES = (
    Stage("select", "S0 seed selector", "M1"),
    Stage("facts", "S1 fact verifier", "M1"),
    Stage("architect", "S2 task architect, and hardening (M2)", "M1"),
    Stage("build", "S3 context builder", "M1"),
    Stage("critic", "S5 critic", "M1"),
    Stage("adversary", "S6 adversary", "M1"),
    Stage("verify", "S7 verifiability gates V1-V5", "M1"),
    Stage("difficulty", "S8 difficulty panel D", "M1"),
    Stage("audit", "S9 trace auditor", "M2"),
    Stage("package", "S10 packager over the run's candidates", "M1"),
    Stage("review", "the review/ pages for hand review", "M1"),
    Stage("evaluate", "in-process runner over a finished dataset", "M0", evaluate_stage),
)


def planned_stages(names: list[str]) -> list[Stage]:
    """The requested stages in pipeline order. Raises PipelineError for an unknown or unbuilt stage."""
    known = {stage.name: stage for stage in STAGES}
    if not names:
        raise PipelineError("run_stages is empty: pick an experiment, e.g. `+experiment=eval`")
    if unknown := [name for name in names if name not in known]:
        raise PipelineError(
            f"unknown stage(s) in run_stages: {', '.join(unknown)}. The stages, in order: {', '.join(known)}"
        )
    stages = [stage for stage in STAGES if stage.name in names]
    unbuilt = [stage for stage in stages if stage.run is None]
    if unbuilt:
        by_milestone: dict[str, list[str]] = {}
        for stage in unbuilt:
            by_milestone.setdefault(stage.milestone, []).append(stage.name)
        later = "; ".join(f"{ms} adds {', '.join(group)}" for ms, group in sorted(by_milestone.items()))
        built = ", ".join(stage.name for stage in STAGES if stage.run is not None)
        raise PipelineError(
            f"run_stages asks for stages that are not built yet ({later}); built so far: {built}"
        )
    return stages


def eval_tasks(cfg: Config) -> list[Path]:
    """The task folders `evaluate` runs on. Raises PipelineError when eval.dataset names none."""
    dataset = cfg.eval.dataset
    if dataset is None:
        raise PipelineError("the evaluate stage needs eval.dataset, a folder of task folders")
    if not dataset.is_dir():
        raise PipelineError(f"eval.dataset {dataset} is not a folder")
    # Absolute, because the oracle runs solution/solve.sh from inside its trial folder.
    tasks = find_tasks(dataset.resolve())
    if not tasks:
        raise PipelineError(f"eval.dataset {dataset} holds no task folders (none has tests/rubric.json)")
    return tasks


def eval_agents(cfg: Config) -> list[Agent]:
    """`eval.agents` as runner agents: oracle and nop as named, a role as its RoleSpec (panel: each model)."""
    agents: list[Agent] = []
    for name in cfg.eval.agents:
        role = name if name in BUILTIN_AGENTS else getattr(cfg.models.roles, name)
        for agent in role if isinstance(role, list) else [role]:
            if agent not in agents:
                agents.append(agent)
    return agents


# -------------------------------------------------------------------------------------------- run


def run_pipeline(cfg: Config, run_dir: Path) -> None:
    """Run the configured stages in order inside `run_dir`, skipping those whose output exists."""
    stages = planned_stages(cfg.run_stages)
    if any(stage.name == "evaluate" for stage in stages):
        eval_tasks(cfg)  # a bad dataset fails here, before anything is written
    for warning in roster_warnings(cfg.models.roles):
        log.warning("roster: %s", warning)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot(cfg, run_dir / "config.json")
    ledger = Ledger(run_dir / "calls.jsonl", budget_usd=cfg.budget_usd)
    log.info("run folder %s, stages: %s", run_dir, ", ".join(stage.name for stage in stages))
    for stage in stages:
        stage.run(cfg, run_dir, ledger)
    log.info("done; ledger spend $%.4f (solver and judge calls)", ledger.spent())


def write_snapshot(cfg: Config, path: Path) -> None:
    """Write config.json; on resume, first check that only RESUMABLE settings changed."""
    new = cfg.model_dump(mode="json")
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if changed := [key for key in changed_keys(old, new) if key.split(".")[0] not in RESUMABLE]:
            raise PipelineError(
                f"{path.parent} was run with a different config (changed: {', '.join(changed)}). "
                "Start a new run folder, or resume with the original settings."
            )
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def changed_keys(old: dict[str, Any], new: dict[str, Any], prefix: str = "") -> list[str]:
    """The dotted keys whose values differ between two nested dicts."""
    keys = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys += changed_keys(a, b, f"{prefix}{key}.")
        elif a != b:
            keys.append(prefix + key)
    return keys


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
