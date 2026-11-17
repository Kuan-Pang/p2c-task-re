"""Tests for the config schema (PLAN §10): every experiment composes through Hydra and validates, a typo
inside a YAML file is rejected with a readable message, and roles can be swapped from the command line."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError

from p2c_task_re import run
from p2c_task_re.cli import CONFIG_DIR
from p2c_task_re.config import Config, describe_errors, roster_warnings, validate

EXPERIMENTS = sorted(path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml"))


def compose_config(*overrides: str, config_dir: Path = CONFIG_DIR) -> Config:
    """The config `p2c-pipeline <overrides>` would run with."""
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return validate(compose(config_name="pipeline", overrides=list(overrides)))


def problems(*overrides: str, config_dir: Path = CONFIG_DIR) -> str:
    """describe_errors() of a config that must not validate."""
    with pytest.raises(ValidationError) as caught:
        compose_config(*overrides, config_dir=config_dir)
    return describe_errors(caught.value)


def test_the_experiments_are_the_three_plan_ones() -> None:
    assert EXPERIMENTS == ["eval", "full", "pilot"]


def test_the_config_dir_is_the_repos_absolute_configs_folder() -> None:
    assert CONFIG_DIR.is_absolute() and (CONFIG_DIR / "pipeline.yaml").is_file()


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_composes_validates_and_names_known_stages(experiment: str) -> None:
    cfg = compose_config(f"+experiment={experiment}")
    assert cfg.run_name == experiment
    known = [stage.name for stage in run.STAGES]
    assert cfg.run_stages and set(cfg.run_stages) <= set(known)


def test_the_base_config_validates_without_an_experiment() -> None:
    cfg = compose_config()
    assert cfg.run_stages == [] and cfg.pilot is None
    assert cfg.models.roles.judge.model == "claude-sonnet-5-5"
    assert [spec.provider for spec in cfg.models.roles.panel] == ["openai", "anthropic"]


def test_the_pilot_pins_five_papers_and_a_reduced_difficulty_panel() -> None:
    cfg = compose_config("+experiment=pilot")
    assert cfg.pilot is not None and len(cfg.pilot.papers) == 5
    assert cfg.stages.gates.difficulty.k == 3
    assert compose_config().stages.gates.difficulty.k == 6


# ----------------------------------------------------------------------------- typos inside YAML


@pytest.mark.parametrize(
    ("file", "old", "new", "expected"),
    [
        ("stages/default.yaml", "judge_samples: 3", "judge_sampels: 3", "stages.grading.judge_sampels"),
        ("data/post_jan.yaml", "meca_root:", "meca_rot:", "data.meca_rot"),
        ("models/default.yaml", "  model: claude-sonnet", "  modle: claude-sonnet", "models.sonnet.modle"),
        ("families/v1.yaml", "f3: {enabled: true,", "f3: {enabeld: true,", "families.f3.enabeld"),
        ("experiment/pilot.yaml", "max_tasks_per_paper:", "max_task_per_paper:", "pilot.max_task_per_paper"),
        ("pipeline.yaml", "concurrency: 8", "concurency: 8", "concurency"),
    ],
)
def test_a_typo_inside_a_yaml_file_is_rejected_readably(
    tmp_path: Path, file: str, old: str, new: str, expected: str
) -> None:
    configs = shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    path = configs / file
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")

    message = problems("+experiment=pilot", config_dir=configs)
    assert any(line.startswith(f"  {expected}: unknown key") for line in message.splitlines()), message
    assert "required key is missing" in message  # the correctly spelled key is reported too


def test_a_typo_on_the_command_line_is_caught_by_hydra() -> None:
    with pytest.raises(ConfigCompositionException, match="eval.agentz"):
        compose_config("eval.agentz=[nop]")


def test_a_wrong_value_names_the_key_and_the_value() -> None:
    message = problems("stages.grading.judge_effort=meduim", "budget_usd=-1")
    assert "stages.grading.judge_effort: Input should be 'low', 'medium'" in message
    assert "(got 'meduim')" in message
    assert "budget_usd: Input should be greater than 0 (got -1)" in message


# ---------------------------------------------------------------------------------------- roster


def test_a_role_is_overridden_from_the_command_line() -> None:
    cfg = compose_config("models.roles.critic=${models.gpt}")
    assert cfg.models.roles.critic.provider == "openai"
    assert cfg.models.roles.critic.model == "gpt-6.1-sol"
    assert cfg.models.roles.generator.model == "claude-sonnet-5-5"
    assert cfg.models.roster()["critic"] == cfg.models.gpt


def test_a_list_role_and_a_single_field_are_overridden_too() -> None:
    cfg = compose_config("models.roles.panel=[${models.sonnet}]", "models.roles.judge.effort=low")
    assert [spec.model for spec in cfg.models.roles.panel] == ["claude-sonnet-5-5"]
    assert cfg.models.roles.judge.effort == "low"
    assert cfg.models.sonnet.effort == "high"  # the override changed the role, not the shared entry


def test_a_role_is_a_rolespec() -> None:
    message = problems("models.roles.critic.provider=mistral", "models.roles.judge.effort=extreme")
    assert "models.roles.critic.provider: Input should be 'anthropic' or 'openai'" in message
    assert "models.roles.judge.effort:" in message


def test_refusal_fallback_goes_through_load_roster() -> None:
    message = problems("models.refusal_fallback=[generator,panel]")
    assert "models: refusal_fallback may list only ['adversary', 'critic', 'generator']" in message
    assert "got ['panel']" in message


def test_the_default_roster_breaks_no_startup_rule() -> None:
    assert roster_warnings(compose_config().models.roles) == []


def test_roster_rules_warn_rather_than_fail() -> None:
    one_model = compose_config("models.roles.checker=[${models.sonnet},${models.sonnet}]")
    (warning,) = roster_warnings(one_model.models.roles)
    assert warning.startswith("checker has no model other than the generator's (claude-sonnet-5-5)")
    heldout = compose_config("models.roles.heldout=${models.sonnet}")
    assert roster_warnings(heldout.models.roles) == [
        "heldout claude-sonnet-5-5 must lie outside panel",
        "heldout claude-sonnet-5-5 must differ from the generator",
    ]


# ----------------------------------------------------------------------------- cross-key checks


def test_an_agent_must_be_builtin_or_a_set_role() -> None:
    agents = ["oracle", "nop", "panel", "judge"]
    assert compose_config(f"eval.agents=[{','.join(agents)}]").eval.agents == agents
    unknown = problems("eval.agents=[oracle,critc]")
    assert "eval.agents: 'critc' is neither oracle nor nop nor a role in models.roles" in unknown
    assert "eval.agents: role 'heldout' is unset" in problems("eval.agents=[heldout]")


def test_a_family_takes_only_its_own_directions() -> None:
    assert "f4.directions: F4 cannot be ['forward']" in problems("families.f4.directions=[forward]")
    assert compose_config("families.f2.directions=[forward]").families.f2.directions == ["forward"]


def test_corpus_priority_must_list_each_corpus() -> None:
    message = problems("data.corpus_priority=[immunology]")
    assert "data: corpus_priority ['immunology'] must list each" in message
