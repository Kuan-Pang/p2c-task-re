"""The composed Hydra config as one pydantic schema (PLAN §10 `config.py`).

Hydra struct-checks only command-line overrides, so a misspelled key inside a YAML file would reach the run
as a silently unused value. Every model here forbids unknown keys and requires every key the YAML files
define; `validate(cfg)` resolves the composed config and checks it, and `describe_errors` turns a
`ValidationError` into one line per bad key.

The values live in configs/; this file only says what shape they have. Model roles are `llm.RoleSpec`s and
`Models.roster()` is `llm.load_roster`, so the run and the schema agree on what a role is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from p2c_task_re import llm
from p2c_task_re.evaluate import BUILTIN_AGENTS
from p2c_task_re.families import DIRECTIONS
from p2c_task_re.llm import Effort, RoleSpec

STRICT = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------- data (PLAN §8)


class Data(BaseModel):
    """configs/data/*.yaml: the read-only inputs, as exact dated folders."""

    model_config = STRICT

    extraction_dirs: dict[str, Path]
    corpus_priority: list[str]
    meca_root: Path
    split_manifest: Path

    @model_validator(mode="after")
    def _priority_lists_each_corpus(self) -> Data:
        if sorted(self.corpus_priority) != sorted(self.extraction_dirs):
            raise ValueError(
                f"corpus_priority {self.corpus_priority} must list each extraction_dirs corpus "
                f"{sorted(self.extraction_dirs)} exactly once"
            )
        return self


# ----------------------------------------------------------------------------------- models (PLAN §3)


class Roles(BaseModel):
    """The §3 roster: one RoleSpec per role; `checker` and `panel` are lists; unset roles are null."""

    model_config = STRICT

    generator: RoleSpec
    critic: RoleSpec
    adversary: RoleSpec
    checker: list[RoleSpec] = Field(min_length=1)
    panel: list[RoleSpec] = Field(min_length=1)
    judge: RoleSpec
    judge_regrade: RoleSpec | None
    heldout: RoleSpec | None


class Models(BaseModel):
    """configs/models/*.yaml. `sonnet` and `gpt` are the named entries the roles point at, so a role can be
    swapped from the command line: `'models.roles.critic=${models.gpt}'`."""

    model_config = STRICT

    sonnet: RoleSpec
    gpt: RoleSpec
    roles: Roles
    refusal_fallback: list[str]

    @model_validator(mode="after")
    def _roster_loads(self) -> Models:
        self.roster()  # llm.load_roster rejects a role that may not use the refusal fallback
        return self

    def roster(self) -> dict[str, RoleSpec | list[RoleSpec] | None]:
        """The roster exactly as `llm.load_roster` builds it from this block."""
        return llm.load_roster(self.model_dump())


def roster_warnings(roles: Roles) -> list[str]:
    """The PLAN §3 roster rules a run checks at startup; each broken one is a warning, not an error."""
    generator = roles.generator.model
    warnings = []
    if all(spec.model == generator for spec in roles.checker):
        warnings.append(
            f"checker has no model other than the generator's ({generator}), so its three votes cannot "
            "span two models"
        )
    if roles.heldout is not None:
        if roles.heldout.model in {spec.model for spec in roles.panel}:
            warnings.append(f"heldout {roles.heldout.model} must lie outside panel")
        if roles.heldout.model == generator:
            warnings.append(f"heldout {roles.heldout.model} must differ from the generator")
    return warnings


# --------------------------------------------------------------------------------- families (PLAN §7)


class Family(BaseModel):
    model_config = STRICT

    enabled: bool
    directions: list[Literal["forward", "backward"]]
    w_core: float = Field(gt=0.30, lt=1.0)
    rationale_word_cap: int = Field(gt=0)


class Families(BaseModel):
    """configs/families/*.yaml. Each family's directions must be a subset of the ones it allows
    (families.DIRECTIONS)."""

    model_config = STRICT

    f1: Family
    f2: Family
    f3: Family
    f4: Family
    f5: Family
    f6: Family
    f7: Family

    @model_validator(mode="after")
    def _directions_are_allowed(self) -> Families:
        for name in type(self).model_fields:
            family, spec = name.upper(), getattr(self, name)
            if extra := set(spec.directions) - set(DIRECTIONS[family]):
                raise ValueError(f"{name}.directions: {family} cannot be {sorted(extra)}")
        return self


# ---------------------------------------------------------------------------- stages (PLAN §3, §5, §6)


class Grading(BaseModel):
    model_config = STRICT

    judge_samples: int = Field(ge=1)
    judge_effort: Effort
    judge_retries: int = Field(ge=0)
    judge_concurrency: int = Field(ge=1)


class Package(BaseModel):
    model_config = STRICT

    environment: Literal["local", "singularity", "cloud"]
    image_sif: Path | None
    image_oci: str | None
    canary: str


class Candidate(BaseModel):
    model_config = STRICT

    max_revisions: int = Field(ge=0)
    max_usd: float = Field(gt=0.0)


class V2Gate(BaseModel):
    model_config = STRICT

    samples_per_model: int = Field(ge=1)
    min_valid_per_model: int = Field(ge=1)
    pass_total: int = Field(ge=1)
    pass_per_model: int = Field(ge=1)


class V3Gate(BaseModel):
    model_config = STRICT

    fork_samples_per_model: int = Field(ge=1)
    fork_pass: int = Field(ge=1)


class V5Gate(BaseModel):
    model_config = STRICT

    must_pass_margin: float
    must_fail_margin: float
    must_not_pass_judge: float


class DifficultyGate(BaseModel):
    model_config = STRICT

    k: int = Field(ge=1)
    k_escalate: int = Field(ge=1)
    easy: float = Field(ge=0.0, le=1.0)
    hard: float = Field(ge=0.0, le=1.0)


class PriorOnlyGate(BaseModel):
    model_config = STRICT

    k: int = Field(ge=1)
    textbook_wilson_lb: float = Field(ge=0.0, le=1.0)


class Gates(BaseModel):
    model_config = STRICT

    v2: V2Gate
    v3: V3Gate
    v5: V5Gate
    difficulty: DifficultyGate
    prior_only: PriorOnlyGate


class Stages(BaseModel):
    """configs/stages/*.yaml."""

    model_config = STRICT

    grading: Grading
    package: Package
    candidate: Candidate
    gates: Gates


# ------------------------------------------------------------------------------- run-level settings


class Eval(BaseModel):
    """The `evaluate` stage: agents over a folder of finished task folders (PLAN §3 in-process runner)."""

    model_config = STRICT

    dataset: Path | None
    agents: list[str] = Field(min_length=1)  # oracle | nop | a role in models.roles (panel = its models)
    k: int = Field(ge=1)


class Pilot(BaseModel):
    """configs/experiment/pilot.yaml (PLAN §11 M1)."""

    model_config = STRICT

    max_tasks: int = Field(ge=1)
    max_tasks_per_paper: int = Field(ge=1)
    papers: list[str] = Field(min_length=1)


class Config(BaseModel):
    """configs/pipeline.yaml plus its config groups, as composed by Hydra (without the `hydra` node)."""

    model_config = STRICT

    data: Data
    models: Models
    families: Families
    stages: Stages
    run_name: str = Field(min_length=1)
    run_stages: list[str]  # names checked against run.STAGES, which also fixes their order
    budget_usd: float | None = Field(gt=0.0)
    concurrency: int = Field(ge=1)
    eval: Eval
    pilot: Pilot | None = None  # only the pilot experiment sets it

    @model_validator(mode="after")
    def _agents_are_known(self) -> Config:
        for agent in self.eval.agents:
            if agent in BUILTIN_AGENTS:
                continue
            if agent not in Roles.model_fields:
                raise ValueError(
                    f"eval.agents: {agent!r} is neither {' nor '.join(BUILTIN_AGENTS)} nor a role in "
                    f"models.roles ({', '.join(Roles.model_fields)})"
                )
            if getattr(self.models.roles, agent) is None:
                raise ValueError(f"eval.agents: role {agent!r} is unset (null) in models.roles")
        return self


def validate(cfg: DictConfig) -> Config:
    """Resolve the composed Hydra config and check it. Raises ValidationError (see `describe_errors`)."""
    return Config.model_validate(OmegaConf.to_container(cfg, resolve=True))


def describe_errors(error: ValidationError) -> str:
    """One line per problem: the dotted key, then what is wrong with it."""
    lines = []
    for problem in error.errors():
        key = ".".join(str(part) for part in problem["loc"])
        if problem["type"] == "extra_forbidden":
            message = "unknown key (misspelled?)"
        elif problem["type"] == "missing":
            message = "required key is missing"
        elif problem["type"] == "value_error":  # a model validator's own message names the keys
            message = problem["msg"].removeprefix("Value error, ")
        else:
            message = f"{problem['msg']} (got {problem['input']!r:.80})"
        lines.append(f"  {key}: {message}" if key else f"  {message}")  # a Config-level check names its keys
    return "\n".join(lines)
