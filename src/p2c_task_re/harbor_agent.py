"""SingleShotAgent: the v1 solver as a Harbor agent, one model call made on the Harbor host (PLAN §9).

Harbor hands the agent the instruction (instruction.md minus its canary line). The agent sends it once, as
the only user message, through `p2c_task_re.llm.complete` (no system prompt, no tools, no schema, exactly
as the in-process runner's model agent does) and uploads the completion text verbatim to
/app/output/answer.json with `environment.upload_file`. It uploads on a refusal and on empty output too,
so those grade as a valid 0 instead of dropping out (PLAN §3). A call that still fails after the SDK's own
retries raises Harbor's `ApiError`, which `harbor run -r N` retries and which otherwise leaves the trial
without a valid reward.

Model: the provider comes from the name as grade.py reads it (`claude-…` → Anthropic, `gpt-…` / `o<digit>…`
→ OpenAI); a LiteLLM-style `anthropic/` or `openai/` prefix is dropped. When configs/models/default.yaml
lists the model (its `panel` entry first), that entry's effort, max_tokens and price are used; otherwise
the RoleSpec defaults with no price. The roster file must exist (run from the source checkout), so a run
never silently records no cost.
`--effort LEVEL` (the same as `--ak reasoning_effort=LEVEL`) overrides the effort.

The agent context (result.json `agent_result`) records the tokens, the cost, the usage per answering model,
and as metadata the record the in-process runner writes as solver.json: the model that answered, refusal,
refusal_category, fallback_used, stop_reason, tokens, cost, latency. The text itself is in the trial's
agent/answer.json and artifacts/app/output/answer.json.

    uv run harbor run -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment \\
        -a p2c_task_re.harbor_agent:SingleShotAgent -m claude-sonnet-5-5 -k 1 -y
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import ApiError
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage
from omegaconf import OmegaConf
from pydantic import Field

from p2c_task_re import __version__, llm
from p2c_task_re.evaluate import SOLVER_ROLE
from p2c_task_re.grading.grade import provider_of
from p2c_task_re.llm import Completion, Effort, Provider, RoleSpec

__all__ = ["SingleShotAgent", "SingleShotOptions", "roster_specs", "solver_spec"]

ANSWER_PATH = "/app/output/answer.json"  # where instruction.md promises the reply is saved
ROSTER = Path(__file__).resolve().parents[2] / "configs" / "models" / "default.yaml"


class SingleShotOptions(AgentOptions):
    """The `--ak` options of SingleShotAgent."""

    reasoning_effort: Effort | None = Field(
        default=None, description="Override the model's effort (default: the roster's, else high)."
    )


class SingleShotAgent(BaseAgent):
    """Send the instruction once, upload the reply as the answer. No tools, no turns."""

    capabilities = AgentCapabilities(reasoning_effort=True)
    options_model = SingleShotOptions
    options: SingleShotOptions

    def __init__(self, logs_dir: Path, model_name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self.spec = solver_spec(model_name, self.options.reasoning_effort)

    @staticmethod
    def name() -> str:
        return "p2c-single-shot"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: inference runs on the Harbor host."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        try:
            completion = await asyncio.to_thread(llm.complete, self.spec, instruction, role=SOLVER_ROLE)
        except llm.LLMError as exc:
            raise ApiError(str(exc)) from exc
        record_completion(context, completion)
        answer = self.logs_dir / "answer.json"
        answer.write_text(completion.text, encoding="utf-8")
        await environment.upload_file(answer, ANSWER_PATH)


def record_completion(context: AgentContext, completion: Completion) -> None:
    """Tokens, cost and the answering model into Harbor's agent context."""
    context.n_input_tokens = completion.input_tokens
    context.n_output_tokens = completion.output_tokens
    context.cost_usd = completion.cost_usd
    context.model_usage = {
        completion.model: ModelUsage(
            n_input_tokens=completion.input_tokens,
            n_output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
        )
    }
    context.metadata = completion.model_dump(exclude={"text", "parsed"})


def solver_spec(model_name: str | None, effort: Effort | None = None) -> RoleSpec:
    """The RoleSpec for Harbor's `-m`: the roster's entry for that model, else one by name prefix."""
    if not model_name:
        raise ValueError("SingleShotAgent needs a model: -m claude-… or -m gpt-…")
    model = model_name.rpartition("/")[2]
    spec = roster_specs().get(model)
    if spec is None:
        provider = provider_of(model)
        if provider is None:
            raise ValueError(f"cannot tell which provider serves {model!r}: expected claude-… or gpt-…")
        spec = RoleSpec(provider=cast(Provider, provider), model=model)
    return spec.model_copy(update={"effort": effort}) if effort else spec


def roster_specs(path: Path = ROSTER) -> dict[str, RoleSpec]:
    """Every model the roster names, by model id; a model's `panel` entry wins."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: SingleShotAgent reads its models and prices there")
    roster = llm.load_roster(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    specs: dict[str, RoleSpec] = {}
    for role in ("panel", *roster):
        value = roster.get(role)
        for spec in value if isinstance(value, list) else [value]:
            if spec is not None:
                specs.setdefault(spec.model, spec)
    return specs
