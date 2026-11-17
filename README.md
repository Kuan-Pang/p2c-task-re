# p2c-task-re

Turns papers2cells perturbation triplets and their source papers into Harbor benchmark tasks. Each
task gives the solver a set of revealed findings and asks for a withheld result, and comes with a hidden
gold answer, a rubric and a standalone grader (deterministic checks plus an LLM judge). v1 tasks are
single-turn with no tools. [PLAN.md](PLAN.md) has the design and milestones, and
[docs/task_format.md](docs/task_format.md) is the binding contract for a task folder and its grading.
The repo is at M0, the harness: data readers, the LLM client and spend ledger, the grader, the packager,
the in-process runner, a container-free Harbor environment and agent, and three hand-built tasks. The
generation stages arrive in M1.

## Setup

```bash
uv sync --extra harbor      # Python 3.12; the harbor extra is needed only for Harbor runs
source ~/.api_keys          # ANTHROPIC_API_KEY, OPENAI_API_KEY: needed only for real model or judge calls
```

Run every command from the repo root.

## Commands

**Pipeline: evaluate a dataset in-process.** Hydra composes `configs/`. The config is validated before
anything runs, so a misspelled key in a YAML file fails with one line per bad key.

```bash
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks eval.agents=[nop]           # free, offline
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks eval.agents=[oracle,nop]    # judge calls
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks eval.agents=[panel] eval.k=3
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks --cfg job     # print the config, run nothing
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks 'models.roles.judge=${models.gpt}'
uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks hydra.run.dir=outputs/runs/eval/<timestamp>
```

A run writes outputs/runs/eval/<timestamp>/: `config.json`, `calls.jsonl` (the ledger: every solver and
judge call, with its cost), `pipeline.log`, `eval/summary.json` and
`eval/<task_id>/<agent>/<NN>/{answer,reward,grading}.json`. Passing an existing folder as `hydra.run.dir`
resumes it: a finished stage or trial is not run again. `nop` is free; `oracle` calls the judge; a model
agent (any role in `models.roles`; `panel` means each of its models) calls the solver and the judge.
`budget_usd=…` caps the run's spend, and `stages.grading.judge_samples=1` cuts the judge calls to a third.
`+experiment=pilot` and `+experiment=full` stop at once and say which milestone adds each of their stages.

**Grade one answer** with the same `grade.py` every task ships in its `tests/`:

```bash
uv run p2c-grade --check --tests-dir examples/tasks/2026.04.25.720790-f4-bwd-01/tests   # rubric + gold, no judge
uv run p2c-grade --tests-dir examples/tasks/2026.04.25.720790-f4-bwd-01/tests \
    --answer tests/fixtures/ablations/2026.04.25.720790-f4-bwd-01/gold_paraphrase.json --out-dir outputs/graded
```

The second writes `reward.json` and `grading.json` to `outputs/graded/`. The judge settings come from
`JUDGE_MODEL`, `JUDGE_REGRADE_MODEL`, `JUDGE_SAMPLES`, `JUDGE_EFFORT`, `JUDGE_RETRIES` and
`JUDGE_CONCURRENCY` (docs/task_format.md §7).

**Sync the example tasks** after editing an `instruction.md`, `question.md`, `rubric.json` or `gold.json`,
or the grader, the judge prompt or a template. It rewrites every generated file (`task.toml`,
`environment/README.md`, `solution/solve.sh`, `tests/test.sh`, `tests/grade.py`, `tests/judge_prompt.md`);
a second run changes nothing, and tests/test_package.py fails until the examples are in sync.

```bash
uv run python -m p2c_task_re.stages.package --sync examples/tasks --split train --provenance handbuilt
```

**Harbor, `local` profile: no container.** Each trial runs on the host, in a folder of its own, with no
isolation. That is acceptable only because a v1 task never executes model-written code.

```bash
uv run harbor run -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment -a nop -y -o outputs/jobs
uv run harbor run -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment -a oracle -y -o outputs/jobs
uv run harbor run -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment \
    -a p2c_task_re.harbor_agent:SingleShotAgent -m claude-sonnet-5-5 -k 1 -y -o outputs/jobs
uv run harbor job regrade outputs/jobs/<job> -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment \
    -o outputs/jobs
```

- Both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` must be set for every agent, `nop` included, because
  task.toml declares them for the verifier; for `nop` any placeholder value will do.
- `-y` accepts Harbor's prompt that the verifier reads host variables; without it the run stops there.
- `--ve JUDGE_SAMPLES=1` (or any other `JUDGE_*` variable) changes the judge for a run or a regrade.
- Each trial's `verifier/grading.json` holds the grader's full record, its own judge spend included
  (`judge_usage`); the agent's tokens and cost are in the trial's `result.json`.
- `LocalEnvironment` runs only `oracle`, `nop` and `SingleShotAgent`, the agents that execute nothing a
  model wrote, and refuses a jobs folder (`-o`) whose path holds a space.
- `scripts/harbor.sbatch` runs one shard of this on Slurm; submit it from the repo root (its jobs go to
  `outputs/jobs`). The `singularity` and `cloud` profiles (PLAN §9, docs/task_format.md §8) are packager
  options (`--environment`, `--image`), kept for later.

## Layout

```
configs/              Hydra values: pipeline.yaml, data/, models/ (the role roster), families/, stages/, experiment/
src/p2c_task_re/      cli.py → config.py (schema) → run.py (stage order); llm.py, ledger.py; evaluate.py (runner)
  grading/            grade.py + judge_prompt.md, standalone, copied into every task's tests/
  stages/             package.py (S10) and its templates/
  data/               extraction-JSON, .meca and JATS readers
  harbor_env.py       LocalEnvironment;  harbor_agent.py  SingleShotAgent
examples/tasks/       three hand-built tasks (F2 forward, F4 backward, F5); see its README
tests/                offline tests and fixtures; tests/live/ calls real APIs or reads /lfs
docs/                 task_format.md and the M0 notes
containers/           p2c-base.def, the Apptainer recipe for the singularity profile
scripts/              harbor.sbatch (one Harbor shard), probe_refusals.py (the M0 refusal probe)
outputs/              run folders (gitignored)
```

## Tests

```bash
uv run pytest -q                                   # offline: no API key, no /lfs, no network
uv run ruff check src tests scripts

source ~/.api_keys && uv run pytest -m live tests/live/test_llm_live.py -q -s        # both providers, < $0.05
source ~/.api_keys && uv run pytest -m live tests/live/test_judge_live.py -q -s      # the real judges on F2
source ~/.api_keys && uv run pytest -m live tests/live/test_evaluate_live.py -q -s   # in-process, <= $2
source ~/.api_keys && uv run pytest -m live tests/live/test_harbor_live.py -q -s     # harbor CLI, <= $1
uv run pytest -m live tests/live/test_corpus_live.py -q                              # reads /lfs, no API
```

Live tests are marked `@pytest.mark.live` and are deselected by default. Each one calls a real API (which
costs money) or reads the /lfs corpus, so run them one file at a time. The two with a budget stop before
they would pass it.
