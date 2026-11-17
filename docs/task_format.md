# Task format and grading contract

The one place that defines what a task folder contains and how it is graded. `grade.py`, the packager,
the in-process runner, the Harbor agent and the hand-built examples all follow it. PLAN.md §4–§5 and §9
give the reasons; this file gives the rules.

## 1. Task folder

```
<task_id>/                  # e.g. 2026.04.25.720790-f4-bwd-01 (PLAN §10 naming)
  instruction.md            # what the solver sees
  task.toml                 # * Harbor config (§8)
  environment/README.md     # * `local` profile only: the placeholder Harbor's task discovery needs (§8)
  solution/solve.sh         # * writes the gold answer (§7)
  tests/
    test.sh                 # * runs grade.py (§7)
    grade.py                # * src/p2c_task_re/grading/grade.py, verbatim
    judge_prompt.md         # * src/p2c_task_re/grading/judge_prompt.md, verbatim
    question.md             # exactly what the solver is sent: instruction.md minus the canary line; the judge's question
    rubric.json             # §4
    gold.json               # §3
```

The author (a person, or the pipeline) writes the unmarked files. The packager writes every `*` file from
them once `grade.py --check` accepts `tests/` and `question.md` equals `instruction.md` as an agent receives
it (below), so edit the inputs and re-sync, never the outputs:
`uv run python -m p2c_task_re.stages.package --sync DIR --split train|dev|test|unassigned
--provenance handbuilt|pipeline [--environment local|singularity|cloud --image IMAGE]`.

`instruction.md`:

~~~markdown
<!-- p2c-canary <GUID from configs/stages/default.yaml> -->
# Background
<the premise set P, as neutral prose>

# Question
<the question>

# Answer format
Reply with exactly one JSON object in this schema; it is saved verbatim to /app/output/answer.json and
graded, including the rationale (at most <cap> words; text beyond the cap is ignored).

```json
<an example object with every field the family requires, placeholder values>
```
~~~

The solver never sees the paper, the triplet JSON, the DOI, the title or any author. A model agent is
sent `instruction.md` with its leading canary line, and the blank lines after it, removed — what Harbor's
`Task.instruction` hands an agent; the in-process runner sends the same text (`evaluate.solver_prompt`), and
`tests/question.md` is exactly that text (`stages.package.strip_canary`), so the judge reads the question
the solver answered.

## 2. Answer envelope

Every answer — the solver's and the gold — is one JSON object:

```json
{"answer": { ...family fields (§5)... }, "rationale": "free text"}
```

A reply wrapped in one markdown fence that encloses all of it is accepted too (§6 step 1).

## 3. `gold.json`

```json
{"task_id": "...", "answer": { ... }, "rationale": "the gold derivation, written as a model answer would be"}
```

## 4. `rubric.json`

```json
{
  "task_id": "2026.04.25.720790-f4-bwd-01",
  "family": "F4",
  "direction": "backward",            // forward | backward | null (F5, F7)
  "tau": 0.70,
  "w_core": 0.60,                     // 0.60 for F1–F4, F6; 0.40 for F5, F7
  "rationale_word_cap": 250,          // 250 for F1–F4, F6; 400 for F5, F7
  "core": { ... },                    // §5
  "criteria": [ ... ]                 // judge criteria, schema below
}
```

Every criterion is graded by the LLM judge:

```json
{
  "id": "c1",                         // unique within the task
  "section": "mechanism",             // free label, mirrors the derivation steps
  "text": "Attributes the loss of expansion to ...",   // self-contained: names entity, direction, context
  "points": 3,                        // integer 0–10; 0 = diagnostic only (notice/act "recognizes cue" items)
  "core": false,                      // true only for F5/F7 core criteria (key entities / essential)
  "capability": "mechanism",          // direction | comparison | control-logic | mechanism | design | uncertainty
  "step": 2,                          // which derivation step it grades, or null
  "grounding": {"source": "paper", "quote": "verbatim sentence from Results or a figure legend"}
                                      // or {"source": "instruction", "quote": "..."} or {"source": "k_premise", "text": "..."}
}
```

Rules a valid rubric satisfies (checked by `grade.py --check`): `tau` and `w_core` are numbers with
`0.30 < w_core < tau`; ids unique; points are integers in 0–10; `core` on a criterion is `true` or `false`;
the non-core criteria have Σ points > 0; F5/F7 have ≥1 core criterion and every other family has none;
`core` (§5) names only fields and values the instruction defines, none of them twice after case-folding.
For F1, F2 and F6 the gold names exactly the declared markers or arms and uses at least two values, so no
constant answer matches it; the gold's rationale and its answer strings each fit the word cap (§6 step 2).

## 5. Family answer fields and `s_core`

Keys and enum values are compared after case-folding and collapsing whitespace. A **missing** entry or an
**unrecognised value** counts as a wrong entry. An **extra or unknown key** (a marker or arm the rubric does
not declare, or a top-level `answer` field the family does not define — in every family, F5 and F7
included) or two keys that fold to the same one is off-contract and sets `s_core = 0`, like an invalid F3
ranking. A wrong **critical** field sets `s_core = 0`. F1/F2/F6 must declare ≥1 critical marker or arm,
and F3 must set `critical_top: true`.

| Family | `answer` fields | `rubric.core` | `s_core` | Chance floor |
|---|---|---|---|---|
| F1 | `{"markers": {"<marker>": "up\|down\|unchanged"}}` | `{"type": "marker_directions", "classes": ["up","down","unchanged"], "markers": [...], "critical": [...]}` | mean, over the classes present in gold, of the recall of that class (balanced accuracy) | ~1/3 |
| F2, F6 | `{"arms": {"<arm>": "<value>"}}` (F6 may add `"state": "..."`, judged) | `{"type": "arm_enum", "values": [...], "arms": [...], "critical": [...]}` | fraction of gold arms whose value matches | 1/len(values) |
| F3 | `{"ranking": [["<arm>", ...], ...]}` — tie sets, strongest effect first, each arm exactly once | `{"type": "ranking", "arms": [...], "critical_top": true}` | over pairs in *different* gold tie sets: fraction ordered the same way by the answer (an answer tie on such a pair is not concordant). Invalid ranking → 0. `critical_top`: the answer's first tie set must equal gold's | 0.5 |
| F4 | `{"choice": "<option>"}` | `{"type": "choice", "options": [...]}` | 1 if the choice matches, else 0. A constant policy is suite-level for F4 (one field per task) | 1/len(options) |
| F5 | `{"mechanism_primary": "<one sentence>", "key_entities": [...]}` | `{"type": "judged"}` | fraction of `core: true` criteria satisfied | ~0 |
| F7 | `{"experiment": "<free text>"}` | `{"type": "judged"}` | fraction of `core: true` criteria satisfied | ~0 |

## 6. Scoring

- `s_judge` = Σ points of satisfied non-core criteria / Σ points of non-core criteria.
- `s = w_core · s_core + (1 − w_core) · s_judge`, in [0, 1].
- **pass** = `s ≥ tau` and `s_core ≥ 0.9` and `s_judge ≥ 0.5`.
- `reward.json` = `{"reward": s, "core": s_core, "judged": s_judge}` (flat, numeric). `pass` is not stored.

**Before any judge call:**
1. **Unusable answer** — a leading byte-order mark and surrounding whitespace are stripped, then exactly
   one markdown fence enclosing the whole reply (```` ``` ```` or ```` ```json ````, info string on the opening line). Nothing else is
   extracted: prose before or after the object, two objects, two fenced blocks, an unclosed or one-line
   fence all stay unusable. If the file is missing, the text is not JSON, or it has no object `answer` →
   write all-zero rewards, `status = "invalid_answer"`, no judge calls. This is a valid 0, not an error.
   `answer_format` records `bare`, `fenced` or `invalid`; a fenced answer grades exactly as unfenced.
2. **Truncate** the rationale to `rationale_word_cap` whitespace-separated words, and the strings of
   `answer` likewise: taken in order, all fields together, they share another `rationale_word_cap` words,
   so a free-text field (`mechanism_primary`, `key_entities`, `state`, `experiment`) cannot carry text
   past the cap. This is what the judge sees; `s_core` is scored on the full answer. `truncated` records
   whether either was cut.
3. **Response text** shown to the judge = `json.dumps(truncated answer, indent=2, ensure_ascii=False)` + `"\n\nRationale:\n"` + truncated rationale.
4. **Masking** — every run of ≥10 consecutive word tokens (`\w+`, lower-cased) of the response text that
   also occurs in `question.md` is a masked span; the answer fields are masked like the rationale.
5. **F5 gate** — `mechanism_primary` must be one non-empty string (a list of candidates fails it);
   otherwise all rewards are 0 with no judge call (`status = "no_primary_mechanism"`).

**Judge protocol** (`judge_prompt.md` holds the text):
- One criterion per call: the judge sees `question.md`, the response text and that one criterion — never the
  others, never the gold.
- The judge returns `{"verdict": "yes" | "no", "evidence": "<verbatim quote from the response>"}`.
- A `yes` counts only if the evidence occurs in the response text (whitespace-normalised and case-folded —
  judges re-case a quote's first letter) and does not lie entirely inside masked spans; otherwise it is
  recorded as `no` with the reason. The mask is compared word token by word token wherever the quote's
  tokens occur in the response (its first and last token may be cut mid-word): the quote is an echo only
  if every such occurrence is masked throughout, so an echoed sentence's closing period does not lift it
  out, and a phrase the solver also wrote in its own words counts. The consistency check's quote is
  recorded but not enforced.
- **Consistency check**, once per response, same sampling: "Does the rationale's conclusion contradict the
  structured answer fields?" A majority `yes` sets every judge criterion to 0 (`status = "contradiction"`).
- **Sampling:** `judge_samples` (3) per criterion, majority vote (a tie counts as `no`). A sample that errors or returns unparseable
  JSON is retried up to `judge_retries` times; a refusal or a sample still failing goes to the regrade judge.
  If a criterion still lacks `judge_samples` valid votes, grading fails: exit code 2, no `reward.json`, and
  the message ends with the last failed attempt's reason (a missing API key shows up there).
- Judge calls run concurrently, at most `judge_concurrency` at a time.

**`grading.json`** (written next to `reward.json`): `task_id`, `status`, `answer_format`, `passed`, `s`,
`s_core`, `s_judge`, `core_detail` (per-field gold / answer / correct), `criteria` (per criterion: id,
points, core, final verdict, every vote with model, counted `verdict`, the judge's `raw_verdict`, evidence,
valid, reason), `consistency`, `truncated`, `masked_spans` (the shared runs as text), `judge_models`
(`[primary, regrade]`), `judge_usage` (`{model: {calls, input_tokens, output_tokens, refusals}}` over every
judge API call, refusals included; `{}` when no real judge ran), `grade_version` (`"1.2"`).

## 7. `grade.py`, `test.sh`, `solve.sh`

`grade.py` imports only the standard library, plus `anthropic` / `openai` lazily when a real judge is built.

```
python3 grade.py --tests-dir DIR --answer FILE --out-dir DIR [--judge-model M] [--regrade-model M]
python3 grade.py --check --tests-dir DIR          # validate rubric.json + gold.json, grade the gold with no judge
```

Defaults come from env: `JUDGE_MODEL` (default `claude-sonnet-5-5`), `JUDGE_REGRADE_MODEL` (default
`gpt-6.1-sol`), `JUDGE_SAMPLES` (3), `JUDGE_EFFORT` (`medium`), `JUDGE_RETRIES` (2), `JUDGE_CONCURRENCY` (8).
`judge_prompt.md` is read from the tests dir, else from beside `grade.py`. Provider is inferred from the model name (`claude-*` →
Anthropic, `gpt-*` / `o*` → OpenAI). Exit 0 = rewards written; 2 = judge failure; 1 = bad tests dir.

Python API: `grade(answer_text: str | None, tests_dir: Path, judge, regrade_judge=None, *, samples=3,
retries=2, concurrency=8, usage: JudgeUsage | None = None) -> dict` (the `grading.json` content). A judge
is any object with `name: str` and `complete(system: str, user: str) -> str | None` (`None` = refusal).
`make_judge(model, effort="medium", on_usage=None)` builds a real one; it calls
`on_usage(model, input_tokens, output_tokens, refusal)` after every API call (a call that raised reports
nothing). `JudgeUsage().add` is such a callback, and the same object passed as `usage` becomes
`judge_usage`. The CLI does exactly this, so a Harbor verifier's grading.json carries its own judge spend.

```bash
# tests/test.sh
#!/usr/bin/env bash
set -euo pipefail
TESTS_DIR="${TESTS_DIR:-/tests}"
mkdir -p "${VERIFIER_DIR:-/logs/verifier}"
python3 "$TESTS_DIR/grade.py" --tests-dir "$TESTS_DIR" \
  --answer "${ANSWER_PATH:-/app/output/answer.json}" --out-dir "${VERIFIER_DIR:-/logs/verifier}"
```

```bash
# solution/solve.sh
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${OUTPUT_DIR:-/app/output}"
cat > "${OUTPUT_DIR:-/app/output}/answer.json" <<'P2C_GOLD'
{"answer": {...}, "rationale": "..."}
P2C_GOLD
```

The env overrides let the in-process runner execute the same scripts outside a container.

## 8. `task.toml` and environment profiles

The packager's `--environment` (with `--image`) picks the profile; configs/stages/default.yaml's
`package` block holds the values the M1 S10 stage will pass it. The profile decides `[environment]`, the
verifier's network and `environment/README.md`; every other file is identical across profiles.

| Profile | Runs on | `[environment]` | `[verifier]` network | `environment/README.md` | Isolation |
|---|---|---|---|---|---|
| **`local`** (default now) | the host, no container: `p2c_task_re.harbor_env:LocalEnvironment` | empty | omitted (= public) | written | none |
| `singularity` (later) | Harbor's singularity environment, once ILC installs Apptainer | `docker_image = "<absolute path ending .sif>"` (Harbor would pull any other value) | omitted (= public) | none | container, host network |
| `cloud` (later) | Modal / Daytona | `docker_image = "<OCI ref @sha256:…>"`, `network_mode = "no-network"` | `network_mode = "allowlist"`, `allowed_hosts = ["api.anthropic.com", "api.openai.com"]` | none | full |

- **`environment/README.md`** is a two-line placeholder. Harbor 0.26 discovers a task folder only if it
  sets `docker_image` or has an `environment/` folder, and `local` has no image. Under a container profile
  Harbor uploads a non-empty `environment/` into the workdir (`/app`), so the packager writes none there
  and removes one an earlier `local` packaging left.
- `local` and `singularity` have no network isolation, which is acceptable only because a v1 task never
  executes model-written code. The packager refuses to emit a task that gives the agent tools (MCP
  servers, skills) under either.

```toml
schema_version = "1.4"
artifacts = ["/app/output"]

[task]
name = "p2c/<task_id>"
version = "1.0.0"

[metadata]
family = "F4"
direction = "backward"            # omitted for F5/F7: rubric.json's null, and TOML has no null
paper_id = "2026.04.25.720790"
split = "train"                   # train | dev | test | unassigned
provenance = "handbuilt"          # handbuilt | pipeline

[environment]                     # profile-specific (table above); empty for `local`

[agent]
timeout_sec = 1800

[verifier]                        # cloud adds network_mode = "allowlist" and allowed_hosts (table above)
environment_mode = "separate"
timeout_sec = 1800

[verifier.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
OPENAI_API_KEY = "${OPENAI_API_KEY}"
JUDGE_MODEL = "${JUDGE_MODEL:-claude-sonnet-5-5}"
JUDGE_REGRADE_MODEL = "${JUDGE_REGRADE_MODEL:-gpt-6.1-sol}"
JUDGE_SAMPLES = "${JUDGE_SAMPLES:-3}"
JUDGE_EFFORT = "${JUDGE_EFFORT:-medium}"
JUDGE_RETRIES = "${JUDGE_RETRIES:-2}"
JUDGE_CONCURRENCY = "${JUDGE_CONCURRENCY:-8}"
```

`[verifier.env]` declares every variable grade.py reads (§7), at grade.py's defaults, so a task states its
own grading settings and a container verifier picks host values up. `--ve NAME=value` overrides one for a
`harbor run` or a `harbor job regrade`.

**Running the `local` profile.**

```bash
uv run harbor run -p <dataset> -e p2c_task_re.harbor_env:LocalEnvironment -a <agent> -y -o <jobs> …
uv run harbor job regrade <jobs>/<job> -p <dataset> -e p2c_task_re.harbor_env:LocalEnvironment -o <jobs> [--ve …]
```

- `-y` is required: Harbor asks to confirm that `[verifier.env]` reads host variables, and a run that
  cannot ask aborts.
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` must both be set for every agent, `nop` and `oracle` included:
  `harbor run` refuses to start while a declared variable without a default is unset, and a regrade's
  trials fail. A placeholder value will do when no judge is called.
- `LocalEnvironment` gives each environment a host folder standing in for `/app`, `/tests`, `/solution`
  and `/logs`; it maps container paths in file operations, in the `exec` command string and `cwd`, and
  through `TESTS_DIR`, `ANSWER_PATH`, `VERIFIER_DIR` and `OUTPUT_DIR` (§7), and puts the project venv's bin
  folder first on `PATH`. It refuses a task with a `docker_image`, an `environment/Dockerfile` or
  `docker-compose.yaml`, MCP servers or skills, or a `no-network` / `allowlist` network, and it refuses
  every agent but `oracle`, `nop` and `p2c_task_re.harbor_agent:SingleShotAgent` (a regrade runs no
  agent). It starts each environment from an empty folder, and keeps the trial folders writable by their
  owner only (Harbor opens them to everyone for a container's user). The jobs folder may not contain a
  space or a shell metacharacter: host paths are pasted into command strings.
- `--ek` options (`harbor job regrade` takes none, so the defaults must do):

  | `--ek` | Default | Effect |
  |---|---|---|
  | `root=DIR` | the trial's own folder | where the per-environment folders go |
  | `path=DIR[:DIR]` | empty | extra `PATH` entries, ahead of the venv's bin folder |
  | `auto_mounting=false` | `true` | keep `/logs` inside the environment folder and let Harbor copy it back, instead of mapping `/logs/{agent,verifier,artifacts}` onto the trial's folders |
