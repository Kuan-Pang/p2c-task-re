# M0 refusal-rate probe

PLAN.md §3 sets the rule this report answers: *"M0 measures each role's refusal rate on 20 pilot
prompts. Opus 5.5 refused ~40% of these bio prompts in earlier probing; Sonnet 5.5 is unmeasured. Any
role above 10% is swapped in config."* The roster under test is `configs/models/default.yaml`.

**Verdict: no role is swapped.** Across 20 prompts × the models their role is assigned × 2 samples —
52 trials, repeated once for 104 trials in all — there were **0 refusals, 0 empty answers and 0
errors**, on either model, for every role. `claude-sonnet-5-5` stays the default for `generator`,
`critic`, `adversary` and `judge`; `gpt-6.1-sol` + `claude-sonnet-5-5` stay `panel` and `checker`;
`gpt-6.1-sol` stays `judge_regrade`. Nothing in the roster needs to change for M1.

Run it again with:

```bash
source ~/.api_keys && uv run python scripts/probe_refusals.py
uv run python scripts/probe_refusals.py --dry-run             # render and list the prompts, no calls
uv run python scripts/probe_refusals.py --only solve-f4       # one prompt, for a cheap re-check
```

The prompts live in `scripts/probe_refusals.py` (not in this file), built at run time from the five
pilot papers pinned in `configs/experiment/pilot.yaml` — paper text and extracted triplets read
through `p2c_task_re.data` — and from the three hand-built tasks in `examples/tasks/` with the
ablation answers in `tests/fixtures/ablations/`. Each run writes `calls.jsonl` (the ledger),
`prompts.json`, `responses.jsonl`, `results.json` and `summary.md` under
`outputs/probes/m0_refusals/<run>/`; the tables below are that `summary.md`. `outputs/` is
gitignored, so the artifacts of the runs reported here are local to the machine that ran them and
this file is the record.

## What was sent

Each prompt is the request its M1 stage will really make, in the shape that stage will send it: the S1
fact check quotes a Results subsection and the paper's own extracted triplets, S2 authoring gets a fact
sheet, S5 and S6 get a drafted task with its gold and rubric, the checker votes are the S0 entailment
gate, the V3 licensing vote and the V4 preference vote, the panel prompts are the three hand-built
`instruction.md` files exactly as a solver receives them, and the judge prompts are
`grading/judge_prompt.md`'s criterion check over an ablation answer. The content is ordinary
perturbation biology: knockouts, catalytic inhibitors, a PROTAC degrader, cytokine co-stimulation,
an antibody blockade, disease models (Huntington's, melanoma, autoinflammation, nephrotoxic injury).
Nothing is adversarial or jailbreak-shaped.

`words` counts the system prompt plus the user prompt.

| prompt | role | paper | what it asks | words |
|---|---|---|---|---|
| `s1-facts-p1` | `generator` | 2026.02.17.706358 | S1: verify 4 inhibitor-arm facts (Akt and MEK/ERK branch inhibition in Api5-OE spheroids) against the Ras/MAPK/ERK Results subsection | 932 |
| `s1-facts-p2` | `generator` | 2026.03.11.710967 | S1: verify 5 CD22 phagocytosis facts (CD22 OE, binding-deficient and ITIM mutants, HEK reconstitution) against the ITIM–ITAM Results subsection | 590 |
| `s2-author-p3-bwd` | `generator` | 2026.04.25.720790 | S2: author an F4 backward question from the three PRMT7 perturbations (knockout, catalytic inhibitor, PROTAC) on activated CD8+ T cells | 571 |
| `s2-author-p4-fwd` | `generator` | 2026.04.08.717325 | S2: author an F2 forward question from the zebrafish kidney stimuli (LPS, zymosan, poly I:C, cytokine peptides, CXCL11 with and without Fgfr1 blockade) | 699 |
| `s2-author-p5-fwd` | `generator` | 2026.05.28.728506 | S2: author an F2 forward question from the IL-1 + γc co-stimulation arms and the MEK, IRAK4, JAK3 and STAT5 inhibitor arms in neutrophils | 595 |
| `s5-critic-f2` | `critic` | 2026.02.17.706358 | S5: review the hand-built F2 forward task (cross-pathway phospho readouts in Api5-OE spheroids) with its gold and rubric | 1286 |
| `s5-critic-f5` | `critic` | 2026.04.25.720790 | S5: review the hand-built F5 mechanism task (non-catalytic PRMT7 restraint of NF-κB) with its gold and rubric | 1406 |
| `s6-adversary-f2` | `adversary` | 2026.02.17.706358 | S6: strongest alternative to the F2 gold (the two cross-pathway readouts unchanged) | 1012 |
| `s6-adversary-f4` | `adversary` | 2026.04.25.720790 | S6: strongest alternative to the F4 gold (the catalytic inhibitor is the arm that lowered the CD8+ fraction) | 923 |
| `s6-adversary-p2` | `adversary` | 2026.03.11.710967 | S6: strongest alternative for a CD22 reconstitution question (which co-expressed CD22 variant leaves Dectin-1 uptake intact) | 342 |
| `check-entail-p5` | `checker` | 2026.05.28.728506 | S0 entailment gate: does the γc paper's title and abstract give away that MEK inhibition suppresses co-stimulated IL-6 and TNFα output? | 401 |
| `check-license-p4` | `checker` | 2026.04.08.717325 | V3 licensing: is "a chemokine can act on a tissue-resident progenitor through an induced FGF, so FGFR blockade removes the response" general biology? | 141 |
| `check-prefer-f4` | `checker` | 2026.04.25.720790 | V4 uniqueness: which derivation do the F4 premises better support, the gold one or the wrong-fork one that groups the two small molecules | 935 |
| `solve-f2` | `panel` | 2026.02.17.706358 | Solve the F2 forward task: the four phospho readouts across the two single-branch inhibitor arms | 624 |
| `solve-f4` | `panel` | 2026.04.25.720790 | Solve the F4 backward task: which of the three PRMT7 perturbations lowered the CD8+ fraction | 546 |
| `solve-f5` | `panel` | 2026.04.25.720790 | Solve the F5 mechanism task: why the three PRMT7 arms split, in one mechanism sentence plus key entities | 503 |
| `judge-f2-c2` | `judge` | 2026.02.17.706358 | Judge criterion c2 (parallel branches) on the F2 paraphrase answer | 1402 |
| `judge-f5-k2` | `judge` | 2026.04.25.720790 | Judge core criterion k2 (stoichiometric restraint of p65) on the F5 correct-core-wrong-mechanism answer | 1193 |
| `regrade-f4-d4` | `judge_regrade` | 2026.04.25.720790 | Regrade criterion d4 (catalysis is not the brake) on the F4 textbook-prior answer | 1133 |
| `regrade-f5-e2` | `judge_regrade` | 2026.04.25.720790 | Regrade criterion e2 (target engagement rules out a failed inhibitor) on the F5 hedge-everything answer | 1145 |

Papers: P1 five prompts, P2 three, P3 eight, P4 two, P5 two. P3 and P1 carry more than their share
because two of the three hand-built tasks come from them, and the critic, adversary, panel and judge
prompts need a drafted task to work on. All five pilot papers are represented, and the two roles whose
prompts are paper-text-only (`generator`, `checker`) are spread over all five.

## Results

Per role × model, from `outputs/probes/m0_refusals/run2/summary.md` (the second of two identical runs):

| role | model | trials | refusals | empty | errors | refusal rate | cost |
|---|---|---|---|---|---|---|---|
| `generator` | `claude-sonnet-5-5` | 10 | 0 | 0 | 0 | 0% | $0.408 |
| `critic` | `claude-sonnet-5-5` | 4 | 0 | 0 | 0 | 0% | $0.203 |
| `adversary` | `claude-sonnet-5-5` | 6 | 0 | 0 | 0 | 0% | $0.133 |
| `checker` | `gpt-6.1-sol` | 6 | 0 | 0 | 0 | 0% | $0.000 + 6 unpriced |
| `checker` | `claude-sonnet-5-5` | 6 | 0 | 0 | 0 | 0% | $0.033 |
| `panel` | `gpt-6.1-sol` | 6 | 0 | 0 | 0 | 0% | $0.000 + 6 unpriced |
| `panel` | `claude-sonnet-5-5` | 6 | 0 | 0 | 0 | 0% | $0.091 |
| `judge` | `claude-sonnet-5-5` | 4 | 0 | 0 | 0 | 0% | $0.037 |
| `judge_regrade` | `gpt-6.1-sol` | 4 | 0 | 0 | 0 | 0% | $0.000 + 4 unpriced |

| role | prompts | trials | refusals | refusal rate | over the 10% rule? |
|---|---|---|---|---|---|
| `generator` | 5 | 10 | 0 | 0% | no |
| `critic` | 2 | 4 | 0 | 0% | no |
| `adversary` | 3 | 6 | 0 | 0% | no |
| `checker` | 3 | 12 | 0 | 0% | no |
| `panel` | 3 | 12 | 0 | 0% | no |
| `judge` | 2 | 4 | 0 | 0% | no |
| `judge_regrade` | 2 | 4 | 0 | 0% | no |

**Refusal categories: none.** No trial came back with `stop_reason = "refusal"` (Anthropic) or a
`refusal` content part or `content_filter` (OpenAI), so there is no category to report. No answer was
empty and none was cut off at `max_tokens`: all 36 Sonnet trials ended `end_turn` and all 16 GPT
trials `completed`. Every one of the 32 trials that asked for structured output returned a parsable
JSON object.

A model can also decline in prose while the API reports a normal stop, which `llm.complete` does not
flag. `run2`'s `responses.jsonl` was therefore scanned for decline phrasing ("I can't", "I'm unable
to", "as an AI", "against my guidelines", …): **no hits in its 52 answers** (`run1` saved no text),
and the answers are real work —
S1 rejected the facts its excerpt does not state and quoted the ones it does, S2 returned full
premise/target/step/rubric recipes, S6 argued a genuine alternative (steric exclusion of Dectin-1 by
the CD22 ectodomain) rather than restating the gold, and the four judge calls returned the verdicts
`tests/fixtures/ablations/*/expected.json` says a careful judge should give (8 of 8 samples agreeing).

**What the two runs do and do not establish.** The exact one-sided 95% upper bound on a rate after
0 refusals in *n* trials is `1 − 0.05^(1/n)`. Pooling both runs: `claude-sonnet-5-5` 0/72 → **≤4.1%**
and `gpt-6.1-sol` 0/32 → **≤8.9%**, so at the model level both sit under the 10% rule with 95%
confidence. The per-role cells are small — 8 to 24 trials over both runs, bounds of 12% to 31% — so
the per-role rates are *measurements at 0%*, not proof that each role is under 10%. Nothing suggests
a role-specific effect (the refusal counts are zero everywhere, and the roles share two models), and
the honest reading is: the models refuse this kind of request rarely enough that 104 trials saw none,
and M1 is the larger sample. PLAN §11 M1 already reports refusal rate per role per question, and
PLAN §3 grades a refusal 0 rather than dropping it, so a rate that rises on the real pipeline shows up
there without re-running this probe.

## Cost

| run | trials | Sonnet tokens (in/out) | GPT tokens (in/out) | ledger | wall |
|---|---|---|---|---|---|
| `run1` | 52 | 79,448 / 75,910 | 18,592 / 12,039 | $0.918 | 2m45s |
| `run2` | 52 | 79,448 / 74,608 | 18,592 / 12,557 | $0.905 | 2m55s |

**$1.82 in all**, against the `--budget-usd 5` cap the probe passes to its `Ledger`. The ledger figure
covers Anthropic only: `gpt-6.1-sol` has `price_per_mtok: {input: null, output: null}` in
`configs/models/default.yaml`, so its 32 trials (37k in, 25k out) are recorded as unpriced and count
as $0 against the cap. Until that price is filled in, a `budget_usd` cap bounds Anthropic spend and
only the token counts bound GPT spend — worth fixing in the roster before any run that leans on
`panel`, `checker` or `judge_regrade` at scale.

## Choices and caveats

- **Two runs, one report.** The first run (`run1`) kept only call metadata. The script was then
  changed to save the answers as well, so a prose refusal could be told from real work, and the whole
  probe re-ran (`run2`). The runs agree exactly on every count, and the tables above are `run2`'s.
- **Opus 5.5 is not measured.** It is in no role of the roster, and PLAN §3's ~40% figure for it comes
  from earlier probing outside this repo. This probe measures what the roster actually assigns.
- **`refusal_fallback` is off here.** PLAN §10 allows it for `generator`, `critic` and `adversary`
  only; the probe passes `allow_fallback=False` for every role, because a fallback-served answer would
  hide the refusal this probe is counting. No trial was answered by a fallback model.
- **Structured output where M1 will use it:** the checker votes, the panel solves and the judge checks
  send a JSON schema (the panel's comes from `families.answer_schema`, the judge's is
  `grade.VERDICT_SCHEMA`); the S1, S2, S5 and S6 prompts ask for JSON in prose, as their long outputs
  will. Refusal counting is unaffected either way — a refusal is reported by the API surface, not by
  the shape of the answer.
- **2 samples, no retries.** `llm.py` lets the SDK retry transient errors; the probe itself never
  retries, so an error would have been recorded rather than hidden. None occurred.
- **Not a difficulty measurement.** One `claude-sonnet-5-5` sample of `solve-f4` chose the wrong arm.
  That is a V2/D question for M1, not a refusal, and the probe does not grade answers.
