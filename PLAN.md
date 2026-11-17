# papers2cells → Harbor task-synthesis pipeline (v1: no-tools reasoning)

**In:** perturbation triplets (`*.extraction.json`) + source papers (`.meca` → `paper.txt`).
**Out:** a Harbor dataset — per task an `instruction.md`, a hidden gold + rubric, a separate verifier (deterministic checks + LLM judge), and a metadata/provenance report.
**v1:** single-turn, no tools, no web. The solver replies with one JSON object; the harness saves it to `/app/output/answer.json`.

**What the solver sees:** a canary comment, the premise set *P* as prose, the question, and the answer schema. Never the paper, the triplet JSON, or any bibliographic identifier.

**Design sources:** LifeSciBench → scoring, normalization, rubric, judge · GeneBench-Pro → task-design principles · AutoSciBench → the *concepts* of verifiability and difficulty, with our own operational standard (§6). RaR / BixBench3 / GeneBench fill gaps, marked inline.

**Notation.** *P* = premises revealed; *T* = the withheld target. **Direction** = forward | backward (§1). **Step** = one licensed inference in the gold derivation; a **DP** (decision point) is a step with a plausible wrong alternative that changes the answer. **S0–S10** = pipeline stages (§3). **V1–V5** = verifiability checks; **D** = the difficulty panel (§6). *s* = normalized task score, `s_core` and `s_judge` its two components (§5); τ = 0.70 = pass threshold; `s̄` = mean *s* over samples; `p_t` = a task's pass rate over its *k* samples; `p̂` = the panel's measured pass rate.

---

## 1. Core design: staged evidence → withheld result

A paper's headline finding is novel, so "predict the outcome from the setup" is usually not derivable. Following GeneBench-Pro's *recoverable target*, each task **reveals** a premise set *P* of verified facts from the same paper (setup, sibling experiments, upstream results, control designs) and **withholds** a target *T* (a result, control outcome, ranking, or mechanism the paper reports), such that *T* follows from *P* plus standard biology through 1–4 licensed steps, at least one of them a DP.

*P* is assembled only from the **verified fact sheet** (S1) — never from raw Results, Abstract or Discussion text — because the abstract alone states the end state for 482 of 2,174 non-self-loop triplets (22%) and names the perturbation for 63% of all triplets. **"Reported" means stated in Results or figure-legend text**; a value readable only off a figure image is never a gold source. Methods excerpts enter *P* under the same provenance rule.

**Question direction is fixed in the seed.** S0 writes a seed spec, and the Architect (S2) must honor its family, direction and field split; it may not change them:

| Direction | Solver is given | Solver predicts | Answer form |
|---|---|---|---|
| **Forward** | start state + perturbation, plus *P* | the outcome: end state, marker directions, a control arm's result, an arm ranking | enums or an ordered list (auto-graded) + rationale (judged) |
| **Backward** | start state + observed outcome, plus *P* | the perturbation that produced it | a closed choice among the paper's own perturbations (auto-graded) + rationale |

```json
{"seed": "2026.03.12.645123-f4-bwd-01", "doi": "10.1101/2026.03.12.645123", "family": "F4", "direction": "backward",
 "target": ["t07"], "distractors": ["t03", "t11"],
 "reveal":   ["starting_cell_state", "ending_cell_state", "delta", "cell_assay", "context"],
 "withhold": ["description_of_perturbation", "perturbation_modality", "mechanism"]}
```

A seed yields 2–3 task candidates; the candidate's `task_id` (§10) carries its own counter.

Backward needs three rules forward does not. (1) **Closed choice, always.** An open "name the gene" answer is almost never unique given *P*, so V4 rejects it. Distractors are the same paper's other perturbations on the same start state, each with a *different* reported outcome, so exactly one option fits. (2) **The revealed outcome is exempt from the State A/B relabelling (§7)**, so the leak scan instead checks the revealed outcome and *P* for the withheld perturbation's tokens: end-state labels such as "TOX-deficient Treg" name the cause. (3) **A separate leak gate on perturbation tokens.** Over the 1,191 backward-eligible seeds, title+abstract recovers **52% of perturbation tokens against 35% of delta tokens**, and 146 of 317 papers exceed a 0.5 mean on perturbations versus 77 of 324 on deltas — with only 41 papers in both sets, so the forward gate would pass 105 backward-leaky papers. F5 and F7 are direction-neutral: their target is neither the outcome nor the perturbation.

Token means are a cheap pre-filter and a reported statistic only. The gate that decides is the per-seed entailment check in §8.

## 2. Task-design principles (GeneBench-Pro)

Principles enforced by a gate are defined once, in §6: **recoverable target** = V1, **unique answer** = V4, **separation from wrong answers** = V5, **anti-memorization** = the prior-only probe. The rest, each logged pass/fail with evidence in task metadata:

| # | Principle | Operationalization | Enforced by |
|---|---|---|---|
| 1 | Minimal viable prompt | *P* defines the estimand, vocabulary and answer contract, presents evidence neutrally (never "note that…"), and never states *T*. | S3 lint, S5 |
| 2 | Threshold robustness | Directions, ordinal bins, tolerance ranges; gold is kept away from bin edges. | S3 lint |
| 3 | Multi-stage inference (**relaxed**) | A one-step derivation is allowed. Every step must be licensed (V3), and **≥1 step must be a DP** (Notation; bookkeeping does not count) — an S3 lint. DP count *beyond one* is recorded and used to stratify results, never gated. | S3, V3 |
| 4 | Representation / nuisance discovery | **Encouraged, not required**; tagged when a DP turns on choosing the comparator, population or readout frame. | S2 |
| 5 | Literature-defensible | The gold derivation uses standard biology. Every step's premise is quote-matched or passes the K-premise check (V3). | S5, V3 |
| 6 | Controlled evidence | *P* comes only from the verified fact sheet, so cues are tunable. | S1, S3 |

**Meta-principle** (GeneBench-Pro): *insensitive to nearby defensible choices, sensitive to missing scientifically necessary stages*. V5's must-pass set and its must-fail answer at each DP are the test.

S5 and S9 tag every issue with a GeneBench-Pro finding type — underspecified estimand, underspecified prompt, alternate defensible answer failing, gold-derivation error, target not recoverable, conflicting evidence in source, contrived setup — and fix in severity order: clarify *P* → re-key or accept the alternative → redesign the recipe → withdraw.

## 3. Agent architecture

```
extraction.json + .meca
 S0  Seed selector (code+LLM)  DOI dedupe · seed tiers · token pre-filter · per-seed entailment gate (§8) ·
                               sibling-echo screen → seed specs (family, direction, reveal/withhold, §1)
 S1  Fact Verifier (agent)     once per paper → papers/<paper_id>/facts.json: each fact with a char-offset span;
                               drops unverifiable rows; confirms F6 chains are sequential, not parallel arms
 S2  Task Architect (LLM)      recipe for one seed spec: P (fact_ids), T, derivation steps each with its licensing
                               premise (≥1 marked DP), answer contract, gold + derivation, judge criteria.
                               Auto criteria are generated by code from the contract. Also runs hardening (§6).
 S3  Context Builder (code)    instruction.md from P · leak scan · rubric lints (§4)
 S5  Critic (LLM)              LifeSciBench's four review categories + §2 checklist; sees S6's best alternative *Alt*
 S6  Adversary (LLM)           must-pass set · must-fail set (incl. the wrong fork at each DP) · best alternative *Alt*
 S7  Verifiability V1–V5       fail → S2 → drop
 S8  Difficulty panel D        easy → ≤1 structural hardening round
 S9  Trace Auditor (LLM)       on p̂ ≤ 0.25 tasks + 10% sample: grader–prompt mismatch, shortcuts, failures
                               concentrated on K-premises or `unlicensed` fields → fix (re-enter S7) or drop
 S10 Packager (code)           Harbor task dirs · dataset.toml · per-task report
```

**Control flow (per candidate).** Every revision is a new folder `v1`, `v2`, … never edited after it is written; each gate reads only its own version.
- **One budget: ≤4 Architect revisions per candidate**, whatever the cause — an S5 review cycle, a V-gate failure and a hardening round each consume one — plus ≤$B per candidate. On exhaustion, drop.
- Order, cheapest first, stopping at the first failure: **S3 lints → V1 → V3 → S6 → S5 → V4, V5 → V2 → D**. V1 (a deterministic quote match) and V3 (licensing) run before the two LLM authoring stages, so a seed dies before anything is spent writing it up.
- V2 and D each need ≥5 valid trials per panel model. Otherwise retry once, then hold — never pass by default. Refusals are graded 0 and reported as a refusal rate; only transient errors make a trial invalid.
- D says easy → ≤1 structural re-target. If that version then fails V or is still easy, keep the **last version that passed V** with its own D stats; it goes to the easy quota or the RL pool, else it is dropped.
- S9 may edit only rubric or judge text, then regrade with no re-solve. Any `instruction.md` change is a new version and re-runs V2 and D.
- Ship only when instruction, rubric, V report and D stats share one version id.

**Models are configuration, not design** (`configs/models/`, §10). The plan fixes only the rules between roles; any role is overridable per run from the command line. Defaults:

| Role | Used in | Default |
|---|---|---|
| `generator` | S1, S2, hardening | `claude-sonnet-5-5` |
| `critic`, `adversary` | S5, S6 | `claude-sonnet-5-5` |
| `panel` | V2, D, the probes | `gpt-6.1-sol` + `claude-sonnet-5-5` |
| `checker` | S0 entailment gate, V3 licensing and fork votes, V4 preference, S5 anchor votes | same as `panel` |
| `judge` / `judge_regrade` | grading / second-family regrade and judge refusals | `claude-sonnet-5-5` / `gpt-6.1-sol` |
| `heldout` | headline difficulty tiers (§6) | unset until M3 |

A run checks these rules at startup and warns rather than fails:
- **Three votes over two models** = each model once, then the **non-generator** model again. V4 records the per-model split; an item where the generator-family model is the only one preferring gold is re-keyed, not passed.
- With the defaults the generator's model also sits in `panel` and `checker`. V2 tolerates this, being open-book. For D, panel `p̂` is always reported per model and the Sonnet-minus-GPT gap on the same tasks is published as the **self-solve bias**; a third model family removes it and is the first thing to add if it is large.
- `heldout` must lie outside `panel` and differ from `generator`. Until one is named (M3), difficulty tiers are labelled panel-measured.
- M0 measures each role's refusal rate on 20 pilot prompts. Opus 5.5 refused ~40% of these bio prompts in earlier probing; Sonnet 5.5 is unmeasured. Any role above 10% is swapped in config.

**One grading module.** `grade.py` exposes `grade(answer_text, tests_dir, judge) → {rewards, criteria}` plus a CLI. It imports only the standard library and the judge SDKs and is copied verbatim into every task's `tests/`, so `test.sh` only calls it and a task folder is self-contained. The gates and the pilot call the same function through the in-process runner (`evaluate.py`), which needs no container. **The in-process runner produces every number in this plan except the released headline, which comes from Harbor (§9).** Parity is checked by regrading the *same stored responses* in both runners: `core` must agree exactly and `|Δreward| ≤ 0.02` (the judge is sampled, so bit-identical rewards are not achievable). V4, V5 and the constant-answer probe use fixed answers, with no solver. V2 and the prior-only probe are emitted as probe tasks `<task_id>--openbook` / `--prior` — same `tests/`, different `instruction.md` — never published, since they reveal *T*'s evidence.

**Built from scratch.** No papers2cells code is imported, vendored or copied; its outputs are read as data. This repo owns its own `.meca`/JATS reader, extraction-JSON models (`table_1` CellState, `table_2` PerturbationTriple, `table_3` Acronym), quote locator (char offsets, for V1 and the fact sheet), LLM client and Harbor packager. The frozen split manifest is read as a JSON file (§8).

**Cost.** Per candidate per pass: generation ≈$0.8; ~34 solves ≈$2.2 (V2 10 + D 12 + prior-only 12, before early exits); graded responses ≈$2.1 batched to $5.6 per-criterion → **≈$14–38 per accepted task** (cf. QA v3 $0.47/item, AutoSciBench $22–58/task). Controls: cheapest-first gates with early exit; 1 batched judge sample while gating, 3 per-criterion samples only on accepted tasks and in eval; sequential D (3 solves per model first; stop only on unanimity, 6/6 or 0/6, else complete to 6 each); Batches API for S2–S6; a per-candidate spend cap. M2 reports $/accepted task by family and direction before M3 is approved.

## 4. Rubric construction (LifeSciBench)

**Criterion schema:** `{id, section, text, points, kind: auto|judge, critical, essential, check, capability, step, grounding_span}`. Sections mirror the derivation steps. `capability` ∈ direction · comparison · control-logic · mechanism · design · uncertainty. `grounding_span` ∈ {paper span, instruction quote, K-premise id}. 6–20 criteria per task.

**Writing rules (LifeSciBench App. C):** specificity · atomicity · evaluability from the response alone · grounding · non-redundancy · operational usefulness · **question–rubric consistency** (reward only what the question asks). Each criterion is self-contained: it names the entity, direction and context, never "the main finding".

**Points.** Positive integers 1–10; **no negative points** (RaR measured no gain from synthetic pitfalls, 37.2 with vs 37.2 without; and in rewardkit only `weighted-sum` admits negative weights, which can push *s* outside [0,1]).

**Points budget, lint-enforced.** Core-answer criteria take share `w_core` with `0.30 < w_core < τ`, so neither a wrong core answer nor a core-only answer can pass. Caveat/uncertainty criteria ≤ 5%. Per family: `w_core = 0.60` for F1–F4 and F6; `0.40` for F5 and F7, whose cores are judged. No per-criterion cap is needed on top: the pass conjunction (§5) requires `s_judge ≥ 0.5`, so no single criterion can carry a core-only answer.

**Critical fields.** Each task declares ≥1 `critical` core field — the discordant marker, the non-null arm, the top-ranked arm. A wrong critical field sets `s_core = 0`. This, not the budget, is what makes a wrong core answer unpassable: on an 8-down/1-up marker panel an "all down" policy otherwise reaches τ exactly.

**Notice/act.** For each DP, a "recognizes cue X" item is diagnostic-only and carries **0 points**; only the paired "uses X to conclude Y" item scores, and its credited quote must contain both the cue and the inference. This blocks the echo exploit while keeping GeneBench's main failure mode visible.

**S3 lints:** budget bounds hold · ≥1 critical field · ≥1 DP in the derivation · every item has a grounding span and capability tag · no duplicates · no item references the paper · no item restates a premise given in *P*.

**S5 anchor rule** (LifeSciBench's ≥90% expert-consensus standard): every criterion and step premise is either quote-matched (V1/V3) or affirmed 3/3 by `checker` votes; otherwise the criterion is dropped.

## 5. Scoring, normalization, LLM verification

**Task score.** `s = w_core · s_core + (1 − w_core) · s_judge`, where `s_judge` = Σ judge points awarded / Σ judge points and `s ∈ [0,1]`. `s_core ∈ [0,1]` is a family-specific function computed in code, because several families are not sums of independent items:

| Family | `s_core` | Chance floor |
|---|---|---|
| F1 | macro-recall over the direction classes present in gold (balanced accuracy) | ~0.33 |
| F2, F6 | fraction of graded arms matching the gold enum | per enum size |
| F3 | fraction of concordant gold-comparable pairs; pairs inside a declared tie set excluded | **0.50** |
| F4 | exact choice | 1/(choice-set size) |
| F5, F7 | fraction of required key entities / `essential` criteria satisfied (judged) | ~0 |

**Pass** = `(s ≥ τ) ∧ (s_core ≥ 0.9) ∧ (s_judge ≥ 0.5 where the family has a judge section)`, with **τ = 0.70 fixed** (LifeSciBench). Each family's chance floor is *measured* by scoring the best constant policy and a uniform random answer through the real verifier; `s_core` is reported raw and chance-corrected.

**Across k samples per (model, task):** `k = 20` for evaluation; `k = 6 → 12` per panel model, escalating, for D; `k = 6` per panel model for the prior-only probe; `k ≥ 8` for the RL band. **k, not suite size, is the lever for ranking power** — on a simulated 250-task/85-paper suite, k=10→20 lifts the power to separate two models 5pp apart from ~60% to roughly 80% (doubling k cannot do better: the SE ratio is capped at √2), while the CI on one model's score barely moves (±0.056 → ±0.055), being dominated by task variance. Buying that power with tasks would cost ~250 more at $14–38 each, against ~$400–800 per model for the extra samples with their grading. Re-simulate at M3 and publish the curve.
- **Normalized score** = mean over tasks of `s̄_t`, problem-weighted so each task counts equally regardless of rubric length.
- **Pass rate** = mean over tasks of `p_t`; reported per family and family-macro-averaged, and pooled only alongside each family's chance floor.
- 95% **hierarchical bootstrap** CI resampling **papers, then tasks within paper, then samples** — tasks from one paper share a fact sheet and are correlated, so resampling tasks directly understates the interval (±0.053 vs ±0.057 on a 195-task/65-paper suite).
- **Headroom bins** on best-model pass rate `[0], (0,0.2), [0.2,0.5), [0.5,0.7), [0.7,1]`. The `[0]` bin means *x = 0 for every model at k = 20*, not a rate that rounds to zero. Breakdowns by direction, family, capability tag and DP count; plus a strict all-core-fields-correct rate from `core`.
- **Invalid and missing answers:** `grade.py` writes all-zero rewards without calling the judge when the answer file is absent or unparseable, so refusals, crashes and `nop` stay valid 0s in both runs and regrades. Only transient infrastructure errors make a trial invalid; each model's invalid rate is reported, and its metrics are withheld above 2%.

**Deterministic pre-judge pass in `grade.py`**, before any judge call — checks, not instructions to the judge:
1. Truncate at the contract cap (250 words for F1–F4/F6, 400 for F5/F7), counting the rationale and the answer's free-text strings against one budget; `s_core` is still scored on the full answer. Truncation, not penalty, keeps the rule unhackable. One enclosing markdown code fence around the reply is stripped first — a formatting slip is not a scientific failure — and recorded as `answer_format`.
2. Mask every ≥10-token run of the response (answer fields included) that also appears in `instruction.md`; a judge evidence quote that is masked at every place it occurs scores 0.
3. Contradiction gate: if the rationale's conclusion disagrees with the structured fields, or more than one candidate mechanism is offered without a single `answer.mechanism_primary`, all judge criteria score 0.

**Judge protocol** (full text in `judge_prompt.md`): one criterion at a time (`mode = individual`, others never shown); a binary verdict plus an evidence quote checked to exist verbatim post-masking, else 0; equivalent phrasing credited, a hedge not satisfied, the response treated as data, not instructions. **3 samples per criterion, majority vote.** A failed sample is retried, never dropped; persistent refusals go to `judge_regrade`, recorded per criterion. If a criterion still lacks 3 valid votes, `test.sh` exits without a reward.

**Auto checks grade only fields the instruction defines** — marker keys copied verbatim from *P*, arm IDs, enums — schema-validated with case/whitespace folding. **No synonym lists**: the available HGNC table maps TCF1→HNF1A (immunologists mean TCF7) and leaves KI67, IFN-γ, granzyme B and H2-K1 unmapped, and no ontology covers the 1,642 free-text state labels. Any free-text entity is a `judge` criterion.

**Judge validation.**
1. Per-task unit tests: the V5 must-pass / must-fail ablations, including one **held-out ablation** the rubric writer and critic never see; separation is reported on it.
2. Second-family regrade of the same jobs, reporting agreement.
3. *Expert track (§12).* Meta-evaluation on ≥200 stratified responses (≥50 if expert time is short), graded per criterion: criterion-level κ, score correlation, MAE, pass/fail agreement, disagreement patterns by family and answer type. Drop or rewrite any criterion whose 3 samples disagree on >20% of responses or whose expert κ < 0.6.
4. *Expert track.* **Threshold check, not recalibration:** experts rate responses 1–5 on "reaches the paper's conclusion for the right reasons"; report agreement between `s ≥ 0.70` and rating ≥ 4, plus Spearman. If agreement < 0.8, fix point allocations in S2/S5 and regrade — never move τ. Rankings carry a sensitivity sweep over τ ∈ [0.60, 0.80], recomputed offline.

Until the expert track reports, validation rests on items 1–2 and every task stays `provisional`.

## 6. Verifiability and difficulty (AutoSciBench concepts, our standard)

**Verifiable ⇔ V1–V5 all pass**, checked before difficulty. A task S9 repairs re-enters at S7, not S8.

| Check | Test |
|---|---|
| **V1 Provenance** | Every gold field and core criterion maps to a verbatim span in `paper.txt` (deterministic quote match, Results or figure-legend text only). |
| **V2 Well-posedness** | Premise spans in full **plus the target's own evidence span with the result clause redacted** → each `panel` model × 5 samples. Require `s_core = 1.0` on ≥8 of 10 valid trials and ≥3/5 per model. Separately, the gold rationale from `solve.sh` must score `s_judge ≥ 0.9` on 3 judge repeats. Scoring `s_core` rather than `s` keeps judge criteria from being made echo-satisfiable — except for F5/F7, whose core is itself judged, where V2 reduces to "every required entity recovered open-book". *V2 does not test ambiguity; V4 does.* |
| **V3 Derivability** | **Solver-visible only.** (a) *Every step* is licensed: its premise is either a quote-matched sentence of `instruction.md` or a **K-premise** — the leak scan finds no entity or direction overlap with any withheld fact, and 3/3 `checker` votes rate it closed-book as "true in general biology, independent of this study". (b) *Each DP* additionally gets a fork test: a checker sees `instruction.md` plus the resolutions of earlier steps — never this step or later ones, never the gold — and chooses, shuffled, between the recipe resolution and its plausible-wrong alternative. It must pick the recipe resolution in ≥5/6 samples (2 checker models × 3). Non-DP steps get only (a). The last step's resolution must equal the gold. Fields no premise licenses move to an `unlicensed` block that is reported but never scored. |
| **V4 Uniqueness** | The Adversary (2 samples) writes its strongest alternative *Alt* with a derivation from *P*. Three `checker` votes (≥2 families) see *P*, the question, and the gold derivation and *Alt* in random X/Y order, **not knowing which is gold**: "which does *P* better support — X, Y, or neither?" Gold 3/3 → pass. *Alt* preferred ≥2 → the gold is suspect: re-key from S1 or drop. Otherwise → add the disambiguating constraint to *P*, or accept *Alt* too, then re-run V2/V5. |
| **V5 Grader separation** | **Must-pass (≥ τ+0.05 on 3 judge repeats):** paraphrase, synonyms, an alternative defensible derivation. **Must-fail (`s ≤ τ−0.25` and `s_core ≤ max(0.2, floor − 0.2)` against that family's *measured* chance floor, so the bar is never set below chance):** textbook prior, flipped direction, other cell type, the constant-enum answer plus the stem's own mediator noun, and for each DP the answer reached by taking its plausible-wrong fork. **Must-not-pass (`s_judge ≤ 0.3`, so the pass conjunction fails):** correct core with wrong mechanism, correct core with no rationale, hedge-everything. Gold−ablation gaps are reported per class. |

**Difficulty D.** Panel: the `panel` models (default 2, from 2 families); `k = 6` per model (n = 12). Report `p̂` = the unweighted mean of the per-model pass rates, with per-model rates and the min–max range — **not a binomial CI**, since the trials are clustered by model. Classes (easy ≥0.75 / medium 0.25–0.75 / hard <0.25 / unsolved 0) are descriptive labels for stratification, never a reported outcome. Two models give one degree of freedom between clusters, so **a cap, the S9 audit trigger or the unsolved split acts only when both models fall on the same side of the boundary**; a split verdict is re-run at `k = 12` per model, and if it is still split the task is stratified as `medium` and the disagreement recorded.

**Probes.** *prior-only*: the same question with *P* removed and a one-line setup built from species, start cell type and perturbation, at `k = 6` per panel model (12 trials). Flag `textbook` iff the pooled Wilson lower bound ≥ 0.25, i.e. ≥ 6 of 12. *constant answer* (per family): the best constant policy must score `s̄ ≤ 0.35` and pass rate 0 through the real verifier.

**Selection.** The eval split caps easy at 15%, `textbook` at 10%, and `unsolved` at 20%; an unsolved task enters only after S9 confirms failures fall on licensed steps and after a `checker` re-derives the gold blind. The RL pool keeps tasks whose reward varies **under the policy being trained**: `0.1 ≤ s̄ ≤ 0.9` and `sd(s) ≥ 0.08` over `k ≥ 8` rollouts of the starting policy, re-estimated at each refresh — not from the frontier panel, which is far stronger than the trainee. For the RL reward each judge criterion contributes the **mean of its 3 samples** (a finer grid); the eval pass decision uses the majority verdict.

**Hardening** (≤1 round, structural, run by S2). The Architect is told only `tier = easy` and never sees panel answers or rationales — solver feedback pushes a generator to make the science weird rather than the reasoning deeper. It re-targets from the fact sheet: a contrastive arm, the discordant readout, a two-premise integration. Panel `p̂` is used **only for selection**; headline tiers are reported on `heldout` once named. A held-out model removes solver-identity bias but not composition bias, so a randomly composed 100-task reference slice (no caps, no hardening) runs alongside the main suite and the frozen composition is published with every score.

## 7. Task families (v1)

Supply columns are **seeds, not tasks**: `immunology_top300` only (4,083 triplets), at the §8 seed tier — A′ for F1–F4 and F6, A for F5 and F7 — and recomputed on the deduped 455-paper corpus at M2, before and after the entailment gate.

| Family | Direction | *P* → *T* | Answer contract | `s_core` | Supply (seeds / papers) |
|---|---|---|---|---|---|
| F1 Marker direction panel | fwd | setup + sibling results → per-marker change | `{marker: up/down/unchanged}`; ≥1 discordant marker **whose direction is licensed by a premise other than the majority trend** | balanced accuracy | 513 / 210 |
| F2 Paired control | fwd · bwd | main effect → outcome of a control/blockade/rescue arm | enum per arm, **paired with a non-null arm** | arm fraction | 609 / 69 nulls; 1,548 / 275 control-language |
| F3 Contrastive ranking | fwd | arms on one start state + the readout → ranking | ordered list with tie sets | pair concordance | 482 groups / 238 |
| F4 Inverse inference | bwd | observed delta → which perturbation | choice among same-paper perturbations with distinct end states | exact | same pool as F3 |
| F5 Mechanism | — | results incl. controls → the paper's explicit mechanism | free text + key-entity fields | judged key entities | 1,726 / 825–907 |
| F6 Two-hop chain | fwd · bwd | hop-1 result → composed or intermediate outcome | enum + state description | arm fraction | 387 / 208 pairs, 39 papers |
| F7 Discriminating experiment | — | competing hypotheses + results so far → the paper's decisive control | free text | judged `essential` criteria | from the F2 pool |

**Backward variants.** F2 bwd: given that one arm lost the effect, which of the paper's interventions was applied. F6 bwd: given the final state, which hop-1 perturbation started the chain. F4 is backward by definition. F3 is forward only — its contract is an ordering, and the backward version is an assignment that `s_core` does not score. Every backward contract is a closed choice over the paper's own perturbations (§1).

The answer is always `{"answer": {...}, "rationale": "..."}`. **End-state labels are replaced by neutral identifiers (State A/B)**, except a backward task's revealed outcome (§1), and the contract enumerates every candidate label in the paper — because 1,102 of 2,174 non-self-loop end-state labels (51%) carry their own valence ("Activated CD8+ T cell", "M1 macrophage", "Pro-fibrotic lung fibroblast"), which a token scan cannot flag since the label belongs to the answer vocabulary. The instruction states honestly that the rationale is graded.

## 8. Data, splits, integrity

- **Corpus.** `immunology_top300` and `cancer_top300` **share 109 DOIs** (cancer immunology); deduped the two post_jan corpora are **455 unique papers**, not 564. Always dedupe before splitting or counting.
- **Splits are deferred; generation uses all the data.** All 455 papers are generation inputs. Each task records its paper's assignment in the frozen 80/10/10 paper-family manifest (`train` / `dev` / `test` / `unassigned`) as metadata, so the eval-vs-RL partition is a later *selection* step needing no regeneration (§12). The 5 pilot papers come from `train` and are excluded from eval under any future split, because prompts and principles get tuned on them.
- **Seed tiers** (triplet quality, pre-generation; orthogonal to §6's difficulty classes). All require `evidence_source = primary` and a rich delta. The mechanism requirement is **family-dependent** — only F5 and F7 grade a mechanism, so requiring `explicitly_described` everywhere discards 26% of usable seeds for no benefit:

  | Tier | Definition | Corpus-wide | dev+test, if the frozen split is kept |
  |---|---|---|---|
  | **A** — F5/F7 seeds | A′ + `mechanism_source = explicitly_described` | 1,109 train | **248** / 63 papers |
  | **A′** — F1–F4, F6 seeds | primary + rich delta + `validated`, or `single_experiment` under the carve-out below | — | **337** / 76 papers |
  | **B** — RL | + (`validated` ∨ `single_experiment`) | **2,653** train | 570 |
  | **C** — paired null arms | self-loop with explicit abolish / no-change language; exempt from rich-delta | — | 73 / 32 papers |

  **Rich delta, stated exactly** — the prototype got all three wrong. (a) The null-phenotype exclusion is a *family* of phrasings, not the literal string "no distinct phenotype"; matching only that phrase left 448 of 2,243 train A′ rows carrying excluded language. (b) A′ and C are **mutually exclusive**: a row matching null language is C, never A′, or a null arm can claim to be a validated gold. (c) "A quantitative value" means a number with a unit, a percentage or a fold change — not any digit, which admits gene names (CD22, IL-6). Otherwise: ≥2 named markers.

  **The `single_experiment` carve-out.** `validated` and the leak gate pull against each other: the arms abstracts omit — the catalytic-inhibitor control, the binding-deficient mutant, the failed arm of a screen, the cross-pathway null — are the least leaky seeds *and* the ones most often marked `single_experiment` (4 of 10 in the trial selection). Such a row is an eval gold only when the paper reports the same conclusion in a second system or assay (S1 verifies and records both spans), or when it is the null half of a pair. Every such task is tagged and results reported with and without them.

  Tier C never carries the gold alone: it is the null half of an F2/F3 pair whose other half is a non-null arm from the same paper and start state. Without it those families are gameable.
- **The leak gate is per seed and sentence-level, not a per-paper token mean.** In a trial selection over 5 train papers, 7 of 10 golds sat in the title or abstract while every per-paper token score was 0.11–0.36, far under the 0.5 gate. A token scan cannot see it: it drops lowercase agents ("lipopolysaccharide", "trametinib"), is synonym-blind (LPS ≠ lipopolysaccharide), and misses an abstract that states a *pattern* ("Akt drives proliferation early, ERK polarity late"), giving away a whole marker panel with no shared token. S0 therefore runs a `checker` **entailment check of each candidate gold against title+abstract** and drops the seed on a hit.
- **Leakage screens beyond the token scan:** a **sibling-echo screen** at S0 rejects a (*P*, *T*) group when ≥1/3 of *T*'s distinctive perturbation tokens already appear in a revealed sibling's `delta` or `mechanism` (48% of triplets cross-mention corpus-wide); and the `mechanism` field of any sibling is **banned from *P*** whenever *T* is a control or blockade arm, because 119 triplets put a control outcome in that field verbatim.
- **Exclusions:** speculated, preliminary, reference; intent-leaking perturbations (133 triplets, 6% of non-self-loop); papers with no parseable methods, for setup details.
- **Extraction quality:** the strict per-paper extraction pass rate is 16–40%, so S1 verification is mandatory. Triplets are candidate facts; the evidence spans are the ground truth.
- **Expected yield, and what happens if it misses.** Under the frozen split if kept, eval tasks come from the 85 dev+test papers, of which **76 carry the 337 A′ seeds** (248 Tier A), split ~40/60 dev/test, so the private half is larger. At 2–3 candidates per seed, ≤5 tasks per paper and about a third surviving S1+V+D, that is **≈220–330 eval tasks**, target **250 with ≥150 private** — but **computed before the entailment gate, which the trial suggests may remove most seeds**. M2 measures post-gate survival on ~30 papers and decides: below 50%, the options in order are (i) draw from the whole corpus rather than dev+test only, now possible because the split is deferred, (ii) lower the target and publish the suite size with its CI, (iii) add families. Relaxing the gate is not an option — a leaked task measures nothing.
- RL = 2,653 Tier-B triplets in train papers, plus the pre2026 pools (~41k rows), wrong for eval and fine for RL.
- **Contamination slice.** Only 20 eval papers (≈80 Tier-A seeds) are from 2026-05/06; the rest are 2026-01..04, inside plausible training windows, and 6 are pre-2026. Report the headline on the full suite **and** on the 2026-05/06 slice. A large gap, or a high prior-only pass rate (§6), is the contamination signal.
- **Integrity:** a canary GUID in `instruction.md`; `tests/` never visible to the agent. A task is `reviewed` once a human has signed its `review/` page (every unsolved task plus 10% per family) and `provisional` until the expert track reports (§5); released tasks must be `reviewed`.

## 9. Harbor packaging (ILC cluster, no docker)

```
<task_id>/            # e.g. 2026.05.31.727600-f2-fwd-01
  instruction.md      # canary · P · question · answer schema · "Reply with exactly one JSON object in this schema;
                      #   it is saved verbatim to /app/output/answer.json and graded, including the rationale."
  task.toml
  solution/solve.sh   # writes the gold answer + gold derivation as rationale
  tests/  test.sh · grade.py · rubric.json · gold.json · question.md
```

```toml
schema_version = "1.4"
artifacts = ["/app/output"]            # top-level; a DIRECTORY, so no-answer trials stay valid 0s and replay on regrade
[task]
name = "p2c/<task_id>"
version = "1.0.0"                      # rubric/judge edit → minor (regrade); instruction edit → major (rerun)
[metadata]                             # family, direction, steps/DPs, provenance spans, V1–V5 verdicts, frozen split,
                                       # models, image sha256 — fixed at packaging; p̂/s̄/class live in the run report
                                       # outside the task folder, so re-measuring difficulty never changes a task
[environment]
docker_image = "<image>"               # see the two profiles below
network_mode = "<policy>"
[agent]
timeout_sec = 1800
[verifier]
environment_mode = "separate"          # regrade requires it; with no [verifier.environment] table the verifier
timeout_sec = 1800                     #   reuses [environment] and uploads tests/ at runtime
[verifier.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"   # both declared so either judge works out of the box;
OPENAI_API_KEY = "${OPENAI_API_KEY}"         #   grade.py picks the key by the judge's provider
JUDGE_MODEL = "${JUDGE_MODEL:-claude-sonnet-5-5}"
```

**Three environment profiles, one packager option** (`--environment local | singularity | cloud` on the packager; M1's S10 reads `package.environment`). `network_mode` lives in task.toml with no CLI override, and a local `.sif` path cannot resolve on a cloud runner, so the profile is fixed at packaging time:

| | **`local`** (default now) | `singularity` (ILC, later) | `cloud` (modal / daytona, later) |
|---|---|---|---|
| Runs on | the host, no container: `p2c_task_re.harbor_env:LocalEnvironment`, a Harbor `BaseEnvironment` | Harbor's singularity environment | the provider's sandbox |
| `docker_image` | none (the task carries an `environment/README.md` placeholder, which Harbor's task discovery requires) | absolute `.sif` path, used as-is | `ghcr.io/…@sha256:…` |
| `network_mode` | omitted = `public` | omitted = `public` — singularity rejects the other two | `no-network`, verifier `allowlist` = the judge hosts |
| Isolation | **none** | container, host network | full |

The same task content is emitted under every profile; the dataset records which profile and which image it was built with.

**`local` — Harbor with no container.** Each trial gets its own host folder standing in for `/app`, `/tests`, `/solution` and `/logs`: file operations, the `exec` command string and `cwd` are mapped onto it, and the scripts also honour `TESTS_DIR`, `ANSWER_PATH`, `VERIFIER_DIR` and `OUTPUT_DIR`. The project venv's python is first on `PATH`, so `tests/grade.py` finds the judge SDKs. It refuses tasks that ship an image or a Dockerfile, and tasks that give the agent tools (MCP servers, skills). Run with `-e p2c_task_re.harbor_env:LocalEnvironment … -y`.

**Runner: Harbor's `singularity` environment** — the only one that runs locally without a daemon (checked against Harbor `ec6aeab`). A `.sif` path in `docker_image` is used directly; a registry ref is pulled and converted with `singularity pull docker://…`, also daemonless. Containers start `--fakeroot --containall` with no `--net`, so they share the host network and `no-network`/`allowlist` raise at trial creation. Separate verifier, directory artifacts, `upload_file` and `harbor job regrade -e singularity` all work. Its bootstrap re-installs python, uvicorn/fastapi, tmux and asciinema on every start unless present, so the image bakes them in.

**Why `local` and `public` are acceptable for v1.** The solver sandbox is not isolated at all under `local` and not network-isolated under `singularity`, and in neither is the verifier's egress limited to the judge host — but in a no-tools task nothing the model writes is executed (`SingleShotAgent` calls the API from the host and uploads the reply as data), so isolation prevents nothing. Two guards keep this from carrying forward: the packager and `LocalEnvironment` refuse a tool-using task under `local` or `singularity`, and a `grade.py` test asserts the judge is its only network call. Never run an installed agent (claude-code, codex, terminus) under `local`.

**Why not `singularity` yet.** ampere2 (ILC: Ubuntu 24.04, Slurm) has neither apptainer nor singularity, and AppArmor blocks the unprivileged user namespaces `--fakeroot` needs (`kernel.apparmor_restrict_unprivileged_userns = 1`), which a user-space install does not get around. **Ask the ILC admins to install the root Apptainer package with its AppArmor profile on the CPU partition.** Only ampere2 was probed; check other nodes first. Until it lands, `local` is the Harbor profile; [docs/ilc_apptainer_request.md](docs/ilc_apptainer_request.md) is the drafted request.

**Image.** `containers/p2c-base.def` (Debian slim, python3, both judge SDKs pinned, Harbor's server dependencies, tmux, asciinema, `/app/output`) builds `images/p2c-base-<date>.sif`, never overwritten, its sha256 recorded in the dataset manifest; a release builds the same recipe as an OCI image for the `cloud` profile.

**Run** inside `sbatch` (`scripts/harbor.sbatch`); containers are local subprocesses on localhost, so Harbor runs on the node itself and tasks shard across jobs.
```bash
uv run harbor run -p <dataset> -e p2c_task_re.harbor_env:LocalEnvironment -a p2c_task_re.harbor_agent:SingleShotAgent -m "$SOLVER_MODEL" -k 20 -n 16 -r 3 -o jobs/ -y
uv run harbor job regrade jobs/<job> -p <dataset> -e p2c_task_re.harbor_env:LocalEnvironment --ve JUDGE_MODEL=gpt-6.1-sol
# later: -e singularity (or modal / daytona) on a dataset packaged with that profile
```

**`reward.json`** (flat, numeric): `{"reward": s, "core": s_core, "judged": s_judge}`; `pass` is **not** stored. `grade.py` also writes `/logs/verifier/grading.json` (per-criterion verdicts, all 3 samples, evidence quotes, judge model), which Harbor syncs back. Pass, pass@k, τ sweeps, point re-weights and capability stats are recomputed offline from it, so a regrade is needed only when the judge model or prompt changes. Multi-key rewards disable Harbor's built-in pass@k. `grade.py` issues its judge calls concurrently, bounded so the worst case (20 criteria × 3 samples) finishes inside `timeout_sec`.

**Solver agent:** a custom external `SingleShotAgent` (`BaseAgent`), so inference runs on the Harbor host. It always writes the completion via `environment.upload_file` — including on a refusal or empty output — and records `refusal: true` in metadata, so refusals are graded 0 rather than excluded. Run with `-r 3`, so only persistent transient errors invalidate a trial. It records tokens and cost.

**Sanity runs:** `-a oracle` must give `core = 1.0` and `reward ≥ 0.9`; any judge criterion the oracle fails in ≥2 of 3 samples is a rubric or gold-rationale defect and returns to S2. `-a nop` must give a valid 0, in both a run and a regrade.

**Release names.** Task ids carry the DOI suffix so they are readable during development. At release, private-test tasks are renamed `p2c/test-0001`, …, the mapping kept in the run folder, so the public listing does not reveal which papers are held out.

## 10. Code layout (uv + Hydra)

A standalone package in this repo, built from scratch (§3). **Python 3.12** (Harbor ≥ 0.26 requires it); **`uv init --package`** for the src layout, lock file and console scripts; **`hydra-core>=1.3.8,<1.4`** with `@hydra.main(version_base="1.3")` and an absolute `config_path` (a relative one breaks once installed as a console script); **one pydantic model validates the composed config** (`extra="forbid"`), since Hydra only struct-checks command-line overrides.

```
p2c-task-re/
├── pyproject.toml            # deps; scripts p2c-pipeline, p2c-grade; dev group pytest, ruff
├── uv.lock                   # committed
├── configs/                  # Hydra: values only
│   ├── pipeline.yaml         # defaults list, run_name, run_stages, budget_usd, hydra.run.dir
│   ├── data/post_jan.yaml    # the two dated extraction folders, .meca root, frozen split manifest
│   ├── models/default.yaml   # the §3 roster
│   ├── families/v1.yaml      # per family: enabled, directions, w_core, rationale cap
│   ├── stages/default.yaml   # gate thresholds, revision cap, $ cap, roles per gate, environment profile
│   └── experiment/           # pilot.yaml (pins the 5 paper ids) · full.yaml · eval.yaml
├── containers/p2c-base.def   # Apptainer recipe (singularity profile, later)
├── docs/                     # task_format.md (the task + grading contract), drafts and probe reports
├── examples/tasks/           # hand-built M0 tasks from train papers
├── scripts/harbor.sbatch     # one Harbor shard
├── src/p2c_task_re/
│   ├── cli.py                # @hydra.main → validate → run
│   ├── config.py             # pydantic schema
│   ├── run.py                # stage order, per-candidate loop, resume by output existence
│   ├── records.py            # on-disk record models
│   ├── llm.py                # provider calls, retries, structured output, spend cap
│   ├── ledger.py             # append-only calls.jsonl
│   ├── families.py           # answer contracts, directions, auto criteria
│   ├── evaluate.py           # in-process solve + grade over task folders
│   ├── report.py             # summary.json: scores, bootstrap CIs, headroom, breakdowns, contamination slice
│   ├── release.py            # split selection, private-test renaming, dataset manifest
│   ├── harbor_agent.py       # SingleShotAgent
│   ├── harbor_env.py         # LocalEnvironment: Harbor BaseEnvironment, no container
│   ├── text.py               # normalize, locate_quote (char offsets)
│   ├── data/                 # extraction.py · splits.py · meca.py · jats.py
│   ├── stages/               # select.py (S0) · facts.py (S1) · architect.py (S2 + hardening) · build.py (S3) ·
│   │                         #   critic.py (S5) · adversary.py (S6) · audit.py (S9) · package.py (S10) ·
│   │                         #   review.py + prompts/*.md, templates/
│   ├── gates/                # provenance · wellposed · derivability · uniqueness · separation · difficulty
│   │                         #   (D, probes, RL-pool selection) + prompts/*.md
│   └── grading/              # grade.py (stdlib + judge SDKs only) · judge_prompt.md
├── tests/                    # fixtures (cut-down extraction, mini .meca, one hand-built task); live API tests opt-in
├── images/                   # gitignored
└── outputs/                  # gitignored; runs/<run_name>/<YYYY-MM-DD_HH-MM-SS>/
```

A run folder holds `pipeline.log`, `.hydra/`, `selection.json`, `seeds.jsonl`, `calls.jsonl`, `summary.json`, then `papers/<paper_id>/{paper.txt, sections.json, facts.json}`, `candidates/<task_id>/v1,v2/{recipe,instruction,rubric,gates}`, `dataset/`, `probes/` (never published) and `review/{index.md, <task_id>.md, verdicts.csv}`. Modules are named for what they do, not by S-number: the numbering has gaps and will shift, `run.py` holds the order, each docstring gives its S-number.

**Names, no hashes.** `paper_id` = the DOI suffix (`10.64898/2026.05.31.727600` → `2026.05.31.727600`), unique across both corpora; for the 28 DOIs with two or more archive versions, dedupe keeps the newest version, then the file with the most triplets, then the corpus order, and records the alternatives. `task_id = <paper_id>-<family>-<fwd|bwd>-<NN>`, the counter running within each (paper, family, direction); F5 and F7 drop the direction (`…-f5-01`). Candidate versions are `v1/`, `v2/`. To resume a run, pass its folder as `hydra.run.dir`.

The model roster is one YAML block mapping each §3 role to `{provider, model, effort, max_tokens}`, with `checker` and `panel` as lists. Two rules it must encode: **no temperature or top-p** (Sonnet 5.5 rejects non-default sampling, so repeated samples are repeated calls), and **`refusal_fallback` applies only to `generator`, `critic` and `adversary`** — never `panel` or `heldout`, whose identity is what is being measured, and for `judge` only via `judge_regrade`, recorded per criterion.

```bash
uv sync
uv run p2c-pipeline +experiment=pilot                     # → outputs/runs/pilot/<timestamp>/
uv run p2c-pipeline +experiment=pilot --cfg job           # print the composed config, run nothing
uv run p2c-pipeline +experiment=pilot 'models.roles.critic=${models.gpt}'
```

**Deliberately not built:** Hydra `_target_`/`instantiate`; a stage base class or registry; ConfigStore; a database; hashed ids, cache keys or checkpoint files (resume checks whether outputs exist); a response cache; an LLM-router library; a job queue; a typer/click layer.

## 11. Milestones

1. **M0, skeleton and harness — done (2026-11-16; offline suite and live acceptance green).** The uv package and Hydra configs · data readers tested on fixtures · `llm.py` and the ledger · `grade.py` with a golden-file test per family (gold, each ablation, the best constant policy, a uniform random answer) · the in-process runner plus 3 hand-built tasks (F2 forward, F4 backward, F5) passing oracle and nop · confirm `gpt-6.1-sol`'s API id · measure each role's refusal rate · file the Apptainer request (§9).
2. **M1, pilot: 5 papers → 10 questions for your review.**
   - **Runs:** the §3 gate order — S0–S3, V1, V3, then S5, S6, V4, V5, then V2 — followed by a **reduced D at `k = 3` per panel model**, below the usual minimum, so every question arrives with gate verdicts and an indicative difficulty that is not used for any cap. No hardening, no S9.
   - **Deliverable:** `review/` — `index.md`, one page per question, `verdicts.csv` (accept / fix / reject + note). Each page shows the instruction exactly as the solver sees it, the gold with its derivation (each step's licensing premise, DPs marked), the rubric with points, the gate verdicts, and two sampled panel answers with their scores. Also: $ per question and refusal rate per role.
   - **Exit bar:** ≥8 of 10 accepted, **zero** entailment-leak hits and **zero** golds contradicting their paper. Below that, fix and re-pilot before M2.
   - **Pilot set.** Five `train` papers, from the 141 with ≥4 Tier-A′ non-self-loop seeds, 8–30 triplets and a parseable Results section plus figure legends, then read by hand. Train, not dev, because they get tuned on; excluded from eval under any later split.

     | | DOI suffix | Subject | Trip. / A′ns | Species · modalities | Why |
     |---|---|---|---|---|---|
     | P1 | `2026.02.17.706358` | Api5–FGF2, breast epithelium | 9 / 8 | human · genetic, small molecule | two branch inhibitors disagree on which phenotypes revert; a cross-pathway null |
     | P2 | `2026.03.11.710967` | Microglial CD22, Huntington's | 25 / 13 | mouse+human · genetic, point mutant, antibody, EV | a HEK reconstitution licenses the ITIM mechanism |
     | P3 | `2026.04.25.720790` | PRMT7 restricts CD8⁺ expansion | 25 / 16 | mouse+human · KO, catalytic inhibitor, PROTAC | three perturbations of one target, one going the opposite way |
     | P4 | `2026.04.08.717325` | Cxcl11, zebrafish kidney regeneration | 21 / 6 | zebrafish · cytokine, injury, small molecule | the only non-mammalian paper; six stimuli on one quiescent state |
     | P5 | `2026.05.28.728506` | γc cytokines license IL-1 inflammation | 30 / 7 | mouse+human · cytokine, inhibitor, genetic | lowest leak (0.11); six cytokines on one start state, two of which fail |

     They cover both directions, six families, four modality classes and three species. **F6 is deferred to M2**: the only verifiable two-hop chain in these five is stated verbatim in its own abstract. The question mix is chosen by the pipeline, not fixed here — a hand-built trial mix over these papers was checked against the sources and **all 10 questions failed**: 7 to abstract leak, 2 golds contradicted their papers, 1 premise set anti-licensed its own gold. That is the evidence behind the §8 entailment gate and tier definitions, and what M1 must show has been fixed.
3. **M2, gates at scale:** every gate including hardening and S9, on ~30 train papers. Post-gate seed survival (the §8 yield decision point), $/accepted task by family and direction, judge validation items 1–2.
4. **M3, full run** over all 455 papers → the candidate pool. Decide the eval split and name `heldout` (§12). Re-simulate the k-vs-N power curve. Harbor parity on 5% of tasks.
5. **M4, release:** contamination slice, private-test renaming, Harbor release run on the `singularity` or `cloud` profile. The expert track runs separately and is what lifts `provisional`.

## 12. Decisions

| # | Decision | Where |
|---|---|---|
| 1 | **Code location — settled.** A standalone package in this repo, built from scratch; papers2cells is a read-only data source. | §3, §10 |
| 2 | **Models — settled as configuration.** Sonnet 5.5 for every role except `judge_regrade`; panel and checker = GPT-6.1 sol + Sonnet 5.5. Open: confirm the GPT API id at M0; name `heldout` at M3. | §3 |
| 3 | **Container host — settled: ILC, no docker.** Harbor runs under the container-free `local` profile now; `singularity` (after an admin Apptainer install) and `cloud` remain selectable profiles. | §9 |
| 4 | **Expert availability — a separate track.** The main line validates with cross-family LLM votes; tasks stay `provisional` until it reports. | §5 |
| 5 | **Eval-split size — deferred; generate on all the data.** Every task carries its paper's frozen-manifest split, so the partition is a selection step at M3. | §8 |

On (5): if nothing changes, the default stays the frozen 80/10/10 split and ~250 eval tasks, with ranking power bought through `k = 20` rather than more tasks. Re-freezing is the one option that is foreclosed — it would move papers out of train and invalidate the MCQ exports already built on that split.

---

### Appendix: one worked task, and one rejected one

**F2, anti-MDA5 fibroblast paper** — *illustrative only; this triplet is `single_experiment` with a no-phenotype delta, so it fails the seed tier.*

- ***P*:** IMR-90 lung fibroblasts treated (1 µg/ml) with recombinant MDA5, with patient anti-MDA5 IgG1, or with the MDA5 + anti-MDA5 immune complex. Readout: real-time impedance. The complex gives the highest normalized cell index (~+25% at 121 h; doubling time 54.4 h → 46.8 h). No cytotoxicity in any arm.
- ***T*** (two arms with different gold, so a constant answer scores `s_core = 0.5` and fails the `s_core ≥ 0.9` gate): (a) anti-MDA5 IgG1 alone in complement-replete serum → **moderate rise from ~60–70 h**; (b) anti-MDA5 ± MDA5 in **serum pre-treated to remove a heat-labile activity** → **lost** (arm (b) is `critical`). The stem says "heat-labile serum activity", not "complement" — the mediator noun is a judge criterion, never a stem cue.
- **Steps:** IgG1 immune complexes fix C1q in serum-containing medium → removing that activity removes the complex-dependent response (DP) → antigen-only activation does not require it, so arm (a) persists (DP).
- **Rubric** (total 20, `w_core = 0.60`): auto [7] arm (b) = lost (critical); [5] arm (a) = increased · judge [3] attributes the loss to complement activation by the IgG immune complex; [3] names the classical pathway / Fc-dependent C1q engagement; [2] uses the antigen-alone premise to explain why arm (a) survives. No judge criterion exceeds 25% of the 8 judge points, and a correct core with a wrong mechanism scores `s_judge = 0.25 < 0.5`, so it fails the pass conjunction.
- The enum is **lost** (abolished or reduced), not "abolished": nothing in *P* licenses complete versus partial loss, so grading that distinction would violate threshold robustness — and a step asserting it is exactly what V3 removes.

**A task V3 rejects.** In the TOX/Treg paper, the premise "TOX rises alongside PD-1, TIM-3, CTLA-4, CD44, Ki67 **and CD103** during tumor progression" does not license the withheld "CD103 **increases** after TOX deletion while the others fall" — it *anti*-licenses it. V3 moves CD103 to the `unlicensed` block, which guts F1's discordance requirement. The derivable version uses different premises from the same paper: TOX-KO Tregs upregulate stem-like genes and gain *Tcf7*/*Ccr7* chromatin accessibility while losing *Ccr8* accessibility, licensing the withheld flow result **CCR4↑ / CCR8↓**. That is the F1 pattern to build.
