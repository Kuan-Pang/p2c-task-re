# papers2cells → Harbor task-synthesis pipeline (v1: no-tools reasoning)

**In:** perturbation triplets (`*.extraction.json`) + source papers (`.meca` → `paper.txt`).
**Out:** a Harbor dataset — per task an `instruction.md`, a hidden gold + rubric, a separate verifier (deterministic checks + LLM judge), and a metadata/provenance report.
**v1:** single-turn, no tools, no web. The solver replies with one JSON object; the harness saves it to `/app/output/answer.json`.

**What the solver sees:** a canary comment, the premise set *P* as prose, the question, and the answer schema. Never the paper, the triplet JSON, or any bibliographic identifier.

**Design sources:** LifeSciBench → scoring, normalization, rubric, judge · GeneBench-Pro → task-design principles · AutoSciBench → the *concepts* of verifiability and difficulty, with our own operational standard (§6). RaR / BixBench3 / GeneBench fill gaps, marked inline.

**Notation.** *P* = premises revealed; *T* = the withheld target. **DP** = decision point (an inferential fork where a plausible wrong choice changes the answer). **S0–S10** = pipeline stages (§3). **V1–V5** = verifiability checks; **D** = the difficulty panel (§6). *s* = normalized task score; τ = 0.70 = pass threshold; `s̄` = mean *s* over samples; `p̂` = measured pass rate.

---

## 1. Core design: staged evidence → withheld result

A paper's headline finding is novel, so "predict the outcome from the setup" is usually not derivable. Following GeneBench-Pro's *recoverable target*, each task **reveals** a premise set *P* of verified facts from the same paper (setup, sibling experiments, upstream results, control designs) and **withholds** a target *T* (a result, control outcome, ranking, or mechanism the paper reports), such that *T* follows from *P* plus standard biology through 2–5 DPs.

*P* is assembled only from the **verified fact sheet** (S1) — never from raw Results, Abstract or Discussion text — because the abstract alone states the end state for 482 of 2,174 non-self-loop triplets (22%) and names the perturbation for 63% of all triplets. **"Reported" means stated in Results or figure-legend text**; a value readable only off a figure image is never a gold source. Methods excerpts may enter *P* only as verified facts (same provenance rule), so a premise that exists only in a figure legend is still usable, but one that exists only in a plot is not.

**Per-paper leak gate (S0):** papers whose mean delta-token recovery from title+abstract exceeds 0.5 are restricted to triplet-fields-only presentation; above 0.8 they are dropped. The distribution is bimodal (0.09 to 0.93 across the 8 papers profiled), so this cannot be a corpus-level assumption.

## 2. Task-design principles (GeneBench-Pro)

Principles enforced by a gate are defined once, in §6: **recoverable target** = V1, **unique answer** = V4, **separation from wrong answers** = V5, **anti-memorization** = the prior-only probe. The remaining principles, each logged pass/fail with evidence in task metadata:

| # | Principle | Operationalization | Enforced by |
|---|---|---|---|
| 1 | Minimal viable prompt | *P* defines the estimand, vocabulary and answer contract. It presents evidence neutrally: never flags which premise is diagnostic ("note that…", "consider…"), never states *T*. | S3 lint, S5 |
| 2 | Threshold robustness | Directions, ordinal bins, tolerance ranges; gold is kept away from bin edges. | S3 lint |
| 3 | Multi-stage inference | 2–5 DPs under GeneBench's counting rule (a distinct fork resolvable from solver-visible evidence; bookkeeping does not count). Pilot failures must spread across DPs (S9). | S2, V3, S9 |
| 4 | Representation / nuisance discovery | ≥1 DP requires picking the right comparator, population or readout frame (composition shift vs per-cell change; a control that rules out a confound). | S2, V3 |
| 5 | Literature-defensible | The gold derivation uses standard biology, not ad-hoc logic. Every DP premise is quote-matched or passes the K-check (§6 V3). | S5, V3 |
| 6 | Controlled evidence | *P* comes only from the verified fact sheet, so cues are tunable — our analogue of GeneBench-Pro's "fully simulated". | S1, S3 |

**Meta-principle** (GeneBench-Pro): *insensitive to nearby defensible choices, sensitive to missing scientifically necessary stages*. V5's must-pass and per-DP must-fail sets are its test.

S5 and S9 tag every issue with a GeneBench-Pro finding type — underspecified estimand, underspecified prompt, alternate defensible answer failing, gold-derivation error, target not recoverable, conflicting evidence in source, contrived setup — and fix in severity order: clarify *P* → re-key or accept the alternative → redesign the recipe → withdraw.

## 3. Agent architecture

```
extraction.json + .meca
 S0  Seed selector (code)      DOI dedupe · frozen paper-family splits · seed tiers · per-paper leak gate · sibling-echo screen
 S1  Fact Verifier (agent)     once per paper → fact_sheets/<paper_id>.json: each fact with char-offset span; drops unverifiable rows; confirms F6 chains are sequential, not parallel arms
 S2  Task Architect (LLM)      recipe: family, P (fact_ids), T, DP chain + licensing premise per DP, answer contract, gold + derivation, judge criteria. Auto criteria are generated by code from the contract.
 S3  Context Builder (code)    instruction.md from P · leak scan · rubric lints (§4)
 S5  Critic (LLM, ≤2)          LifeSciBench's four review categories + §2 checklist; sees S6's best alternative
 S6  Adversary (LLM)           must-pass set · per-DP must-fail set · best alternative answer A′
 S7  Verifiability V1–V5       fail → S2 → drop
 S8  Difficulty panel D        easy → ≤1 structural hardening round
 S9  Trace Auditor (LLM)       on p̂ ≤ 0.25 tasks + 10% sample: grader–prompt mismatch, shortcuts, failures piling on one DP → fix (re-enter S7) or drop
 S10 Packager (code)           Harbor task dirs · dataset.toml · per-task report
```

**Control flow (per candidate).** Every revision is an immutable version `vK`; each gate reads only its own version.
- Budget: ≤4 Architect revisions total, whatever the cause, and ≤$B per candidate. On exhaustion, drop.
- Order, cheapest first, stopping at the first failure: S3 lints → S6 → S5 → V1, V3 → V4, V5 (fixed answers, grader only, no solver) → V2 → D.
- V2 and D need ≥8/10 and ≥10/12 **valid** trials; otherwise retry once, then hold. Never pass by default. Refusals are graded 0 and reported as a refusal rate; only transient errors make a trial invalid.
- D says easy → ≤1 structural re-target. If that version then fails V or is still easy, keep the **last version that passed V** with its own D stats; it goes to the easy quota or the RL pool, else it is dropped.
- S9 may edit only rubric or judge text, then regrade with no re-solve. Any `instruction.md` change is a new version and re-runs V2 and D.
- Ship only when instruction, rubric, V report and D stats share one version id.

**Roster** (constrained by what actually works here: Gemini key rejected; Opus 5.5 refuses ~40% of these bio prompts).
- Generator S1–S6: Opus 4.8 (S1 reuses the claude-agent-sdk loop, so the generator is Anthropic).
- Held-out eval model: Opus 5 — same family as the generator, reported as a limitation.
- Panel (V2/D): GPT-5.5 and Kimi K3. Judge: GPT-5.5, with Kimi K3 regrading all panel trials so self-preference on GPT-5.5's own solves is measured. Adversary: Kimi K3.
- Optional third panel family: an open-weight model served locally, which also becomes the RL base policy.

**One grading library.** `grade.py` exposes `grade(answer_text, tests_dir, judge) → {rewards, criteria}` plus a CLI; `test.sh` only calls it. S7/S8 call it in-process through the repo's `CallManager` (budget, ledger, checkpoints). V4, V5 and the constant-answer probe are fixed answers needing no container. V2 and the prior-only probe are emitted as probe tasks `p2c/<id>--openbook` / `--prior` — same `tests/`, different `instruction.md` — never published, since they reveal *T*'s evidence. Harbor runs end-to-end at M0, on a 5% parity sample per batch (rewards must match the in-process runner), and on the released set.

**Storage.** `outputs/task_synthesis/<run>/{split_manifest.json, fact_sheets/, candidates/<id>/v<K>/{recipe.json, rubric.json, instruction.md, gates.json}, calls.jsonl, checkpoints/, summary.json}`. `task_id = hash(paper family, task family, sorted T fact_ids)`. Recipes reference `fact_id`s and never copy text.

**Reuse:** `xml_parser.parse_paper`; `sources.discover_sources`; `splits.assign_splits` with `existing_manifest=<benchmark100_v2_opus5 manifest.json>` (456 post_jan papers are already frozen — extend, don't re-freeze); `evidence.retrieve_evidence` for V2 context; from `checks.py` only the text primitives plus a new `locate_quote` returning char offsets for V1; from `optionset.py` only the polarity/null helpers for the direction-cue scan; `api.CallManager`; `solver_gate`'s role policy and no-repair rule, not its MCQ scorer. S1 subclasses `AgenticRefinePipeline`, overriding `_extract` (load the existing extraction) and `_extra_mcp_tools` (a `verify_fact_sheet` tool).

**Cost.** Per candidate per pass: generation ≈$0.8; ~22 solves ≈$1.4; graded responses ≈$2.1 (batched judging) to $5.6 (per-criterion) → **≈$13–36 per accepted task** (cf. QA v3 $0.47/item, AutoSciBench $22–58/task). Controls: cheapest-first gates with early exit; 1 batched judge sample during gating, 3 per-criterion samples only on accepted tasks and in eval; sequential D (4 solves, stop at 4/4 or 0/4); Batches API for S2–S6; a per-candidate cap in `CallManager`. M1 reports $/accepted task per family before M3 is approved.

## 4. Rubric construction (LifeSciBench)

**Criterion schema:** `{id, section, text, points, kind: auto|judge, critical, check, capability, dp, grounding_span}`. Sections mirror the DP chain. `capability` ∈ direction · comparison · control-logic · mechanism · design · uncertainty. `grounding_span` ∈ {paper span, instruction quote, K-premise id}. 6–20 criteria per task.

**Writing rules (LifeSciBench App. C):** specificity · atomicity · evaluability from the response alone · grounding · non-redundancy · operational usefulness · **question–rubric consistency** (reward only what the question asks). Each criterion is self-contained: it names the entity, direction and context, never "the main finding".

**Points.** Positive integers 1–10; **no negative points** (RaR measured no gain from synthetic pitfalls, 37.2 with vs 37.2 without; and in rewardkit only `weighted-sum` admits negative weights, which can push *s* outside [0,1]).

**Points budget, lint-enforced.** Core-answer criteria (the fields the question asks for) take share `w_core` with `0.30 < w_core < τ` — so neither a wrong core answer nor a core-only answer can pass. Caveat/uncertainty criteria ≤ 5%, and no single judge criterion can lift a core-only answer over τ. Per family: `w_core = 0.60` for F1–F4 and F6; `0.40` for F5 (key-entity criteria, judged); for F7 there is no auto core, and its pass rule is instead *all `essential`-tagged criteria satisfied, and s ≥ τ*.

**Critical fields.** Each task declares ≥1 `critical` core field — the discordant marker, the non-null arm, the top-ranked arm. A wrong critical field sets `s_core = 0`. This, not the budget, is what makes a wrong core answer unpassable: on an 8-down/1-up marker panel an "all down" policy otherwise reaches τ exactly.

**Notice/act.** For each key DP, a "recognizes cue X" item is diagnostic-only and carries **0 points**; only the paired "uses X to conclude Y" item scores, and its credited quote must contain both the cue and the inference. This blocks the echo exploit while keeping GeneBench's main failure mode visible.

**S3 lints:** budget bounds hold · ≥1 critical field · every item has a grounding span and capability tag · no duplicates · no item references the paper · no item restates a premise given in *P*.

**S5 anchor rule** (LifeSciBench's ≥90% expert-consensus standard): every criterion and DP premise is either quote-matched (V1/V3) or affirmed 3/3 by cross-family reviewers; otherwise the criterion is dropped. Review cycles run to a clean pass, capped at 6.

## 5. Scoring, normalization, LLM verification

**Task score.** `s = w_core · s_core + (1 − w_core) · (Σ judge points awarded / Σ judge points)`, `s ∈ [0,1]`. `s_core ∈ [0,1]` is a family-specific function computed in code, because several families are not sums of independent items:

| Family | `s_core` |
|---|---|
| F1 | macro-recall over the direction classes present in gold (balanced accuracy) |
| F2, F6 | exact enum match over all graded arms |
| F3 | fraction of concordant gold-comparable pairs; pairs inside a declared tie set excluded |
| F4 | exact choice |
| F5 | fraction of required key entities (judged) |

**Pass** = `(s ≥ τ) ∧ (s_core ≥ 0.9) ∧ (s_judge ≥ 0.5 where the family has a judge section)`, with **τ = 0.70 fixed** (LifeSciBench). Each family also declares a **chance floor**, measured by scoring the best constant policy and a uniform random answer through the real verifier; `s_core` is reported raw and chance-corrected.

**Across k samples per (model, task):** `k = 20` for evaluation; `k = 4 → 12` escalating for the difficulty panel; `k ≥ 8` for the RL band. **k, not suite size, is the lever for ranking power** — simulated on a 250-task/85-paper suite, going k=10→20 lifts the power to separate two models 5pp apart from ~60% to ~94% while the CI on a single model's score barely moves (±0.056 → ±0.055), because that CI is dominated by task variance. Buying the same power with more tasks would cost ~250 extra tasks at $13–36 each, versus ~$130–330 per model for the extra samples.
- **Normalized score** = mean over tasks of `s̄_t`, problem-weighted so each task counts equally regardless of rubric length.
- **Pass rate** = mean over tasks of `p_t`; reported per family and family-macro-averaged, and pooled only alongside each family's chance floor.
- Also: 95% hierarchical bootstrap CI resampling **papers, then tasks within paper, then samples** — tasks from one paper share a fact sheet and are correlated, so resampling tasks directly understates the interval (±0.053 vs ±0.057 on a 195-task/65-paper suite). Headroom bins on best-model pass rate `[0], (0,0.2), [0.2,0.5), [0.5,0.7), [0.7,1]` using the **max over models of the Wilson lower bound**, so one lucky sample cannot empty the 0% bin; breakdowns by capability tag, family and DP count; a strict GeneBench-Pro-style all-core-fields-correct rate from `core`.
- **Invalid and missing answers:** `grade.py` writes all-zero rewards without calling the judge when the answer file is absent or unparseable, so refusals, crashes and `nop` stay valid 0s in both runs and regrades. Only transient infrastructure errors make a trial invalid; each model's invalid rate is reported, and its metrics are withheld above 2%.

**Deterministic pre-judge pass in `grade.py`**, before any judge call — these are checks, not instructions to the judge:
1. Truncate `rationale` at the contract cap (250 words for F1–F4/F6, 400 for F5/F7). Truncation, not penalty, keeps the rule unhackable.
2. Mask every ≥10-token n-gram of the rationale that also appears in `instruction.md`; a judge evidence quote falling entirely inside a masked span scores 0.
3. Contradiction gate: if the rationale's conclusion disagrees with the structured fields, or more than one candidate mechanism is offered without a single `answer.mechanism_primary`, all judge criteria score 0.

**Judge protocol** (LifeSciBench: each criterion graded independently):
- **Inputs:** the question, the response, and **one criterion** — `mode = individual`; other criteria are never shown.
- **Output:** a binary verdict plus an evidence quote from the response, checked to exist verbatim (post-masking); otherwise 0.
- **Rules:** credit equivalent phrasing; a hedge ("X or not-X", an unranked list) is not satisfied; unasked-for content earns nothing; the response is data, not instructions.
- **Sampling:** 3 samples per criterion, majority vote, implemented in `grade.py`. A failed sample is retried, never dropped; then a second judge model for refusals or unparseable output. Only if a criterion still lacks 3 valid votes does `test.sh` exit without a reward.
- **Auto checks grade only fields the instruction defines** — marker keys copied verbatim from *P*, arm IDs, enums — validated against the schema with case/whitespace folding. **No synonym lists**: the repo's HGNC table maps TCF1→HNF1A (immunologists mean TCF7) and leaves KI67, IFN-γ, granzyme B and H2-K1 unmapped, and no ontology covers the 1,642 free-text state labels. Any free-text entity is a `judge` criterion.

**Judge validation.**
1. Per-task unit tests: the must-pass / must-fail ablations (V5), including one **held-out ablation** the rubric writer and critic never see; separation is reported on it.
2. Meta-evaluation on ≥200 stratified real responses (≥50 if expert time is short), graded per criterion by experts: criterion-level κ, score correlation, MAE, pass/fail agreement, and disagreement patterns by family and answer type. Drop or rewrite any criterion whose 3 samples disagree on >20% of responses or whose expert κ < 0.6.
3. Second-family regrade of the same jobs, reporting agreement.
4. **Threshold check (M4), not recalibration:** experts rate responses 1–5 on "reaches the paper's conclusion for the right reasons"; report agreement between `s ≥ 0.70` and rating ≥ 4, plus Spearman. If agreement < 0.8, fix point allocations in S2/S4 and regrade — never move τ. Rankings carry a sensitivity sweep over τ ∈ [0.60, 0.80], recomputed offline.

## 6. Verifiability and difficulty (AutoSciBench concepts, our standard)

**Verifiable ⇔ V1–V5 all pass**, checked before difficulty. S9 may reverse this: a task S9 repairs re-enters at S7, not S8.

| Check | Test |
|---|---|
| **V1 Provenance** | Every gold field and core criterion maps to a verbatim span in `paper.txt` (deterministic quote match, Results or figure-legend text only). |
| **V2 Well-posedness** | Premise spans in full **plus the target's own evidence span with the result clause redacted** → 2 non-generator models × 5 samples. Require `s_core = 1.0` on ≥8/10 and ≥3/5 per model. Separately, the gold rationale from `solve.sh` must score `s_judge ≥ 0.9` on 3 judge repeats. Scoring `s_core` rather than `s` removes the perverse pressure to make judge criteria echo-satisfiable. *V2 does not test ambiguity; V4 does.* |
| **V3 Derivability** | **Stepwise, solver-visible only.** For DP *i*, a checker sees `instruction.md` plus the resolutions of DPs 1..*i*−1 — never DP *i* or later, never the gold — and answers DP *i* as a shuffled closed choice between the recipe resolution and its plausible-wrong fork(s). Pass iff every DP gets the recipe resolution in ≥5/6 samples (2 families × 3), the last DP's resolution equals the gold, and each licensing premise is (a) a quote-matched sentence of `instruction.md` or (b) a **K-premise**: the leak scan finds no entity or direction overlap with any withheld fact, and 3/3 cross-family judges rate it closed-book as "true in general biology, independent of this study". Fields no premise licenses move to an `unlicensed` block that is reported but never scored. |
| **V4 Uniqueness** | The Adversary (2 samples) writes its strongest alternative A′ with a derivation from *P*. Three judges (≥2 families, not the generator) see *P*, the question, and the gold derivation and A′ in random X/Y order, **not knowing which is gold**: "which does *P* better support — X, Y, or neither?" Gold 3/3 → pass. A′ preferred ≥2 → the gold is suspect: re-key from S1 or drop. Otherwise → add the disambiguating constraint to *P*, or accept A′ too, then re-run V2/V5. |
| **V5 Grader separation** | **Must-pass (≥ τ+0.05 on 3 judge repeats):** paraphrase, synonyms, an alternative defensible derivation. **Must-fail (`s_core ≤ 1/3` and `s ≤ τ−0.25`):** textbook prior, flipped direction, other cell type, the constant-enum answer plus the stem's own mediator noun, and for each DP the answer reached by taking its plausible-wrong fork. **Must-not-pass (`s_judge ≤ 0.3`, so the pass conjunction fails):** correct core with wrong mechanism, correct core with no rationale, hedge-everything. Gold−ablation gaps are reported per class. |

**Difficulty D.** Panel: 3 models, ≥2 families, excluding the generator and the held-out model; `k = 4` (n = 12). Report `p̂` = the unweighted mean of the per-model pass rates, with per-model rates and the min–max range — **not a binomial CI**, since the 12 trials are clustered by model. Classes (easy ≥0.75 / medium 0.25–0.75 / hard <0.25 / unsolved 0) are descriptive labels for stratification, never a reported outcome; any task near a live boundary (`p̂ ≥ 0.58` for the easy cap, `p̂ ≤ 0.25` for the hardening or unsolved split) is re-run at `k = 12` per model before the cap acts on it.

**Probes.** *prior-only*: the same question with *P* removed and a one-line setup built from species, start cell type and perturbation. Flag `textbook` iff the prior-only Wilson lower bound ≥ 0.25. *constant answer* (per family): the best constant policy must score `s̄ ≤ 0.35` and pass rate 0 through the real verifier.

**Selection.** The eval split caps easy at 15%, `textbook` at 10%, and `unsolved` at 20%; an unsolved task enters only after S9 confirms failures fall on licensed DPs and after a blind re-derivation of the gold. The RL pool keeps tasks whose reward actually varies **under the policy being trained**: `0.1 ≤ s̄ ≤ 0.9` and `sd(s) ≥ 0.08` over `k ≥ 8` rollouts of the starting policy, re-estimated at each refresh — not from the frontier panel, which is far stronger than the trainee. For the RL reward each judge criterion contributes the **mean of its 3 samples** (a finer grid); the eval pass decision uses the majority verdict.

**Hardening** (≤1 round, structural). The Architect is told only `tier = easy`; it never sees panel answers or rationales — the repo's own finding is that showing solver feedback pushes the generator to make the science weird rather than the reasoning deeper. It re-targets from the fact sheet: a contrastive arm, the discordant readout, a two-premise integration. Panel `p̂` is used **only for selection**; tiers, headroom and difficulty claims are reported **on the held-out model**. A held-out model removes solver-identity bias, not composition bias — the suite is still conditioned on panel-measured difficulty, so a randomly composed 100-task reference slice (no caps, no hardening) runs alongside the main suite to measure the selection effect, and the frozen composition is published with every score.

## 7. Task families (v1)

Supply = all of `immunology_top300` (4,083 triplets) / **seed tier** (§8) — A′ for F1–F4 and F6, A for F5 and F7. These are seeds, not tasks, and are corpus-wide; the eval-split pools are in §8.

| Family | *P* → *T* | Answer contract | `s_core` | Supply |
|---|---|---|---|---|
| F1 Marker direction panel | setup + sibling results → per-marker change | `{marker: up/down/unchanged}`; ≥1 discordant marker **whose direction is licensed by a premise other than the majority trend** | balanced accuracy | 513 / 210 |
| F2 Paired control | main effect → outcome of a control/blockade/rescue arm | enum per arm, **paired with a non-null arm** | exact | 609 / 69 nulls; 1,548 / 275 control-language |
| F3 Contrastive ranking | arms on one start state + the readout → ranking | ordered list with tie sets | pair concordance | 482 groups, 238 papers / 108, 93 |
| F4 Inverse inference | observed delta → which perturbation | choice among same-paper perturbations with distinct end states | exact | 482 / 108 (same pool as F3) |
| F5 Mechanism | results incl. controls → the paper's explicit mechanism | free text + key-entity fields | judged key entities | 1,726 / 825–907 |
| F6 Two-hop chain | hop-1 result → composed or intermediate outcome | enum + state description | exact | 387 / 208 pairs, 39 papers |
| F7 Discriminating experiment | competing hypotheses + results so far → the paper's decisive control | free text | n/a — `essential` criteria + s ≥ τ | from the F2 pool |

The answer is always `{"answer": {...}, "rationale": "..."}`. **End-state labels are replaced by neutral identifiers (State A/B)** and the contract enumerates every candidate label in the paper, because 1,102 of 2,174 non-self-loop end-state labels (51%) carry their own valence ("Activated CD8+ T cell", "M1 macrophage", "Pro-fibrotic lung fibroblast") — a token scan cannot flag that, since the label belongs to the answer vocabulary. The instruction states honestly that the rationale is graded.

## 8. Data, splits, integrity

- **Corpus.** `immunology_top300` and `cancer_top300` **share 109 DOIs** (cancer immunology); deduped the two post_jan corpora are **455 unique papers**, not 564. Always dedupe across corpora before splitting or counting.
- **Splits: keep the existing frozen 80/10/10 paper-family manifest. Do not re-freeze** (§11.4 settles this). It already covers 428 of the 455 papers, and the 27 it omits carry **zero** eval-grade seeds — so raising the eval fraction would mean *moving* papers out of train, which invalidates the MCQ dev/test exports already published against this split (88+112, 124+132, 146+154 items across three runs). Sharing the split is also a feature: a model RL-trained on our train pool is automatically clean against the MCQ test set, and vice versa.
- **Seed tiers** (triplet quality, pre-generation; orthogonal to §6's difficulty classes). All require `evidence_source = primary` and a rich delta (no "no distinct phenotype" / "not measured" language, and ≥2 named markers or a quantitative value). The mechanism requirement is **family-dependent** — only F5 and F7 grade a mechanism, so requiring `explicitly_described` everywhere discards 26% of usable seeds for no benefit:

  | Tier | Definition | Corpus-wide | In eval split (dev+test) |
  |---|---|---|---|
  | **A** — F5/F7 seeds | + `validated` + `explicitly_described` | 1,109 train | **248** / 63 papers |
  | **A′** — F1–F4, F6 seeds | + `validated` | — | **337** / 76 papers |
  | **B** — RL | + (`validated` ∨ `single_experiment`) | **2,653** train | 570 |
  | **C** — paired null arms | self-loop with explicit abolish / no-change language; exempt from the rich-delta filter | — | 73 / 32 papers |

  Tier C never stands alone and never carries the gold: it is the null half of an F2/F3 pair whose other half is a non-null arm from the same paper and start state. Without it those families are gameable, and the rich-delta rule excludes exactly the wording null arms use.
- **Leakage screens beyond the token scan:** a **sibling-echo screen** at S0 rejects a (*P*, *T*) group when ≥1/3 of *T*'s distinctive perturbation tokens already appear in a revealed sibling's `delta` or `mechanism` (48% of triplets cross-mention corpus-wide); and the `mechanism` field of any sibling is **banned from *P*** whenever *T* is a control or blockade arm, because 119 triplets put a control outcome in that field verbatim.
- **Exclusions:** speculated, preliminary, reference; intent-leaking perturbations (133 triplets, 6% of non-self-loop); papers with no parseable methods, for setup details.
- **Extraction quality:** the strict per-paper extraction pass rate is 16–40%, so S1 verification is mandatory. Triplets are candidate facts; the evidence spans are the ground truth.
- **Expected yield.** Eval tasks come only from dev+test paper families: deduped across both post_jan corpora that is **85 papers carrying 337 A′ seeds** (248 of them Tier A), split roughly 40/60 dev/test, so the private half is the larger one. At 2–3 candidates per seed, ≤5 tasks per paper and about a third surviving S1+V+D, expect **≈220–330 eval tasks** — the target is **250, with ≥150 in private test**. RL = 2,653 Tier-B triplets in train-split papers, plus the pre2026 pools (~41k rows), which are wrong for eval and fine for RL. M1 measures the real per-gate survival and replaces these numbers.
- **Contamination slice.** Only 20 eval papers (80 Tier-A seeds) are from 2026-05/06; the rest are 2026-01..04, inside plausible training windows, and 6 are pre-2026. Report the headline on the full suite **and** on the 2026-05/06 slice. A large gap between them, or a high prior-only pass rate (§6), is the contamination signal.
- **Integrity:** a canary GUID in `instruction.md`; `tests/` never visible to the agent; items stay `provisional` until human review (every unsolved task plus 10% per family), captured in a `viewer/task.html` that shows the instruction, *P* highlighted in `paper.txt`, gold, rubric, V/D report and 2 sampled solves, writing verdicts to `reviews.jsonl`.

## 9. Harbor packaging

```
p2c-<split>-<family>-<id>/
  instruction.md   # canary · P · question · answer schema · "Reply with exactly one JSON object in this schema;
                   #   it is saved verbatim to /app/output/answer.json and graded, including the rationale."
  task.toml
  solution/solve.sh   # writes the gold answer + gold derivation as rationale
  tests/  test.sh · grade.py · rubric.json · gold.json · question.md
```

```toml
schema_version = "1.4"
artifacts = ["/app/output"]            # top-level, before any table; a DIRECTORY, so no-answer trials
                                       # stay valid 0s and replay on regrade (a missing file artifact
                                       # is recorded "failed" and makes every regrade exclude the trial)
[task]
name = "p2c/<split>-<family>-<id>"     # required by dataset.toml and by regrade matching
version = "1.0.0"                      # rubric/judge edit → minor (regrade); instruction edit → major (rerun)

[metadata]                             # family, DPs, provenance spans, V1–V5 verdicts, split, models —
                                       # fixed at packaging; p̂/s̄/class live in the S10 report outside the
                                       # task dir, since editing task.toml changes the digest and breaks --diff

[environment]
docker_image = "<registry>/p2c-base@sha256:<digest>"   # shared image: python + pinned judge client + /app/output
network_mode = "no-network"
[agent]
timeout_sec = 1800
[verifier]
environment_mode = "separate"
timeout_sec = 1800                     # a verifier timeout is invalid and is not retried
[verifier.environment]                 # no image here and no tests/Dockerfile ⇒ reuses [environment].docker_image
network_mode = "allowlist"             #   and uploads tests/ at runtime, so regrade needs no rebuild
allowed_hosts = ["<judge API host>"]
[verifier.env]
JUDGE_API_KEY = "${JUDGE_API_KEY}"
JUDGE_MODEL = "${JUDGE_MODEL:-<default>}"   # second-family regrade via --ve
```

There is **no `environment/` dir and no `tests/Dockerfile`**: one shared digest-pinned `p2c-base` image serves both sandboxes, so the verifier never runs `pip install` (its allowlist holds only the judge host) and ~1k per-task image builds are avoided.

**`reward.json`** (flat, numeric): `{"reward": s, "core": s_core, "judged": s_judge}`. `pass` is **not** stored. `grade.py` also writes `/logs/verifier/grading.json` (per-criterion verdicts, all 3 samples, evidence quotes, judge model), which Harbor syncs back; pass, pass@k, τ sweeps, point re-weights and capability stats are recomputed offline from it, so a regrade is needed only when the judge model or prompt changes. Multi-key rewards disable Harbor's built-in pass@k.

**Solver agent:** a custom external `SingleShotAgent` (`BaseAgent`), so inference runs on the Harbor host and the sandbox stays offline. It always writes the completion via `environment.upload_file` — including on a refusal or empty output — and records `refusal: true` in metadata, so refusals are graded 0 rather than excluded. Run with `-r 3`, so only persistent transient errors invalidate a trial. It records tokens and cost.

**Sanity runs:** `-a oracle` must give `core = 1.0` and `reward ≥ 0.9`; any judge criterion the oracle fails in ≥2 of 3 samples is a rubric or gold-rationale defect and returns to S2. `-a nop` must give a valid 0, in both a run and a regrade.

**Infrastructure:** no-network and allowlist need docker or podman on a host whose kernel passes Harbor's nftables egress probe, or Daytona; singularity rejects both. The data host has no container runtime, so M0 names its runner first. The in-process runner needs no containers.

## 10. Milestones

1. **M0, harness:** `p2c-base` image + runner; `SingleShotAgent`; `grade.py` with a golden-file test per family (gold, each ablation, best constant policy, uniform random); 10 hand-built tasks from F1, F3, F5. Oracle/nop pass; ablations separate.
2. **M1:** S0–S6 on 20 papers for F1–F4; hand review; measured per-gate survival and $/accepted task.
3. **M2:** S7–S9 gates; judge meta-evaluation on ≥200 responses.
4. **M3:** full run over both post_jan corpora (deduped) → eval v0 (~250 tasks) + RL pool.
5. **M4:** expert review, threshold check, contamination slice, release of splits.

## 11. Open decisions

1. **Code location:** this repo importing papers2cells as a uv path dependency, *or* a sibling `task_synthesis/` inside papers2cells (that repo's convention).
2. **Expert availability** for judge meta-evaluation and the threshold check. Without experts, validation is cross-family LLM only and items stay provisional.
3. **Container host** for Harbor parity runs and eval.

**Settled — eval-split size.** Keep the frozen 80/10/10 split and target **250 eval tasks (≥150 private)** from the 85 deduped dev+test papers / 337 A′ seeds, buying ranking power with `k = 20` rather than more tasks. Re-freezing was the obvious move and is wrong on four counts: there are no free eval-grade papers to promote (the 27 unassigned papers carry zero seeds), so raising the eval fraction means moving papers out of train and invalidating published MCQ exports; the shared split keeps the RL pool clean against the MCQ benchmark; moving 10% of papers would cost ~265 Tier-B seeds for ~30 extra eval tasks; and k is ~20× cheaper per unit of ranking power. The two things that *did* raise capacity were free: deduping the 109 papers shared between the corpora, and dropping the mechanism requirement for the five families that never grade a mechanism (248 → 337 seeds).

---

### Appendix: one worked task, and one rejected one

**F2/F5, anti-MDA5 fibroblast paper** — *illustrative; this triplet is `single_experiment` with a no-phenotype delta, so it fails seed Tier A. It exercises the F2 contract only.*

- ***P*:** IMR-90 lung fibroblasts treated (1 µg/ml) with recombinant MDA5, with patient anti-MDA5 IgG1, or with the MDA5 + anti-MDA5 immune complex. Readout: real-time impedance. The complex gives the highest normalized cell index (~+25% at 121 h; doubling time 54.4 h → 46.8 h). No cytotoxicity in any arm.
- ***T*** (two arms, different gold — a constant answer then scores `s_core = 0.5` and fails the critical-field gate):
  (a) anti-MDA5 IgG1 alone in complement-replete serum → **moderate rise from ~60–70 h; strongest IFN-α/β/γ and pro-collagen α1 secretion**;
  (b) anti-MDA5 ± MDA5 in **serum pre-treated to remove a heat-labile activity** → **lost** (arm (b) is `critical`).
  The stem says "heat-labile serum activity", not "complement" — the mediator noun is a judge criterion, never a stem cue.
- **DPs:** IgG1 immune complexes fix C1q in serum-containing medium → removing that activity should remove the complex-dependent response → antigen-only activation does not require it, so arm (a) persists.
- **Rubric** (total 20, `w_core = 0.55`): auto [7] arm (b) = lost (critical); [4] arm (a) = increased · judge [4] attributes the loss to complement activation by the IgG immune complex; [3] names the classical pathway / Fc-dependent C1q engagement; [2] uses the antigen-alone premise to explain why arm (a) survives. A correct core with a wrong mechanism scores 11/20 = 0.55 < τ.
- Note the enum is **lost** (abolished or reduced), not "abolished": nothing in *P* licenses complete versus partial loss, so grading that distinction would violate threshold robustness. A fourth DP asserting "abolished, not merely reduced" is exactly what V3 removes.

**A task V3 rejects.** In the TOX/Treg paper, the premise "TOX rises alongside PD-1, TIM-3, CTLA-4, CD44, Ki67 **and CD103** during tumor progression" does not license the withheld "CD103 **increases** after TOX deletion while the others fall" — the premise *anti-*licenses it. V3 moves CD103 to the `unlicensed` block, which guts F1's discordance requirement. The derivable version uses different premises from the same paper: TOX-KO Tregs upregulate stem-like genes and gain *Tcf7*/*Ccr7* chromatin accessibility while losing *Ccr8* accessibility, which genuinely licenses the withheld flow result **CCR4↑ / CCR8↓**. That is the F1 pattern to build.
