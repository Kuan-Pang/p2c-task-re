#!/usr/bin/env python
"""M0 refusal-rate probe: 20 realistic pilot prompts x the roster's models (PLAN §3, §11 M0).

PLAN §3 states the rule this script measures: "M0 measures each role's refusal rate on 20 pilot
prompts ... Any role above 10% is swapped in config." The prompts are the requests M1 will really
send -- S1 fact verification, S2 task authoring, S5 critique, S6 adversary, the S0/V3/V4 checker
votes, a panel solve and a judge criterion check -- built at run time from the five pilot papers of
configs/experiment/pilot.yaml (paper text and extracted triplets, read through `p2c_task_re.data`)
and from the three hand-built tasks in examples/tasks.

Every prompt goes to each model configs/models/default.yaml assigns to its role, `--samples` times,
through `llm.complete` with a `Ledger` capped at `--budget-usd`. Refusals and their categories,
empty answers, errors, tokens and cost land in `<out-dir>/`: `calls.jsonl` (the ledger),
`prompts.json` (what was sent), `responses.jsonl` (what came back, so a refusal written as prose
can be told from real work), `results.json` (one record per trial) and `summary.md` (the two tables).
docs/m0_refusal_probe.md reports a run.

    source ~/.api_keys && uv run python scripts/probe_refusals.py
    uv run python scripts/probe_refusals.py --dry-run    # render and list the prompts, call nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from p2c_task_re import families
from p2c_task_re.data.extraction import PaperExtraction, load_corpus
from p2c_task_re.data.jats import PaperText, parse_jats
from p2c_task_re.data.meca import MecaIndex
from p2c_task_re.grading import grade
from p2c_task_re.ledger import BudgetExceeded, Ledger
from p2c_task_re.llm import LLMError, RoleSpec, complete, load_roster

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
EXAMPLE_TASKS = REPO / "examples" / "tasks"
ABLATIONS = REPO / "tests" / "fixtures" / "ablations"
DEFAULT_OUT = REPO / "outputs" / "probes" / "m0_refusals"

# The five pilot papers (PLAN §11 M1), pinned in configs/experiment/pilot.yaml.
P1 = "2026.02.17.706358"  # Api5-FGF2, breast epithelium
P2 = "2026.03.11.710967"  # microglial CD22, Huntington's
P3 = "2026.04.25.720790"  # PRMT7 restricts CD8+ expansion
P4 = "2026.04.08.717325"  # Cxcl11, zebrafish kidney regeneration
P5 = "2026.05.28.728506"  # γc cytokines license IL-1 inflammation

# The three hand-built M0 tasks (examples/tasks/README.md).
TASK_F2 = f"{P1}-f2-fwd-01"
TASK_F4 = f"{P3}-f4-bwd-01"
TASK_F5 = f"{P3}-f5-01"


# ── inputs ────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExampleTask:
    """One hand-built task folder, plus the ablation answers in tests/fixtures/ablations/."""

    task_id: str
    instruction: str
    question: str
    rubric: dict
    gold: dict
    ablations: dict[str, dict]

    @classmethod
    def load(cls, folder: Path) -> ExampleTask:
        tests = folder / "tests"
        fixtures = ABLATIONS / folder.name
        return cls(
            task_id=folder.name,
            instruction=(folder / "instruction.md").read_text(encoding="utf-8"),
            question=(tests / "question.md").read_text(encoding="utf-8"),
            rubric=json.loads((tests / "rubric.json").read_text(encoding="utf-8")),
            gold=json.loads((tests / "gold.json").read_text(encoding="utf-8")),
            ablations={
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(fixtures.glob("*.json"))
                if path.name != "expected.json"
            },
        )

    def criterion(self, criterion_id: str) -> dict:
        for criterion in self.rubric["criteria"]:
            if criterion["id"] == criterion_id:
                return criterion
        raise KeyError(f"{self.task_id} has no criterion {criterion_id!r}")


@dataclass(frozen=True)
class Sources:
    """Everything the prompt builders read: pilot papers, their text, the example tasks."""

    papers: dict[str, PaperExtraction]
    texts: dict[str, PaperText]
    tasks: dict[str, ExampleTask]
    judge_prompts: dict[str, str]


def load_sources() -> Sources:
    data = OmegaConf.load(CONFIGS / "data" / "post_jan.yaml")
    pilot = OmegaConf.load(CONFIGS / "experiment" / "pilot.yaml")
    meca = MecaIndex(Path(data.meca_root))
    corpus = load_corpus(
        {name: Path(folder) for name, folder in data.extraction_dirs.items()},
        list(data.corpus_priority),
        meca,
    )
    papers = {paper_id: corpus[paper_id] for paper_id in pilot.pilot.papers}
    return Sources(
        papers=papers,
        texts={pid: parse_jats(meca.open_jats(paper.uuid).xml) for pid, paper in papers.items()},
        tasks={
            folder.name: ExampleTask.load(folder)
            for folder in sorted(EXAMPLE_TASKS.iterdir())
            if folder.is_dir()
        },
        judge_prompts=grade.load_prompts(),
    )


# ── prompt rendering ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prompt:
    """One pilot prompt: what a role is asked, in the shape M1 sends it."""

    id: str
    role: str
    paper_id: str
    summary: str
    user: str
    system: str | None = None
    schema: dict | None = None

    @property
    def words(self) -> int:
        return len((self.system or "").split()) + len(self.user.split())

    def meta(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "paper_id": self.paper_id,
            "summary": self.summary,
            "words": self.words,
            "chars": len(self.system or "") + len(self.user),
            "structured_output": self.schema is not None,
        }


S1_SYSTEM = (
    "You are the Fact Verifier of a benchmark-construction pipeline for perturbation biology. You check "
    "machine-extracted facts about one paper against that paper's own Results text and keep only what the "
    "text states, each with the sentence that states it."
)
S2_SYSTEM = (
    "You are the Task Architect of a benchmark-construction pipeline for perturbation biology. From verified "
    "facts of one paper you write a single reasoning question: the premises you reveal, the result you "
    "withhold, the derivation that gets from one to the other, and the rubric that grades it."
)
CRITIC_SYSTEM = (
    "You are the Critic of a benchmark-construction pipeline for perturbation biology. You review a drafted "
    "question, its reference answer and its rubric, and report the defects a careful reviewer would raise."
)
ADVERSARY_SYSTEM = (
    "You are the Adversary of a benchmark-construction pipeline for perturbation biology. Given a question "
    "and its reference answer, you write the strongest alternative answer the stated premises could support, "
    "so the question can be tested for ambiguity."
)
CHECKER_SYSTEM = (
    "You are a checker in a benchmark-construction pipeline for perturbation biology. You answer one narrow "
    "question about a piece of scientific text with a verdict and a one-line reason, and nothing else."
)


def vote_schema(options: list[str]) -> dict:
    """The structured-output schema of a checker vote (strict subset: every field required)."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": options},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


def results_subsection(text: PaperText, title_contains: str) -> str:
    """The one Results subsection whose heading contains `title_contains`, heading included."""
    wanted = title_contains.casefold()
    for section in text.sections:
        if section.kind == "results" and wanted in section.title.casefold():
            return text.section_text(section)
    raise KeyError(f"no Results subsection whose title contains {title_contains!r}")


def triplet_block(paper: PaperExtraction, rows: list[int]) -> str:
    """Extracted triplets as numbered fact candidates, `fN` for the prompt, `tNN` for provenance."""
    blocks = []
    for n, row in enumerate(rows, start=1):
        triplet = paper.triplets[row]
        lines = [
            f"start state: {triplet.starting_cell_state}",
            f"perturbation: {triplet.description_of_perturbation}",
            f"end state: {triplet.ending_cell_state}",
            f"assay: {triplet.cell_assay}",
            f"reported change: {triplet.delta}",
        ]
        if triplet.mechanism:
            lines.append(f"stated mechanism: {triplet.mechanism}")
        blocks.append(f"f{n} (table_2 row t{triplet.index:02d})\n  " + "\n  ".join(lines))
    return "\n".join(blocks)


def rubric_block(task: ExampleTask) -> str:
    return "\n".join(
        f"{c['id']} [{c['points']} points, {c['section']}] {c['text']}" for c in task.rubric["criteria"]
    )


def fact_check_user(src: Sources, paper_id: str, section_title: str, rows: list[int]) -> str:
    """S1: Results text + extracted triplets -> verify each fact with a verbatim quote."""
    return f"""# Paper excerpt (Results, verbatim)

{results_subsection(src.texts[paper_id], section_title)}

# Candidate facts extracted from this paper by an automated pipeline

{triplet_block(src.papers[paper_id], rows)}

# What to do

For each candidate fact, decide whether the excerpt above states it. Supporting evidence must be a
sentence of the excerpt copied character for character; a fact with no such sentence is unsupported,
however plausible it is. Keep the directions and numbers exactly as the excerpt gives them.

Return one JSON object and nothing else:
{{"facts": [{{"id": "f1", "verdict": "supported" | "partly_supported" | "unsupported",
            "quote": "<sentence copied from the excerpt, or empty>", "note": "<one line>"}}]}}"""


def authoring_user(
    src: Sources,
    paper_id: str,
    rows: list[int],
    *,
    family: str,
    direction: str,
    contract: str,
    withheld: str,
) -> str:
    """S2: verified facts -> a question with premises, a withheld target and a rubric."""
    return f"""# Verified facts from one paper (the fact sheet)

{triplet_block(src.papers[paper_id], rows)}

# What to write

One {family} {direction} question from these facts.

- Withheld target: {withheld}. The premises must never state it, nor the direction of its effect.
- Premises: the other facts above, rewritten as neutral prose with their assay and controls. Present
  the evidence flatly, with no "note that" and no hint of which way the answer goes.
- Derivation: one to four steps, each naming the premise that licenses it. Mark at least one step a
  decision point: a step with a plausible wrong alternative that changes the answer, and say what that
  wrong alternative is.
- Answer contract: {contract}
- Gold answer, plus the derivation written as a model answer would write it.
- Six to twenty rubric criteria with integer points, each self-contained (naming entity, direction and
  context) and gradeable from a response alone.

Return one JSON object and nothing else, with the keys: premises, question, withheld_target, steps,
answer_contract, gold, criteria."""


def critic_user(task: ExampleTask) -> str:
    """S5: the drafted task as the solver sees it, with gold and rubric, for review."""
    return f"""# The question, as the solver sees it

{task.question}

# Reference answer

{json.dumps(task.gold["answer"], indent=2, ensure_ascii=False)}

Reference derivation: {task.gold["rationale"]}

# Rubric

{rubric_block(task)}

# What to do

Review this task in four categories and report every defect you find, each with the text that shows it:

1. Scientific validity: is the reference answer what the premises support, and is any step wrong?
2. Question-rubric consistency: does every criterion reward only what the question asks, and is any
   criterion unanswerable from a response alone, redundant, or a restatement of a premise?
3. Derivability and difficulty: is each step licensed by a stated premise or by standard biology, and
   does the question have a decision point that a careless reader would get wrong?
4. Leakage and contrivance: does the question give away its own answer, and would a domain reader call
   the setup contrived?

Return one JSON object and nothing else: {{"findings": [{{"category": 1-4, "severity":
"blocker" | "fix" | "note", "what": "...", "evidence": "<quoted text>", "fix": "..."}}],
"verdict": "accept" | "fix" | "reject"}}"""


def adversary_user(question: str, gold_answer: dict, gold_rationale: str) -> str:
    """S6: question + gold -> the strongest alternative answer, and one must-fail answer."""
    return f"""# The question

{question}

# The reference answer

{json.dumps(gold_answer, indent=2, ensure_ascii=False)}

Reference derivation: {gold_rationale}

# What to write

1. The strongest alternative answer the premises could support, with the derivation that argues for it
   from the premises alone. Make it the best case a careful reader could make, not a straw man. If the
   premises really do force the reference answer, say so and explain which premise closes the door.
2. One answer that a grader must score as wrong: the answer reached by taking the wrong fork at the
   question's decision point, written as convincingly as its reasoning allows.

Return one JSON object and nothing else: {{"alternative": {{"answer": ..., "derivation": "...",
"which_premise_is_decisive": "..."}}, "must_fail": {{"answer": ..., "derivation": "...",
"fork_taken": "..."}}}}"""


def entailment_user(src: Sources, paper_id: str, candidate: str) -> str:
    """S0 entailment gate: does the paper's own title and abstract give the candidate gold away?"""
    text = src.texts[paper_id]
    return f"""# Title and abstract of one paper

{text.title}

{text.abstract}

# Candidate finding, withheld from a question built on this paper

{candidate}

# What to decide

Does the title or abstract above state this finding, or entail it closely enough that a reader who had
seen only them could give it as an answer? Paraphrase counts. A general statement that commits to the
same direction for this readout counts. A statement that merely names the molecules, the model or the
assay does not.

Answer `entailed` or `not_entailed`, with one line saying which sentence decided it."""


def licensing_user(premise: str) -> str:
    """V3 licensing vote: is this premise standard biology, independent of any one study?"""
    return f"""# Premise

{premise}

# What to decide

Is this true in general biology, independent of any single study -- the kind of statement a graduate
textbook or a review would support -- or is it a finding that would have to be taken from one paper's
own results?

Answer `general` or `study_specific`, with one line of reason. Judge the statement as written,
including its direction and its scope."""


def preference_user(question: str, derivation_x: str, derivation_y: str) -> str:
    """V4 uniqueness vote: which derivation do the premises better support, X, Y or neither?"""
    return f"""# The question and its premises

{question}

# Derivation X

{derivation_x}

# Derivation Y

{derivation_y}

# What to decide

Which derivation do the premises better support -- X, Y, or neither? You are not told which, if
either, is the reference answer; judge only how well each follows from the premises above.

Answer `X`, `Y` or `neither`, with one line naming the premise that decided it."""


def judge_user(src: Sources, task: ExampleTask, criterion_id: str, ablation: str) -> str:
    """The judge's criterion check, exactly as grade.py sends it (docs/task_format.md §6)."""
    envelope = task.ablations[ablation]
    rationale, _ = grade.truncate_words(envelope["rationale"], task.rubric["rationale_word_cap"])
    response = grade.build_response_text(envelope["answer"], rationale)
    criterion = task.criterion(criterion_id)
    return _fill(
        src.judge_prompts["criterion_check"],
        question=task.question,
        response=response,
        criterion_id=criterion["id"],
        criterion_text=criterion["text"],
    )


def _fill(template: str, **values: str) -> str:
    """`{{KEY}}` substitution, the same rule grade.py applies to judge_prompt.md."""
    for key, value in values.items():
        template = template.replace("{{" + key.upper() + "}}", value)
    return template


# ── the 20 pilot prompts ──────────────────────────────────────────────────────────────────────────


def build_prompts(src: Sources) -> list[Prompt]:
    """The 20 prompts: five papers, seven roles, every request shaped as its M1 stage sends it."""
    task_f2, task_f4, task_f5 = src.tasks[TASK_F2], src.tasks[TASK_F4], src.tasks[TASK_F5]
    return [
        # S1, Fact Verifier (role `generator`).
        Prompt(
            id="s1-facts-p1",
            role="generator",
            paper_id=P1,
            summary="S1: verify 4 inhibitor-arm facts (Akt and MEK/ERK branch inhibition in Api5-OE "
            "spheroids) against the Ras/MAPK/ERK Results subsection",
            system=S1_SYSTEM,
            user=fact_check_user(src, P1, "Ras/MAPK/ERK signalling regulates", [5, 6, 7, 8]),
        ),
        Prompt(
            id="s1-facts-p2",
            role="generator",
            paper_id=P2,
            summary="S1: verify 5 CD22 phagocytosis facts (CD22 OE, binding-deficient and ITIM mutants, "
            "HEK reconstitution) against the ITIM-ITAM Results subsection",
            system=S1_SYSTEM,
            user=fact_check_user(src, P2, "ITIM", [2, 4, 7, 8, 9]),
        ),
        # S2, Task Architect (role `generator`).
        Prompt(
            id="s2-author-p3-bwd",
            role="generator",
            paper_id=P3,
            summary="S2: author an F4 backward question from the three PRMT7 perturbations (knockout, "
            "catalytic inhibitor, PROTAC) on activated CD8+ T cells",
            system=S2_SYSTEM,
            user=authoring_user(
                src,
                P3,
                [10, 12, 16, 18],
                family="F4 (inverse inference)",
                direction="backward",
                contract='{"choice": "<one of the paper\'s own perturbations>"} plus a rationale, '
                "the options being the three PRMT7 perturbations above",
                withheld="which perturbation produced the fall in the CD8+ fraction reported in f2",
            ),
        ),
        Prompt(
            id="s2-author-p4-fwd",
            role="generator",
            paper_id=P4,
            summary="S2: author an F2 forward question from the zebrafish kidney stimuli (LPS, zymosan, "
            "poly I:C, cytokine peptides, CXCL11 with and without Fgfr1 blockade)",
            system=S2_SYSTEM,
            user=authoring_user(
                src,
                P4,
                [8, 11, 12, 13, 14, 15, 16],
                family="F2 (paired control)",
                direction="forward",
                contract='{"arms": {"<arm>": "aggregation | no aggregation"}} plus a rationale, one '
                "arm per stimulus named in the premises",
                withheld="the outcome of the Fgfr1-antagonist arm in f6 and of the cytokine-peptide "
                "arm in f4",
            ),
        ),
        Prompt(
            id="s2-author-p5-fwd",
            role="generator",
            paper_id=P5,
            summary="S2: author an F2 forward question from the IL-1 + γc co-stimulation arms and the "
            "MEK, IRAK4, JAK3 and STAT5 inhibitor arms in neutrophils",
            system=S2_SYSTEM,
            user=authoring_user(
                src,
                P5,
                [1, 3, 5, 7, 9, 11],
                family="F2 (paired control)",
                direction="forward",
                contract='{"arms": {"<arm>": "abolished | reduced | unchanged"}} plus a rationale, one '
                "arm per inhibitor or receptor-deficiency arm named in the premises",
                withheld="the effect of the MEK inhibitor arm in f5 on co-stimulated cytokine output",
            ),
        ),
        # S5, Critic.
        Prompt(
            id="s5-critic-f2",
            role="critic",
            paper_id=P1,
            summary="S5: review the hand-built F2 forward task (cross-pathway phospho readouts in "
            "Api5-OE spheroids) with its gold and rubric",
            system=CRITIC_SYSTEM,
            user=critic_user(task_f2),
        ),
        Prompt(
            id="s5-critic-f5",
            role="critic",
            paper_id=P3,
            summary="S5: review the hand-built F5 mechanism task (non-catalytic PRMT7 restraint of "
            "NF-kB) with its gold and rubric",
            system=CRITIC_SYSTEM,
            user=critic_user(task_f5),
        ),
        # S6, Adversary.
        Prompt(
            id="s6-adversary-f2",
            role="adversary",
            paper_id=P1,
            summary="S6: strongest alternative to the F2 gold (the two cross-pathway readouts unchanged)",
            system=ADVERSARY_SYSTEM,
            user=adversary_user(task_f2.question, task_f2.gold["answer"], task_f2.gold["rationale"]),
        ),
        Prompt(
            id="s6-adversary-f4",
            role="adversary",
            paper_id=P3,
            summary="S6: strongest alternative to the F4 gold (the catalytic inhibitor is the arm that "
            "lowered the CD8+ fraction)",
            system=ADVERSARY_SYSTEM,
            user=adversary_user(task_f4.question, task_f4.gold["answer"], task_f4.gold["rationale"]),
        ),
        Prompt(
            id="s6-adversary-p2",
            role="adversary",
            paper_id=P2,
            summary="S6: strongest alternative for a CD22 reconstitution question (which co-expressed "
            "CD22 variant leaves Dectin-1 uptake intact)",
            system=ADVERSARY_SYSTEM,
            user=adversary_user(
                "Dectin-1 (an ITAM-bearing phagocytic receptor) is expressed in HEK293T cells, which then "
                "take up zymosan bioparticles. Three CD22 constructs are co-expressed with Dectin-1 in "
                "separate wells: wild-type CD22, CD22 R130A (its sialic-acid-binding site mutated), and CD22 "
                "with its three ITIM tyrosines mutated to phenylalanine. CD22 and Dectin-1 are confirmed to "
                "sit in molecular proximity for each construct. Premises: in microglia, CD22 suppresses "
                "uptake and the R130A mutant suppresses it as strongly as wild type, so the suppression does "
                "not need sialic-acid binding; CD22 carries an ITIM motif, which recruits SHP-1 when it is "
                "engaged.\n\nQuestion: of the three co-expressed constructs, which leaves zymosan uptake at "
                "the level Dectin-1 alone gives, and which suppress it?",
                {
                    "arms": {
                        "cd22_wt": "suppressed",
                        "cd22_r130a": "suppressed",
                        "cd22_itim_mutant": "uptake retained",
                    }
                },
                "Suppression survives loss of sialic-acid binding, so the ligand-binding site is not the "
                "part that acts; what CD22 contributes in cis is its ITIM, whose phosphotyrosines recruit "
                "SHP-1 to dephosphorylate the ITAM-bearing receptor beside it. Mutating the three ITIM "
                "tyrosines removes that recruitment while leaving the protein in proximity to Dectin-1, so "
                "that arm alone keeps uptake; wild-type and R130A both suppress it.",
            ),
        ),
        # Checker votes: S0 entailment gate, V3 licensing, V4 uniqueness.
        Prompt(
            id="check-entail-p5",
            role="checker",
            paper_id=P5,
            summary="S0 entailment gate: does the γc paper's title and abstract give away that MEK "
            "inhibition suppresses co-stimulated IL-6 and TNFα output?",
            system=CHECKER_SYSTEM,
            user=entailment_user(
                src,
                P5,
                "In IL-1α + IL-15 co-stimulated neutrophils, the MEK inhibitor trametinib suppresses both "
                "IL-6 and TNFα secretion substantially and across its full dose range.",
            ),
            schema=vote_schema(["entailed", "not_entailed"]),
        ),
        Prompt(
            id="check-license-p4",
            role="checker",
            paper_id=P4,
            summary="V3 licensing: is 'a chemokine can act on a tissue-resident progenitor through an "
            "induced FGF, so FGFR blockade removes the response' general biology?",
            system=CHECKER_SYSTEM,
            user=licensing_user(
                "A chemokine released at a site of tissue damage can act on resident progenitor cells "
                "indirectly, by inducing fibroblast growth factors that signal through FGFR1 on those "
                "progenitors, so blocking FGFR1 removes the progenitor response to that chemokine without "
                "removing the chemokine."
            ),
            schema=vote_schema(["general", "study_specific"]),
        ),
        Prompt(
            id="check-prefer-f4",
            role="checker",
            paper_id=P3,
            summary="V4 uniqueness: which derivation do the F4 premises better support, the gold one or "
            "the wrong-fork one that groups the two small molecules",
            system=CHECKER_SYSTEM,
            user=preference_user(
                task_f4.question,
                task_f4.gold["rationale"],
                src.tasks[TASK_F4].ablations["wrong_core_dp_fork.json"]["rationale"],
            ),
            schema=vote_schema(["X", "Y", "neither"]),
        ),
        # Panel solves: the three hand-built instructions, exactly as the solver receives them.
        Prompt(
            id="solve-f2",
            role="panel",
            paper_id=P1,
            summary="Solve the F2 forward task: the four phospho readouts across the two single-branch "
            "inhibitor arms",
            user=task_f2.instruction,
            schema=families.answer_schema(task_f2.rubric["family"], task_f2.rubric["core"]),
        ),
        Prompt(
            id="solve-f4",
            role="panel",
            paper_id=P3,
            summary="Solve the F4 backward task: which of the three PRMT7 perturbations lowered the CD8+ "
            "fraction",
            user=task_f4.instruction,
            schema=families.answer_schema(task_f4.rubric["family"], task_f4.rubric["core"]),
        ),
        Prompt(
            id="solve-f5",
            role="panel",
            paper_id=P3,
            summary="Solve the F5 mechanism task: why the three PRMT7 arms split, in one mechanism "
            "sentence plus key entities",
            user=task_f5.instruction,
            schema=families.answer_schema(task_f5.rubric["family"], task_f5.rubric["core"]),
        ),
        # Judge criterion checks, in judge_prompt.md's own format.
        Prompt(
            id="judge-f2-c2",
            role="judge",
            paper_id=P1,
            summary="Judge criterion c2 (parallel branches) on the F2 paraphrase answer",
            system=src.judge_prompts["system"],
            user=judge_user(src, task_f2, "c2", "gold_paraphrase.json"),
            schema=grade.VERDICT_SCHEMA,
        ),
        Prompt(
            id="judge-f5-k2",
            role="judge",
            paper_id=P3,
            summary="Judge core criterion k2 (stoichiometric restraint of p65) on the F5 "
            "correct-core-wrong-mechanism answer",
            system=src.judge_prompts["system"],
            user=judge_user(src, task_f5, "k2", "correct_core_wrong_mechanism.json"),
            schema=grade.VERDICT_SCHEMA,
        ),
        Prompt(
            id="regrade-f4-d4",
            role="judge_regrade",
            paper_id=P3,
            summary="Regrade criterion d4 (catalysis is not the brake) on the F4 textbook-prior answer",
            system=src.judge_prompts["system"],
            user=judge_user(src, task_f4, "d4", "textbook_prior.json"),
            schema=grade.VERDICT_SCHEMA,
        ),
        Prompt(
            id="regrade-f5-e2",
            role="judge_regrade",
            paper_id=P3,
            summary="Regrade criterion e2 (target engagement rules out a failed inhibitor) on the F5 "
            "hedge-everything answer",
            system=src.judge_prompts["system"],
            user=judge_user(src, task_f5, "e2", "hedge_everything.json"),
            schema=grade.VERDICT_SCHEMA,
        ),
    ]


# ── running ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class CallResult:
    """One (prompt, model, sample) trial. `status` is what the probe counts."""

    prompt_id: str
    role: str
    model: str
    sample: int
    status: str  # ok | refusal | empty | error
    refusal_category: str | None = None
    stop_reason: str | None = None
    parsed_ok: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    note: str = ""
    # The answer itself, kept in responses.jsonl: a model that declines in prose while the API
    # reports a normal stop is not a refusal by `llm.complete`, and only the text shows it.
    text: str = ""


def role_models(roster: dict[str, RoleSpec | list[RoleSpec] | None], role: str) -> list[RoleSpec]:
    """The models the roster assigns to `role`; a list role keeps its order."""
    spec = roster[role]
    if spec is None:
        raise ValueError(f"role {role!r} has no model in the roster")
    return list(spec) if isinstance(spec, list) else [spec]


def run_trial(prompt: Prompt, spec: RoleSpec, sample: int, ledger: Ledger) -> CallResult:
    """One call. A refusal, an empty answer and an error are all recorded, never retried here."""
    try:
        out = complete(
            spec,
            prompt.user,
            role=prompt.role,
            system=prompt.system,
            schema=prompt.schema,
            ledger=ledger,
            tag=prompt.id,
        )
    except BudgetExceeded as exc:
        return CallResult(prompt.id, prompt.role, spec.model, sample, "error", note=f"budget: {exc}")
    except LLMError as exc:
        return CallResult(prompt.id, prompt.role, spec.model, sample, "error", note=str(exc)[:300])

    if out.refusal:
        status = "refusal"
    elif not out.text.strip():
        status = "empty"
    else:
        status = "ok"
    return CallResult(
        prompt_id=prompt.id,
        role=prompt.role,
        model=spec.model,
        sample=sample,
        status=status,
        refusal_category=out.refusal_category,
        stop_reason=out.stop_reason,
        parsed_ok=None if prompt.schema is None else out.parsed is not None,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
        cost_usd=out.cost_usd,
        latency_s=out.latency_s,
        note="" if out.model == spec.model else f"answered by {out.model}",
        text=out.text,
    )


def run_probe(
    prompts: list[Prompt],
    roster: dict[str, RoleSpec | list[RoleSpec] | None],
    ledger: Ledger,
    *,
    samples: int,
    concurrency: int,
) -> list[CallResult]:
    """Every (prompt, model the role is assigned, sample), bounded-concurrently."""
    trials = [
        (prompt, spec, sample)
        for prompt in prompts
        for spec in role_models(roster, prompt.role)
        for sample in range(samples)
    ]
    print(f"{len(trials)} trials: {len(prompts)} prompts x their role's models x {samples} samples")
    results: list[CallResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_trial, *trial, ledger) for trial in trials]
        for done, future in enumerate(futures, start=1):
            result = future.result()
            results.append(result)
            detail = " ".join(filter(None, [result.status, result.refusal_category, result.note]))
            print(
                f"[{done}/{len(trials)}] {result.role:14s} {result.model:18s} "
                f"{result.prompt_id:20s} {detail}",
                flush=True,
            )
    return results


# ── reporting ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Tally:
    """Trial counts for one (role, model) cell. The rate excludes errors (PLAN §3: only transient
    errors make a trial invalid; refusals are valid 0s and reported as a rate)."""

    calls: int = 0
    refusals: int = 0
    empty: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    categories: Counter = field(default_factory=Counter)

    @property
    def valid(self) -> int:
        return self.calls - self.errors

    @property
    def refusal_rate(self) -> float | None:
        return self.refusals / self.valid if self.valid else None


def tally(results: list[CallResult], key: Callable[[CallResult], Hashable]) -> dict[Hashable, Tally]:
    """Tallies keyed by `key(result)`, in first-seen order."""
    cells: dict[Hashable, Tally] = {}
    for result in results:
        cell = cells.setdefault(key(result), Tally())
        cell.calls += 1
        cell.refusals += result.status == "refusal"
        cell.empty += result.status == "empty"
        cell.errors += result.status == "error"
        if result.cost_usd is None:
            cell.unpriced_calls += 1
        else:
            cell.cost_usd += result.cost_usd
        if result.refusal_category:
            cell.categories[result.refusal_category] += 1
    return cells


def rate_text(cell: Tally) -> str:
    rate = cell.refusal_rate
    return "n/a" if rate is None else f"{rate:.0%}"


def summary_markdown(results: list[CallResult], prompts: list[Prompt], threshold: float) -> str:
    """The two tables docs/m0_refusal_probe.md reports, regenerated on every run."""
    by_cell = tally(results, lambda r: (r.role, r.model))
    by_role = tally(results, lambda r: r.role)
    lines = [
        "| role | model | trials | refusals | empty | errors | refusal rate | cost |",
        "|---" * 8 + "|",
    ]
    for (role, model), cell in by_cell.items():
        cost = (
            f"${cell.cost_usd:.3f}"
            if cell.unpriced_calls == 0
            else f"${cell.cost_usd:.3f} + {cell.unpriced_calls} unpriced"
        )
        lines.append(
            f"| `{role}` | `{model}` | {cell.calls} | {cell.refusals} | {cell.empty} | {cell.errors} "
            f"| {rate_text(cell)} | {cost} |"
        )
    lines += [
        "",
        "| role | prompts | trials | refusals | refusal rate | over the 10% rule? |",
        "|---|---|---|---|---|---|",
    ]
    prompt_counts = Counter(p.role for p in prompts)
    for role, cell in by_role.items():
        rate = cell.refusal_rate
        verdict = "n/a" if rate is None else ("**yes — swap**" if rate > threshold else "no")
        lines.append(
            f"| `{role}` | {prompt_counts[role]} | {cell.calls} | {cell.refusals} | {rate_text(cell)} "
            f"| {verdict} |"
        )
    categories = Counter()
    for cell in by_cell.values():
        categories += cell.categories
    lines += ["", f"Refusal categories: {dict(categories) or 'none'}"]
    priced = sum(c.cost_usd for c in by_cell.values())
    unpriced = sum(c.unpriced_calls for c in by_cell.values())
    lines.append(
        f"Cost: ${priced:.2f} over {len(results) - unpriced} priced trials, {unpriced} unpriced trials."
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path, prompts: list[Prompt], results: list[CallResult], meta: dict, threshold: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(
        json.dumps([{**p.meta(), "system": p.system, "user": p.user} for p in prompts], indent=2),
        encoding="utf-8",
    )
    trials = [asdict(r) for r in results]
    (out_dir / "responses.jsonl").write_text(
        "".join(
            json.dumps(
                {k: trial[k] for k in ("prompt_id", "role", "model", "sample", "status", "text")},
                ensure_ascii=False,
            )
            + "\n"
            for trial in trials
        ),
        encoding="utf-8",
    )
    (out_dir / "results.json").write_text(
        json.dumps({**meta, "trials": [{**t, "text": ""} for t in trials]}, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(summary_markdown(results, prompts, threshold), encoding="utf-8")


# ── cli ───────────────────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=2, help="samples per (prompt, model); default 2")
    parser.add_argument("--budget-usd", type=float, default=5.0, help="ledger spend cap; default 5")
    parser.add_argument("--concurrency", type=int, default=6, help="calls in flight; default 6")
    parser.add_argument("--threshold", type=float, default=0.10, help="the PLAN §3 swap rule; default 0.10")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default {DEFAULT_OUT}/<timestamp>")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run these prompt ids only")
    parser.add_argument("--dry-run", action="store_true", help="render and list the prompts, call nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = load_sources()
    prompts = build_prompts(src)
    if args.only:
        prompts = [p for p in prompts if p.id in args.only]
        if unknown := set(args.only) - {p.id for p in prompts}:
            raise SystemExit(f"no such prompt id(s): {sorted(unknown)}")
    roster = load_roster(
        OmegaConf.to_container(OmegaConf.load(CONFIGS / "models" / "default.yaml"), resolve=True)
    )

    papers, roles = {p.paper_id for p in prompts}, {p.role for p in prompts}
    print(f"{len(prompts)} prompts over {len(papers)} papers and {len(roles)} roles")
    for prompt in prompts:
        models = ", ".join(spec.model for spec in role_models(roster, prompt.role))
        print(f"  {prompt.id:20s} {prompt.role:14s} {prompt.paper_id} {prompt.words:6d}w  -> {models}")
    if args.dry_run:
        return 0

    out_dir = args.out_dir or DEFAULT_OUT / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out_dir / "calls.jsonl", budget_usd=args.budget_usd)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    results = run_probe(prompts, roster, ledger, samples=args.samples, concurrency=args.concurrency)
    meta = {
        "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "samples": args.samples,
        "budget_usd": args.budget_usd,
        "threshold": args.threshold,
        "ledger_spent_usd": round(ledger.spent(), 4),
        "prompts": [p.meta() for p in prompts],
    }
    write_outputs(out_dir, prompts, results, meta, args.threshold)
    print()
    print(summary_markdown(results, prompts, args.threshold))
    print(f"wrote {out_dir}/{{calls.jsonl, prompts.json, responses.jsonl, results.json, summary.md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
