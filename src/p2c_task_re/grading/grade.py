"""Grade one task answer against its rubric — docs/task_format.md §4–§7.

This file is standalone by contract. Every packaged task gets a verbatim copy of it and of
`judge_prompt.md` in its `tests/` folder, and the verifier runs it with a bare `python3` that has
only the standard library plus the `anthropic` / `openai` SDKs. So: no imports from `p2c_task_re`,
and the SDKs are imported inside the judge classes, so an answer that needs no judge call (a missing
or unparseable answer) grades with no SDK and no API key present.

Two places where §5's "a missing, extra or unrecognised value counts as wrong" needs an operational
reading, because the family `s_core` formulas are recall-shaped and cannot see an extra key:

* **missing** — a gold marker/arm the answer does not mention is scored as a wrong entry.
* **extra or unrecognised** — an answer that names a marker, arm or option the task never defined, or
  that carries a top-level field the family does not define, is off-contract rather than merely wrong,
  so its core score is 0, as §5 already prescribes for an invalid F3 ranking.

Evidence quotes are matched after collapsing whitespace and case-folding; everything else about the
check is strict, so an invented or edited quote still fails it. Whether a quote only echoes the
question is decided word token by word token, wherever the quote occurs in the response, so the
punctuation at a quote's edge cannot move it out of a masked span.

Two more operational readings, both recorded in grading.json so their rate can be measured:

* **a fenced answer** — a reply that is one JSON object wrapped in a single markdown code fence is a
  formatting slip, not a scientific failure, so the enclosing fence is stripped before parsing and
  `answer_format` says whether the answer arrived `bare`, `fenced` or `invalid`. Nothing else is
  extracted: prose around the object, several objects, or a fence over part of the reply stay invalid.
* **judge spend** — judging is most of the cost of a run, so every real judge takes an `on_usage`
  callback and whoever builds the judges totals it (`JudgeUsage`): the CLI writes the totals to
  grading.json as `judge_usage`, and the in-process runner also turns each call into a ledger line.

CLI (§7):

    python3 grade.py --tests-dir DIR --answer FILE --out-dir DIR [--judge-model M] [--regrade-model M]
    python3 grade.py --check --tests-dir DIR

Exit codes: 0 = rewards written, 2 = judge failure (no reward.json), 1 = bad tests dir.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

GRADE_VERSION = "1.2"

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
DEFAULT_REGRADE_MODEL = "gpt-6.1-sol"
DEFAULT_SAMPLES = 3
DEFAULT_RETRIES = 2
DEFAULT_CONCURRENCY = 8

# One binary call per criterion over a few hundred words. The model vendors put plain
# classification at `low`, but a criterion check has to weigh equivalent phrasing against hedging and
# then copy an exact quote, which `low` tends to skip; `high` multiplies thinking tokens across
# (criteria x samples) calls per answer for no gain we can measure. `medium` is the default, and
# JUDGE_EFFORT overrides it.
DEFAULT_JUDGE_EFFORT = "medium"

MASK_MIN_TOKENS = 10
JUDGE_MAX_TOKENS = 4000

FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
CORE_TYPE_BY_FAMILY = {
    "F1": "marker_directions",
    "F2": "arm_enum",
    "F3": "ranking",
    "F4": "choice",
    "F5": "judged",
    "F6": "arm_enum",
    "F7": "judged",
}
# The `answer` fields each family defines (docs/task_format.md §5).
REQUIRED_ANSWER_FIELDS = {
    "F1": ("markers",),
    "F2": ("arms",),
    "F3": ("ranking",),
    "F4": ("choice",),
    "F5": ("mechanism_primary", "key_entities"),
    "F6": ("arms",),
    "F7": ("experiment",),
}
OPTIONAL_ANSWER_FIELDS = {"F6": ("state",)}
CAPABILITIES = ("direction", "comparison", "control-logic", "mechanism", "design", "uncertainty")
GROUNDING_SOURCES = ("paper", "instruction", "k_premise")

# The JSON the judge returns, for both checks.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}


class TestsDirError(Exception):
    """The tests dir is unusable: a missing file, bad JSON, or an invalid rubric. Exit 1."""


class GradingError(Exception):
    """A criterion never collected `samples` valid votes. Exit 2, and no reward.json."""


class Judge(Protocol):
    """Anything that answers a judge prompt. `complete` returns None for a refusal."""

    name: str

    def complete(self, system: str, user: str) -> str | None: ...


# What a real judge reports after every API call: (model, input_tokens, output_tokens, refusal). A call
# that raised reports nothing, since it has no usage to report.
UsageCallback = Callable[[str, int, int, bool], None]


class JudgeUsage:
    """Judge API usage totalled per model — grading.json's `judge_usage` (docs/task_format.md §6).

    `add` is the `on_usage` callback the real judges take; judge calls run concurrently, so it locks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, input_tokens: int, output_tokens: int, refusal: bool = False) -> None:
        with self._lock:
            counts = self._per_model.setdefault(
                model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusals": 0}
            )
            counts["calls"] += 1
            counts["input_tokens"] += input_tokens
            counts["output_tokens"] += output_tokens
            counts["refusals"] += int(refusal)

    def per_model(self) -> dict[str, dict[str, int]]:
        """A snapshot: model id -> {calls, input_tokens, output_tokens, refusals}."""
        with self._lock:
            return {model: dict(counts) for model, counts in sorted(self._per_model.items())}

    def calls(self) -> int:
        with self._lock:
            return sum(counts["calls"] for counts in self._per_model.values())


# --------------------------------------------------------------------------------------- text utils


def fold(value: object) -> str:
    """Case-fold and collapse whitespace — the §5 comparison for keys and enum values."""
    return " ".join(str(value).split()).casefold()


def provider_of(model: str) -> str | None:
    """Which API serves a model, from its name (§7): `claude-*` → anthropic, `gpt-*` / `o<digit>*` →
    openai; None for any other name."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-") or re.fullmatch(r"o\d.*", model):
        return "openai"
    return None


def truncate_words(text: str, cap: int) -> tuple[str, bool]:
    """Cut `text` to `cap` whitespace-separated words (§6 step 2). Returns (text, truncated)."""
    words = text.split()
    if len(words) <= cap:
        return text, False
    return " ".join(words[:cap]), True


def truncate_answer(answer: Any, cap: int) -> tuple[Any, bool]:
    """The answer object as the judge sees it: its strings, in order, share `cap` words (§6 step 2).

    Free-text fields (`mechanism_primary`, `experiment`, `state`, ...) would otherwise carry any amount
    of text past the rationale cap. Code-scored fields hold a word or two per value, so a valid answer
    is never cut; the core is scored on the full answer either way.
    """
    left = cap

    def cut(value: Any) -> Any:
        nonlocal left
        if isinstance(value, str):
            text, _ = truncate_words(value, left)
            left = max(0, left - len(value.split()))
            return text
        if isinstance(value, list):
            return [cut(item) for item in value]
        if isinstance(value, dict):
            return {key: cut(item) for key, item in value.items()}
        return value

    shown = cut(answer)
    return shown, shown != answer


def build_response_text(answer: dict[str, Any], rationale: str) -> str:
    """The exact text the judge is shown for this answer (§6 step 3)."""
    return json.dumps(answer, indent=2, ensure_ascii=False) + "\n\nRationale:\n" + rationale


_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _mask(text: str, question: str, min_tokens: int) -> tuple[list[re.Match[str]], list[bool]]:
    """The word tokens of `text`, each flagged when it lies in a run of >= `min_tokens` tokens that
    also occurs in `question` (§6 step 4)."""
    tokens = list(_WORD.finditer(text))
    words = [m.group(0).lower() for m in tokens]
    question_words = _words(question)
    grams = {tuple(question_words[i : i + min_tokens]) for i in range(len(question_words) - min_tokens + 1)}
    masked = [False] * len(tokens)
    for i in range(len(tokens) - min_tokens + 1):
        if tuple(words[i : i + min_tokens]) in grams:
            masked[i : i + min_tokens] = [True] * min_tokens
    return tokens, masked


def masked_spans(text: str, question: str, min_tokens: int = MASK_MIN_TOKENS) -> list[str]:
    """Runs of >= `min_tokens` word tokens that `text` shares with question.md (§6 step 4).

    Returns the maximal shared runs as slices of `text` — the response text, when grading. A judge
    evidence quote that lies wholly inside them is the solver echoing the question back, and earns
    nothing.
    """
    tokens, masked = _mask(text, question, min_tokens)
    spans: list[str] = []
    start: int | None = None
    for i, flag in enumerate([*masked, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append(text[tokens[start].start() : tokens[i - 1].end()])
            start = None
    return spans


# -------------------------------------------------------------------------------------- core scores


@dataclass
class CoreResult:
    """`s_core` and the per-field record that goes into grading.json as `core_detail`."""

    score: float
    detail: list[dict[str, Any]]


def _field(name: str, gold: Any, got: Any, correct: bool, critical: bool = False, note: str = "") -> dict:
    entry = {"field": name, "gold": gold, "answer": got, "correct": correct}
    if critical:
        entry["critical"] = True
    if note:
        entry["note"] = note
    return entry


def _invalid(detail: list[dict[str, Any]], note: str) -> CoreResult:
    return CoreResult(0.0, detail + [_field("<answer>", None, None, False, note=note)])


def _folded_map(value: Any) -> dict[str, str] | None:
    """A {key: value} answer field, folded on both sides; None when it is not an object of strings or
    two of its keys fold to the same one (the answer would then say two things about one entry)."""
    if not isinstance(value, dict):
        return None
    out: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(val, str) or fold(key) in out:
            return None
        out[fold(key)] = fold(val)
    return out


def _enum_map_detail(
    core: dict, gold_answer: dict, answer: dict, field: str, values: str
) -> tuple[list[dict[str, Any]], str | None]:
    """F1's markers and F2/F6's arms: one detail entry per entry `core[field]` declares, plus a note
    when the answer is off-contract (not an object of strings, or naming an entry the task does not
    declare)."""
    allowed = {fold(v) for v in core[values]}
    critical = {fold(name) for name in core.get("critical", ())}
    gold_map = _folded_map(gold_answer.get(field)) or {}
    got_map = _folded_map(answer.get(field))
    if got_map is None:
        return [], f"`{field}` is not an object of strings with distinct keys"

    detail = []
    for name in core[field]:
        key = fold(name)
        gold_value, got_value = gold_map.get(key), got_map.get(key)
        correct = got_value is not None and got_value in allowed and got_value == gold_value
        detail.append(_field(name, gold_value, got_value, correct, key in critical))
    unknown = sorted(set(got_map) - {fold(name) for name in core[field]})
    return detail, f"{field} the task does not define: {', '.join(unknown)}" if unknown else None


def s_core_marker_directions(core: dict, gold_answer: dict, answer: dict) -> CoreResult:
    """F1: mean over the direction classes present in gold of that class's recall (§5)."""
    detail, off_contract = _enum_map_detail(core, gold_answer, answer, "markers", "classes")
    if off_contract:
        return _invalid(detail, off_contract)
    if any(entry.get("critical") and not entry["correct"] for entry in detail):
        return CoreResult(0.0, detail)

    recalls = []
    for class_name in sorted({e["gold"] for e in detail if e["gold"] is not None}):
        members = [e for e in detail if e["gold"] == class_name]
        recalls.append(sum(1 for e in members if e["correct"]) / len(members))
    return CoreResult(sum(recalls) / len(recalls) if recalls else 0.0, detail)


def s_core_arm_enum(core: dict, gold_answer: dict, answer: dict) -> CoreResult:
    """F2 and F6: the fraction of gold arms whose enum value matches (§5)."""
    detail, off_contract = _enum_map_detail(core, gold_answer, answer, "arms", "values")
    if off_contract:
        return _invalid(detail, off_contract)
    if any(entry.get("critical") and not entry["correct"] for entry in detail):
        return CoreResult(0.0, detail)
    return CoreResult(sum(1 for e in detail if e["correct"]) / len(detail), detail)


def _tie_sets(ranking: Any, arms: Sequence[str]) -> list[set[str]] | None:
    """Validate an F3 ranking and fold it: every declared arm exactly once, no unknown arms."""
    if not isinstance(ranking, list) or not ranking:
        return None
    expected = {fold(a) for a in arms}
    seen: set[str] = set()
    out: list[set[str]] = []
    for group in ranking:
        if not isinstance(group, list) or not group:
            return None
        folded = set()
        for arm in group:
            if not isinstance(arm, str):
                return None
            key = fold(arm)
            if key in seen or key not in expected:
                return None
            seen.add(key)
            folded.add(key)
        out.append(folded)
    return out if seen == expected else None


def s_core_ranking(core: dict, gold_answer: dict, answer: dict) -> CoreResult:
    """F3: the fraction of gold-comparable arm pairs the answer orders the same way (§5)."""
    arms = core["arms"]
    gold_sets = _tie_sets(gold_answer.get("ranking"), arms)
    got_sets = _tie_sets(answer.get("ranking"), arms)
    if gold_sets is None:
        raise TestsDirError("gold.json holds an invalid F3 ranking")
    if got_sets is None:
        return _invalid([], "`ranking` is not a valid ordering of the task's arms")

    gold_rank = {arm: i for i, group in enumerate(gold_sets) for arm in group}
    got_rank = {arm: i for i, group in enumerate(got_sets) for arm in group}
    detail = [
        _field(
            f"rank {i + 1}",
            sorted(gold_sets[i]),
            sorted(got_sets[i]) if i < len(got_sets) else None,
            i < len(got_sets) and got_sets[i] == gold_sets[i],
            critical=bool(core.get("critical_top")) and i == 0,
        )
        for i in range(len(gold_sets))
    ]
    if core.get("critical_top") and (not got_sets or got_sets[0] != gold_sets[0]):
        return CoreResult(0.0, detail)

    pairs = [
        (a, b)
        for a in gold_rank
        for b in gold_rank
        if a < b and gold_rank[a] != gold_rank[b]
    ]
    if not pairs:
        return CoreResult(0.0, detail)
    concordant = sum(
        1
        for a, b in pairs
        if got_rank[a] != got_rank[b]
        and (got_rank[a] < got_rank[b]) == (gold_rank[a] < gold_rank[b])
    )
    return CoreResult(concordant / len(pairs), detail)


def s_core_choice(core: dict, gold_answer: dict, answer: dict) -> CoreResult:
    """F4: 1 if the answer picks the gold option, else 0 (§5)."""
    options = {fold(o) for o in core["options"]}
    gold_choice = fold(gold_answer.get("choice"))
    got = answer.get("choice")
    got_choice = fold(got) if isinstance(got, str) else None
    correct = got_choice is not None and got_choice in options and got_choice == gold_choice
    return CoreResult(1.0 if correct else 0.0, [_field("choice", gold_choice, got_choice, correct, True)])


def s_core_judged(criteria: Sequence[dict], satisfied: dict[str, bool]) -> CoreResult:
    """F5 and F7: the fraction of `core: true` criteria the judge satisfied (§5)."""
    core_criteria = [c for c in criteria if c.get("core")]
    detail = [
        _field(c["id"], "satisfied", "satisfied" if satisfied.get(c["id"]) else "not satisfied",
               bool(satisfied.get(c["id"])), True)
        for c in core_criteria
    ]
    if not core_criteria:
        return CoreResult(0.0, detail)
    return CoreResult(sum(1 for c in core_criteria if satisfied.get(c["id"])) / len(core_criteria), detail)


CORE_SCORERS: dict[str, Callable[[dict, dict, dict], CoreResult]] = {
    "marker_directions": s_core_marker_directions,
    "arm_enum": s_core_arm_enum,
    "ranking": s_core_ranking,
    "choice": s_core_choice,
}


def extra_answer_fields(family: str, answer: dict) -> list[str]:
    """Top-level `answer` fields the family does not define — off-contract, so the core scores 0."""
    allowed = set(REQUIRED_ANSWER_FIELDS[family]) | set(OPTIONAL_ANSWER_FIELDS.get(family, ()))
    return sorted(set(answer) - allowed)


def envelope_problems(family: str, answer: dict) -> list[str]:
    """Fields of `answer` that the family does not define, or defines and the answer omits."""
    problems = [f"missing field `{n}`" for n in REQUIRED_ANSWER_FIELDS[family] if n not in answer]
    extra = extra_answer_fields(family, answer)
    return problems + [f"field `{n}` is not part of the {family} answer" for n in extra]


def score_core(
    rubric: dict,
    gold_answer: dict,
    answer: dict,
    satisfied: dict[str, bool] | None = None,
) -> CoreResult:
    """`s_core` for any family (§5). `satisfied` carries the judge verdicts F5/F7 score on."""
    family = rubric["family"]
    core = rubric["core"]
    extra = extra_answer_fields(family, answer)
    if extra:
        return _invalid([], f"fields the {family} answer does not define: {', '.join(extra)}")
    if core["type"] == "judged":
        return s_core_judged(rubric["criteria"], satisfied or {})
    return CORE_SCORERS[core["type"]](core, gold_answer, answer)


# ------------------------------------------------------------------------------ rubric validation


def validate_rubric(rubric: Any, question: str | None = None) -> list[str]:
    """The §4 rules, as a list of human-readable problems. Empty list = valid."""
    problems: list[str] = []
    if not isinstance(rubric, dict):
        return ["rubric.json is not a JSON object"]

    family = rubric.get("family")
    if family not in FAMILIES:
        return [f"family {family!r} is not one of {', '.join(FAMILIES)}"]
    for key in ("task_id", "direction", "tau", "w_core", "rationale_word_cap", "core", "criteria"):
        if key not in rubric:
            problems.append(f"missing key `{key}`")
    if problems:
        return problems

    direction = rubric.get("direction")
    if direction not in ("forward", "backward", None):
        problems.append(f"direction {direction!r} is not forward, backward or null")
    if family in ("F5", "F7") and direction is not None:
        problems.append(f"{family} is direction-neutral, so direction must be null")
    if family not in ("F5", "F7") and direction is None:
        problems.append(f"{family} is asked in one direction, so direction may not be null")

    tau, w_core = rubric["tau"], rubric["w_core"]
    if not _is_number(tau) or not 0 < tau <= 1:
        problems.append(f"tau {tau!r} is not in (0, 1]")
    elif not _is_number(w_core) or not 0.30 < w_core < tau:
        problems.append(f"w_core {w_core!r} breaks 0.30 < w_core < tau ({tau})")
    cap = rubric["rationale_word_cap"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        problems.append(f"rationale_word_cap {cap!r} is not a positive integer")

    problems += _core_spec_problems(family, rubric["core"], question)
    problems += _criteria_problems(family, rubric["criteria"])
    return problems


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# The two map-shaped cores: (the declared entries, their allowed values, what one entry is).
_ENUM_MAP_CORES = {
    "marker_directions": ("markers", "classes", "a marker"),
    "arm_enum": ("arms", "values", "an arm"),
}


def _core_spec_problems(family: str, core: Any, question: str | None) -> list[str]:
    """The `core` block matches the family, and names only what the instruction defines."""
    if not isinstance(core, dict):
        return ["core is not a JSON object"]
    expected = CORE_TYPE_BY_FAMILY[family]
    if core.get("type") != expected:
        return [f"core.type {core.get('type')!r} is not {expected!r} as {family} requires"]

    problems: list[str] = []
    names: list[str] = []
    if expected in _ENUM_MAP_CORES:
        field, values, noun = _ENUM_MAP_CORES[expected]
        for key in (values, field, "critical"):
            if not isinstance(core.get(key), list) or not core[key]:
                problems.append(f"core.{key} is not a non-empty list")
        if not problems:
            names = list(core[values]) + list(core[field])
            unknown = [n for n in core["critical"] if fold(n) not in {fold(x) for x in core[field]}]
            problems += [f"core.critical names {n!r}, which is not {noun}" for n in unknown]
            problems += _duplicate_problems(core, (values, field))
    elif expected == "ranking":
        if not isinstance(core.get("arms"), list) or len(core.get("arms", ())) < 2:
            problems.append("core.arms needs at least two arms")
        else:
            names = list(core["arms"])
            problems += _duplicate_problems(core, ("arms",))
        # critical_top is F3's critical field, and PLAN §4 asks every task for at least one.
        if core.get("critical_top") is not True:
            problems.append("core.critical_top must be true: F3's top tie set is its critical field")
    elif expected == "choice":
        if not isinstance(core.get("options"), list) or len(core.get("options", ())) < 2:
            problems.append("core.options needs at least two options")
        else:
            names = list(core["options"])
            problems += _duplicate_problems(core, ("options",))

    if question is not None:
        folded = fold(question)
        problems += [
            f"core names {name!r}, which the instruction never defines"
            for name in names
            if fold(name) not in folded
        ]
    return problems


def _duplicate_problems(core: dict, keys: Sequence[str]) -> list[str]:
    """Entries of a core list that fold to the same value, so one would be counted twice."""
    return [
        f"core.{key} lists an entry twice (after case-folding)"
        for key in keys
        if len({fold(entry) for entry in core[key]}) < len(core[key])
    ]


def _criteria_problems(family: str, criteria: Any) -> list[str]:
    if not isinstance(criteria, list) or not criteria:
        return ["criteria is not a non-empty list"]

    problems: list[str] = []
    seen: set[str] = set()
    judged_points = 0
    core_count = 0
    for index, criterion in enumerate(criteria):
        where = f"criteria[{index}]"
        if not isinstance(criterion, dict):
            problems.append(f"{where} is not an object")
            continue
        cid = criterion.get("id")
        if not isinstance(cid, str) or not cid.strip():
            problems.append(f"{where} has no id")
        elif cid in seen:
            problems.append(f"duplicate criterion id {cid!r}")
        else:
            seen.add(cid)
        points = criterion.get("points")
        if not isinstance(points, int) or isinstance(points, bool) or not 0 <= points <= 10:
            problems.append(f"{where} points {points!r} is not an integer in 0..10")
            points = 0
        if not isinstance(criterion.get("text"), str) or not criterion.get("text", "").strip():
            problems.append(f"{where} has no text")
        if criterion.get("capability") not in CAPABILITIES:
            capability = criterion.get("capability")
            problems.append(f"{where} capability {capability!r} is not one of {CAPABILITIES}")
        step = criterion.get("step", None)
        if step is not None and (not isinstance(step, int) or isinstance(step, bool)):
            problems.append(f"{where} step {step!r} is not an integer or null")
        grounding = criterion.get("grounding")
        if not isinstance(grounding, dict) or grounding.get("source") not in GROUNDING_SOURCES:
            problems.append(f"{where} needs a grounding with source in {GROUNDING_SOURCES}")
        elif not (grounding.get("quote") or grounding.get("text")):
            problems.append(f"{where} grounding has no quote or text")
        if not isinstance(criterion.get("core", False), bool):
            problems.append(f"{where} core {criterion['core']!r} is not true or false")
        if criterion.get("core"):
            core_count += 1
        else:
            judged_points += points

    if judged_points <= 0:
        problems.append("the non-core criteria carry no points, so s_judge is undefined")
    if family in ("F5", "F7") and core_count == 0:
        problems.append(f"{family} needs at least one `core: true` criterion")
    if family not in ("F5", "F7") and core_count:
        problems.append(f"{family} scores its core in code, so no criterion may be `core: true`")
    return problems


# ----------------------------------------------------------------------------------- judge plumbing


@dataclass
class Vote:
    """One judge sample. `valid` false means it never produced a verdict and does not count."""

    model: str
    verdict: str | None
    evidence: str | None
    valid: bool
    reason: str | None = None
    raw_verdict: str | None = None


def prompt_path(tests_dir: Path | str | None = None) -> Path:
    """Where judge_prompt.md lives: the tests dir's copy, else the one beside this script.

    A packaged task has both in the same folder (§1). A task folder that has not been packaged yet
    has only the one beside grade.py, so `--check` and the in-process runner still work on it.
    """
    if tests_dir is not None:
        packaged = Path(tests_dir) / "judge_prompt.md"
        if packaged.exists():
            return packaged
    return Path(__file__).resolve().with_name("judge_prompt.md")


def load_prompts(path: Path | str | None = None) -> dict[str, str]:
    """Read judge_prompt.md and split it on its `##` headings."""
    path = Path(path) if path is not None else prompt_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TestsDirError(f"cannot read {path}: {exc}") from exc
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        heading = re.fullmatch(r"##\s+(system|criterion_check|consistency_check)\s*", line)
        if heading:
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = heading.group(1), []
        elif current:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    missing = [k for k in ("system", "criterion_check", "consistency_check") if not sections.get(k)]
    if missing:
        raise TestsDirError(f"{path} is missing the section(s): {', '.join(missing)}")
    return sections


def _fill(template: str, **values: str) -> str:
    """Replace the template's {{NAME}} placeholders in one pass, so a placeholder inside an inserted
    value (a solver can write "{{CRITERION_TEXT}}") stays literal text."""
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1).lower(), m.group(0)), template)


def parse_verdict(raw: str) -> tuple[str, str] | None:
    """The judge's JSON object, tolerating a markdown fence. None when it is unusable."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    verdict = fold(data.get("verdict", ""))
    evidence = data.get("evidence", "")
    if verdict not in ("yes", "no") or not isinstance(evidence, str):
        return None
    return verdict, evidence


def _evidence_checker(response_text: str, question: str) -> Callable[[str, str, str], Vote]:
    """A `yes` survives only if its quote is in the response and not wholly inside masked text.

    The mask is checked word token by word token at every place the quote's tokens occur in the
    response (its first and last token may be cut mid-word): the quote echoes the question only when
    each of those occurrences is masked throughout. So the closing period of an echoed sentence does
    not lift it out of the mask, and a short quote the solver also wrote in its own words counts.
    """
    response = fold(response_text)
    tokens, masked = _mask(response_text, question, MASK_MIN_TOKENS)
    words = [m.group(0).lower() for m in tokens]

    def occurs_at(i: int, target: list[str]) -> bool:
        if len(target) == 1:
            return target[0] in words[i]
        return (
            words[i].endswith(target[0])
            and words[i + 1 : i + len(target) - 1] == target[1:-1]
            and words[i + len(target) - 1].startswith(target[-1])
        )

    def echoes_only(quote: str) -> bool:
        target = _words(quote)
        if not target or not any(masked):
            return False
        hits = [i for i in range(len(words) - len(target) + 1) if occurs_at(i, target)]
        return bool(hits) and all(all(masked[i : i + len(target)]) for i in hits)

    def check(model: str, verdict: str, evidence: str) -> Vote:
        if verdict != "yes":
            return Vote(model, "no", evidence, True, raw_verdict=verdict)
        quote = fold(evidence)
        if not quote:
            return Vote(model, "no", evidence, True, "no evidence quote", verdict)
        if quote not in response:
            return Vote(model, "no", evidence, True, "evidence is not in the response", verdict)
        if echoes_only(evidence):
            return Vote(model, "no", evidence, True, "evidence only echoes the question", verdict)
        return Vote(model, "yes", evidence, True, raw_verdict=verdict)

    return check


def _plain_checker(model: str, verdict: str, evidence: str) -> Vote:
    """The consistency check returns a verdict about the whole response, so no quote is required."""
    return Vote(model, verdict, evidence, True, raw_verdict=verdict)


def collect_vote(
    judges: Sequence[Judge],
    system: str,
    user: str,
    retries: int,
    check: Callable[[str, str, str], Vote],
) -> list[Vote]:
    """One sample slot: retry the primary judge, then fall back to the regrade judge (§6).

    Returns every attempt, in order. At most the last one is valid; none is valid when the slot
    failed outright.
    """
    attempts: list[Vote] = []
    for judge in judges:
        for _ in range(retries + 1):
            try:
                raw = judge.complete(system, user)
            except Exception as exc:  # any SDK or transport failure is a retryable sample failure
                attempts.append(Vote(judge.name, None, None, False, f"error: {type(exc).__name__}: {exc}"))
                continue
            if raw is None:
                attempts.append(Vote(judge.name, None, None, False, "refusal"))
                break  # a refusal is not transient: go to the regrade judge
            parsed = parse_verdict(raw)
            if parsed is None:
                attempts.append(Vote(judge.name, None, None, False, "unparseable response"))
                continue
            attempts.append(check(judge.name, parsed[0], parsed[1]))
            return attempts
    return attempts


def _majority(votes: Iterable[Vote]) -> bool:
    """True when the valid votes say yes more often than no; a tie is a no."""
    verdicts = [v.verdict for v in votes if v.valid]
    return verdicts.count("yes") > verdicts.count("no")


# ------------------------------------------------------------------------------------- real judges


class _ApiJudge:
    """What both real judges share: the model, the effort, the usage callback and a lazy client."""

    def __init__(
        self, model: str, effort: str = DEFAULT_JUDGE_EFFORT, on_usage: UsageCallback | None = None
    ) -> None:
        self.name = model
        self.model = model
        self.effort = effort
        self.on_usage = on_usage
        self._client: Any = None

    def _report(self, input_tokens: int, output_tokens: int, refusal: bool) -> None:
        if self.on_usage is not None:
            self.on_usage(self.name, input_tokens, output_tokens, refusal)


class AnthropicJudge(_ApiJudge):
    """A Claude judge. The SDK and the client are built on first use, never at import."""

    def client(self) -> Any:
        if self._client is None:
            import anthropic  # noqa: PLC0415 — lazy by contract (§7)

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str) -> str | None:
        # Adaptive thinking is on by default and depth comes from output_config.effort; structured
        # outputs give us the verdict JSON directly. No temperature or top_p: the model rejects
        # non-default sampling, so repeated samples are repeated calls.
        message = self.client().messages.create(
            model=self.model,
            max_tokens=JUDGE_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
        )
        refusal = message.stop_reason == "refusal"
        self._report(message.usage.input_tokens, message.usage.output_tokens, refusal)
        if refusal:
            return None
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAIJudge(_ApiJudge):
    """A GPT judge on the Responses API, used as the regrade judge."""

    def client(self) -> Any:
        if self._client is None:
            import openai  # noqa: PLC0415 — lazy by contract (§7)

            self._client = openai.OpenAI()
        return self._client

    def complete(self, system: str, user: str) -> str | None:
        response = self.client().responses.create(
            model=self.model,
            instructions=system,
            input=user,
            max_output_tokens=JUDGE_MAX_TOKENS,
            reasoning={"effort": self.effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "verdict",
                    "schema": VERDICT_SCHEMA,
                    "strict": True,
                }
            },
        )
        refusal = any(
            getattr(part, "type", None) == "refusal"
            for item in response.output
            if getattr(item, "type", None) == "message"
            for part in item.content
        )
        usage = response.usage
        self._report(usage.input_tokens if usage else 0, usage.output_tokens if usage else 0, refusal)
        return None if refusal else response.output_text


def make_judge(
    model: str, effort: str = DEFAULT_JUDGE_EFFORT, on_usage: UsageCallback | None = None
) -> Judge:
    """The real judge for `model`, its provider read from the name (`provider_of`, §7)."""
    provider = provider_of(model)
    if provider is None:
        raise TestsDirError(f"cannot tell which provider serves model {model!r}")
    judge = AnthropicJudge if provider == "anthropic" else OpenAIJudge
    return judge(model, effort, on_usage)


# -------------------------------------------------------------------------------------- the grader


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TestsDirError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise TestsDirError(f"{path} is not valid JSON: {exc}") from exc


def _load_tests_dir(tests_dir: Path) -> tuple[Any, Any, str]:
    """rubric.json, gold.json and question.md, as read from the tests dir; TestsDirError if unreadable."""
    try:
        question = (tests_dir / "question.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise TestsDirError(f"cannot read {tests_dir / 'question.md'}: {exc}") from exc
    return _load_json(tests_dir / "rubric.json"), _load_json(tests_dir / "gold.json"), question


@dataclass(frozen=True)
class ParsedAnswer:
    """A usable answer envelope: the `answer` object, its rationale, and how the reply was formatted."""

    answer: dict[str, Any]
    rationale: str
    answer_format: str  # "bare" | "fenced"


# One markdown fence around the whole reply, with its info string on the opening line (```, ```json):
# a formatting slip, stripped before parsing. The group is non-greedy but anchored at both ends by
# `fullmatch`, so a reply holding two fenced blocks keeps the inner fences and fails to parse, as do
# prose around the object and a fence over only part of the reply.
_ENCLOSING_FENCE = re.compile(r"```[^\n`]*\r?\n(?P<body>.*?)\r?\n?```", re.DOTALL)


def parse_answer(answer_text: str | None) -> ParsedAnswer | None:
    """The §6 step-1 gate. None when the answer is unusable, which grading.json records as `invalid`."""
    if answer_text is None:
        return None
    text = answer_text.lstrip("﻿").strip()  # a byte-order mark is an encoding artefact, not text
    answer_format = "bare"
    fenced = _ENCLOSING_FENCE.fullmatch(text)
    if fenced:
        text, answer_format = fenced.group("body").strip(), "fenced"
    try:
        envelope = json.loads(text)
    except ValueError:
        return None
    if not isinstance(envelope, dict) or not isinstance(envelope.get("answer"), dict):
        return None
    rationale = envelope.get("rationale", "")
    return ParsedAnswer(
        envelope["answer"], rationale if isinstance(rationale, str) else "", answer_format
    )


def _result(
    rubric: dict,
    status: str,
    s_core: CoreResult,
    criteria: list[dict[str, Any]],
    consistency: dict[str, Any],
    truncated: bool,
    spans: Sequence[str],
    judge_models: Sequence[str],
    *,
    answer_format: str,
    usage: JudgeUsage | None = None,
) -> dict[str, Any]:
    judged = [c for c in rubric["criteria"] if not c.get("core")]
    total = sum(c["points"] for c in judged)
    verdicts = {c["id"]: c["verdict"] == "yes" for c in criteria}
    awarded = sum(c["points"] for c in judged if verdicts.get(c["id"]))
    s_judge = awarded / total if total else 0.0
    s = rubric["w_core"] * s_core.score + (1 - rubric["w_core"]) * s_judge
    passed = s >= rubric["tau"] and s_core.score >= 0.9 and s_judge >= 0.5
    return {
        "task_id": rubric["task_id"],
        "status": status,
        "answer_format": answer_format,
        "passed": passed,
        "s": round(s, 6),
        "s_core": round(s_core.score, 6),
        "s_judge": round(s_judge, 6),
        "core_detail": s_core.detail,
        "criteria": criteria,
        "consistency": consistency,
        "truncated": truncated,
        "masked_spans": list(spans),
        "judge_models": list(judge_models),
        "judge_usage": usage.per_model() if usage is not None else {},
        "grade_version": GRADE_VERSION,
    }


def _unjudged_criteria(rubric: dict) -> list[dict[str, Any]]:
    return [
        {"id": c["id"], "points": c["points"], "core": bool(c.get("core")), "verdict": "no", "votes": []}
        for c in rubric["criteria"]
    ]


def grade(
    answer_text: str | None,
    tests_dir: Path | str,
    judge: Judge,
    regrade_judge: Judge | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
    retries: int = DEFAULT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    usage: JudgeUsage | None = None,
) -> dict[str, Any]:
    """Grade one answer against the rubric in `tests_dir`; returns the grading.json content (§6).

    `usage` is the totals object the judges were built with (`make_judge(..., on_usage=usage.add)`); it
    is reported as `judge_usage`. Without it that field is empty, which is what a judge that costs
    nothing — a scripted one in a test — reports anyway.
    """
    tests_dir = Path(tests_dir)
    rubric, gold, question = _load_tests_dir(tests_dir)
    problems = validate_rubric(rubric)
    if problems:
        raise TestsDirError(f"{tests_dir / 'rubric.json'} is invalid: " + "; ".join(problems))
    if not isinstance(gold, dict) or not isinstance(gold.get("answer"), dict):
        raise TestsDirError(f"{tests_dir / 'gold.json'} has no object `answer`")

    judge_models = [judge.name] + ([regrade_judge.name] if regrade_judge else [])
    no_consistency = {"contradiction": False, "votes": []}

    parsed = parse_answer(answer_text)
    if parsed is None:
        return _result(rubric, "invalid_answer", CoreResult(0.0, []), _unjudged_criteria(rubric),
                       no_consistency, False, [], judge_models,
                       answer_format="invalid", usage=usage)
    answer = parsed.answer

    # The judge's view (§6 steps 2-4): the rationale and the answer's strings each cut to the cap, then
    # the mask over all of it. The core is scored on the full answer.
    cap = rubric["rationale_word_cap"]
    rationale, rationale_cut = truncate_words(parsed.rationale, cap)
    shown, answer_cut = truncate_answer(answer, cap)
    truncated = rationale_cut or answer_cut
    response_text = build_response_text(shown, rationale)
    spans = masked_spans(response_text, question)

    if rubric["family"] == "F5" and not _has_primary_mechanism(answer):
        return _result(rubric, "no_primary_mechanism", CoreResult(0.0, []), _unjudged_criteria(rubric),
                       no_consistency, truncated, spans, judge_models,
                       answer_format=parsed.answer_format, usage=usage)

    criteria, consistency = _run_judge(
        rubric, tests_dir, question, response_text, judge, regrade_judge, samples, retries, concurrency
    )
    status = "ok"
    if consistency["contradiction"]:
        status = "contradiction"
        for entry in criteria:
            entry["verdict"] = "no"
    satisfied = {c["id"]: c["verdict"] == "yes" for c in criteria}
    core = score_core(rubric, gold["answer"], answer, satisfied)
    return _result(rubric, status, core, criteria, consistency, truncated, spans, judge_models,
                   answer_format=parsed.answer_format, usage=usage)


def _has_primary_mechanism(answer: dict[str, Any]) -> bool:
    """The §6 step-5 gate: exactly one non-empty `mechanism_primary` string."""
    value = answer.get("mechanism_primary")
    return isinstance(value, str) and bool(value.strip())


def _run_judge(
    rubric: dict,
    tests_dir: Path,
    question: str,
    response_text: str,
    judge: Judge,
    regrade_judge: Judge | None,
    samples: int,
    retries: int,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every criterion x sample plus the consistency check, bounded-concurrently (§6)."""
    prompts = load_prompts(prompt_path(tests_dir))
    judges = [judge] + ([regrade_judge] if regrade_judge else [])
    check = _evidence_checker(response_text, question)

    jobs: dict[tuple[str, int], tuple[str, Callable[[str, str, str], Vote]]] = {}
    for criterion in rubric["criteria"]:
        user = _fill(
            prompts["criterion_check"],
            question=question,
            response=response_text,
            criterion_id=criterion["id"],
            criterion_text=criterion["text"],
        )
        for sample in range(samples):
            jobs[(criterion["id"], sample)] = (user, check)
    consistency_user = _fill(prompts["consistency_check"], question=question, response=response_text)
    for sample in range(samples):
        jobs[("", sample)] = (consistency_user, _plain_checker)

    votes: dict[tuple[str, int], list[Vote]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            key: pool.submit(collect_vote, judges, prompts["system"], user, retries, job_check)
            for key, (user, job_check) in jobs.items()
        }
        for key, future in futures.items():
            votes[key] = future.result()

    def valid_votes(key: str, what: str) -> tuple[list[Vote], list[Vote]]:
        attempts = [v for sample in range(samples) for v in votes[(key, sample)]]
        valid = [v for v in attempts if v.valid]
        if len(valid) < samples:
            # The last failure says why (a missing API key, a refusal, ...), so it goes in the message.
            reason = next((v.reason for v in reversed(attempts) if not v.valid), "no judge call made")
            raise GradingError(
                f"{what} collected {len(valid)} of {samples} valid judge votes; last failure: {reason}"
            )
        return attempts, valid

    criteria: list[dict[str, Any]] = []
    for criterion in rubric["criteria"]:
        attempts, valid = valid_votes(criterion["id"], f"criterion {criterion['id']}")
        criteria.append(
            {
                "id": criterion["id"],
                "points": criterion["points"],
                "core": bool(criterion.get("core")),
                "verdict": "yes" if _majority(valid) else "no",
                "votes": [asdict(v) for v in attempts],
            }
        )

    attempts, valid = valid_votes("", "the consistency check")
    consistency = {"contradiction": _majority(valid), "votes": [asdict(v) for v in attempts]}
    return criteria, consistency


def write_outputs(result: dict[str, Any], out_dir: Path | str) -> None:
    """reward.json (flat and numeric, §6) and grading.json beside it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward = {"reward": result["s"], "core": result["s_core"], "judged": result["s_judge"]}
    (out_dir / "reward.json").write_text(json.dumps(reward) + "\n", encoding="utf-8")
    (out_dir / "grading.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def check_tests_dir(tests_dir: Path | str) -> list[str]:
    """`--check`: validate rubric.json and gold.json, and grade the gold with no judge."""
    tests_dir = Path(tests_dir)
    problems: list[str] = []
    for name in ("rubric.json", "gold.json", "question.md"):
        if not (tests_dir / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    try:
        load_prompts(prompt_path(tests_dir))
    except TestsDirError as exc:
        problems.append(str(exc))

    rubric, gold, question = _load_tests_dir(tests_dir)
    problems += validate_rubric(rubric, question)
    if problems:
        return problems

    if not isinstance(gold, dict) or not isinstance(gold.get("answer"), dict):
        return ["gold.json has no object `answer`"]
    if not isinstance(gold.get("rationale"), str) or not gold["rationale"].strip():
        problems.append("gold.json has no rationale")
    problems += [f"gold.json: {p}" for p in envelope_problems(rubric["family"], gold["answer"])]
    if problems:
        return problems

    core = rubric["core"]
    if core["type"] in _ENUM_MAP_CORES:
        # The gold names exactly the declared entries, with two values at least: a gold missing one
        # would make every complete answer off-contract, and a one-value gold rewards a constant policy.
        field, _, _ = _ENUM_MAP_CORES[core["type"]]
        gold_map = _folded_map(gold["answer"][field]) or {}
        if set(gold_map) != {fold(name) for name in core[field]}:
            problems.append(f"gold.json's {field} are not exactly core.{field}")
        if len(set(gold_map.values())) < 2:
            problems.append(f"gold.json gives every one of its {field} the same value")
    if core["type"] != "judged":
        scored = score_core(rubric, gold["answer"], gold["answer"])
        if scored.score < 1.0:
            problems.append(f"the gold answer scores s_core = {scored.score:.3f} against its own core spec")
    cap = rubric["rationale_word_cap"]
    if truncate_words(gold["rationale"], cap)[1] or truncate_answer(gold["answer"], cap)[1]:
        problems.append(f"the gold rationale or answer is longer than the {cap}-word cap")
    return problems


# --------------------------------------------------------------------------------------------- CLI


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests-dir", required=True, help="the task's tests/ folder")
    parser.add_argument("--answer", help="the answer JSON the solver wrote")
    parser.add_argument("--out-dir", help="where reward.json and grading.json are written")
    parser.add_argument("--check", action="store_true", help="validate the tests dir, call no judge")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument(
        "--regrade-model", default=os.environ.get("JUDGE_REGRADE_MODEL", DEFAULT_REGRADE_MODEL)
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tests_dir = Path(args.tests_dir)
    try:
        if args.check:
            problems = check_tests_dir(tests_dir)
            for problem in problems:
                print(f"invalid: {problem}", file=sys.stderr)
            if not problems:
                print(f"ok: {tests_dir}")
            return 1 if problems else 0

        if not args.answer or not args.out_dir:
            print("--answer and --out-dir are required without --check", file=sys.stderr)
            return 1
        answer_path = Path(args.answer)
        answer_text = answer_path.read_text(encoding="utf-8") if answer_path.exists() else None
        effort = os.environ.get("JUDGE_EFFORT", DEFAULT_JUDGE_EFFORT)
        usage = JudgeUsage()  # so a Harbor run's grading.json carries its own judge spend
        result = grade(
            answer_text,
            tests_dir,
            make_judge(args.judge_model, effort, usage.add),
            make_judge(args.regrade_model, effort, usage.add) if args.regrade_model else None,
            samples=_env_int("JUDGE_SAMPLES", DEFAULT_SAMPLES),
            retries=_env_int("JUDGE_RETRIES", DEFAULT_RETRIES),
            concurrency=_env_int("JUDGE_CONCURRENCY", DEFAULT_CONCURRENCY),
            usage=usage,
        )
    except TestsDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except GradingError as exc:
        print(f"judge failure: {exc}", file=sys.stderr)
        return 2

    write_outputs(result, args.out_dir)
    print(
        f"{result['task_id']} status={result['status']} answer={result['answer_format']} "
        f"reward={result['s']:.3f} core={result['s_core']:.3f} judged={result['s_judge']:.3f} "
        f"passed={result['passed']} judge_calls={usage.calls()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
