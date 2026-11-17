"""Answer contracts for the seven task families (PLAN §7, docs/task_format.md §5).

The package side of the contract: which directions a family may take, the JSON schema of its answer
envelope (for the solver's structured output and for the instruction's schema block), and a
placeholder example answer that the instruction renders.

Scoring lives in `p2c_task_re.grading.grade` (standalone, so it can be copied into a task folder). The
`example_answer` wording is provisional and belongs to the instruction builder once that exists.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DIRECTIONS", "answer_fields_schema", "answer_schema", "example_answer"]

# Mirrors configs/families/v1.yaml. F5 and F7 are direction-neutral: their target is neither the
# outcome nor the perturbation (PLAN §1).
DIRECTIONS: dict[str, tuple[str, ...]] = {
    "F1": ("forward",),
    "F2": ("forward", "backward"),
    "F3": ("forward",),
    "F4": ("backward",),
    "F5": (),
    "F6": ("forward", "backward"),
    "F7": (),
}


def _str_array(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": description}


def _enum_map(keys: list[str], values: list[str], description: str) -> dict[str, Any]:
    """An object with one fixed key per arm or marker, each holding one of `values`."""
    return {
        "type": "object",
        "description": description,
        "properties": {key: {"type": "string", "enum": list(values)} for key in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


def answer_fields_schema(family: str, core: dict[str, Any]) -> dict[str, Any]:
    """The JSON schema of the `answer` object alone, built from the rubric's `core` block."""
    family = family.upper()
    if family == "F1":
        return {
            "type": "object",
            "properties": {
                "markers": _enum_map(core["markers"], core["classes"], "one direction per marker")
            },
            "required": ["markers"],
            "additionalProperties": False,
        }
    if family in ("F2", "F6"):
        properties = {"arms": _enum_map(core["arms"], core["values"], "one outcome per arm")}
        required = ["arms"]
        if family == "F6":
            properties["state"] = {"type": "string", "description": "the resulting cell state"}
            required.append("state")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    if family == "F3":
        return {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "description": "tie sets, strongest effect first, each arm exactly once",
                    "items": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(core["arms"])},
                        "minItems": 1,
                    },
                    "minItems": 1,
                }
            },
            "required": ["ranking"],
            "additionalProperties": False,
        }
    if family == "F4":
        return {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": list(core["options"])}},
            "required": ["choice"],
            "additionalProperties": False,
        }
    if family == "F5":
        return {
            "type": "object",
            "properties": {
                "mechanism_primary": {
                    "type": "string",
                    "description": "the primary mechanism, in one sentence",
                },
                "key_entities": _str_array("the entities the mechanism turns on"),
            },
            "required": ["mechanism_primary", "key_entities"],
            "additionalProperties": False,
        }
    if family == "F7":
        return {
            "type": "object",
            "properties": {"experiment": {"type": "string", "description": "the decisive experiment"}},
            "required": ["experiment"],
            "additionalProperties": False,
        }
    raise ValueError(f"unknown family {family!r}")


def answer_schema(family: str, core: dict[str, Any]) -> dict[str, Any]:
    """The JSON schema of the whole answer envelope (§2): `{"answer": ..., "rationale": ...}`."""
    return {
        "type": "object",
        "properties": {
            "answer": answer_fields_schema(family, core),
            "rationale": {"type": "string", "description": "the derivation, graded by the judge"},
        },
        "required": ["answer", "rationale"],
        "additionalProperties": False,
    }


def example_answer(family: str, core: dict[str, Any]) -> dict[str, Any]:
    """A placeholder envelope for the instruction's schema block (§1). Values are not answers."""
    family = family.upper()
    if family == "F1":
        choices = "|".join(core["classes"])
        answer: dict[str, Any] = {"markers": {marker: choices for marker in core["markers"]}}
    elif family in ("F2", "F6"):
        choices = "|".join(core["values"])
        answer = {"arms": {arm: choices for arm in core["arms"]}}
        if family == "F6":
            answer["state"] = "<the resulting cell state>"
    elif family == "F3":
        arms = list(core["arms"])
        answer = {"ranking": [[arms[0]], arms[1:]] if len(arms) > 2 else [[arm] for arm in arms]}
    elif family == "F4":
        answer = {"choice": "|".join(core["options"])}
    elif family == "F5":
        answer = {
            "mechanism_primary": "<one sentence: the mechanism you conclude>",
            "key_entities": ["<entity>", "<entity>"],
        }
    elif family == "F7":
        answer = {"experiment": "<the experiment that would decide between the hypotheses>"}
    else:
        raise ValueError(f"unknown family {family!r}")
    return {"answer": answer, "rationale": "<your derivation>"}
