"""S10 Packager, first piece: write a task folder's Harbor files (PLAN §9, docs/task_format.md §1, §7, §8).

A task folder arrives with what its author wrote — `instruction.md` and `tests/{question.md, rubric.json,
gold.json}` — and this module writes the rest, after `grade.py --check` accepts the tests dir:

    task.toml              templates/task.toml: task metadata and the profile's [environment] table
    tests/test.sh          templates/test.sh, verbatim
    tests/grade.py         grading/grade.py, verbatim
    tests/judge_prompt.md  grading/judge_prompt.md, verbatim
    solution/solve.sh      templates/solve.sh, with gold.json's answer envelope in a quoted heredoc
    environment/README.md  templates/environment_README.md, verbatim; `local` profile only

Metadata comes from rubric.json (family, direction), the task id (paper_id, its prefix) and the caller
(split, provenance). `tests/question.md` must be exactly what the solver is sent, instruction.md without
its canary line (`strip_canary`), so the judge and the solver read the same question. The environment
profile is the `--environment` option (configs/stages/default.yaml's `package` block holds the values the
M1 S10 stage will pass; the `--sync` command here does not read it):

    local        no container (the default): an empty [environment] table, no image, and the
                 environment/README.md placeholder, since Harbor 0.26 discovers a task folder only when it
                 names a docker_image or has an environment/ folder
    singularity  docker_image = the absolute path of a .sif image (`--image`)
    cloud        docker_image = an OCI ref pinned by digest (`--image`), no network for the agent, and the
                 verifier allowed only the judge hosts

The container profiles get no placeholder: Harbor uploads a non-empty environment/ into the image's
workdir (/app). Packaging a folder for one of them removes a placeholder an earlier `local` run wrote.

`local` and `singularity` give no network isolation, so a task that hands the agent tools is refused
under them (PLAN §9); v1 tasks never do.

    uv run python -m p2c_task_re.stages.package --sync examples/tasks --split train --provenance handbuilt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from string import Template

from p2c_task_re.grading import grade

TEMPLATES = Path(__file__).resolve().with_name("templates")
GRADE_PY = Path(grade.__file__).resolve()
JUDGE_PROMPT = GRADE_PY.with_name("judge_prompt.md")

PROFILES = ("local", "singularity", "cloud")
UNISOLATED_PROFILES = ("local", "singularity")
SPLITS = ("train", "dev", "test", "unassigned")
PROVENANCES = ("handbuilt", "pipeline")
JUDGE_HOSTS = ("api.anthropic.com", "api.openai.com")
DIRECTION_TAGS = {"forward": "-fwd", "backward": "-bwd", None: ""}
LOCAL_PLACEHOLDER = "environment/README.md"  # the `local` profile's stand-in for an image definition

EXECUTABLE = 0o755

# Harbor's `strip_canary` (harbor.models.task.task): leading canary comment lines, then blank lines.
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


class PackageError(Exception):
    """The task folder cannot be packaged as asked: invalid tests dir, bad id, or a forbidden profile."""


def find_tasks(root: Path | str) -> list[Path]:
    """The task folders directly under `root`: every folder that holds a tests/rubric.json."""
    return sorted(path.parent.parent for path in Path(root).glob("*/tests/rubric.json"))


def strip_canary(instruction: str) -> str:
    """instruction.md as Harbor hands it to an agent: leading canary lines and blank lines removed."""
    lines = instruction.split("\n")
    start = 0
    while start < len(lines) and _CANARY_LINE.match(lines[start].strip()):
        start += 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    return "\n".join(lines[start:])


def paper_id_of(task_id: str, family: str, direction: str | None) -> str:
    """The paper id a task id starts with (PLAN §10: `<paper_id>-<family>-<fwd|bwd>-<NN>`; F5/F7 drop
    the direction)."""
    pattern = rf"(?P<paper_id>[^-]+)-{family.lower()}{DIRECTION_TAGS[direction]}-\d{{2}}"
    match = re.fullmatch(pattern, task_id)
    if not match:
        raise PackageError(f"task id {task_id!r} does not follow PLAN §10 naming for {family} {direction}")
    return match.group("paper_id")


def render(
    task_dir: Path | str,
    *,
    split: str,
    provenance: str,
    environment: str = "local",
    image: str | None = None,
) -> dict[str, bytes]:
    """Every generated file of one task folder, as {path relative to the folder: content}. Writes nothing."""
    task_dir = Path(task_dir)
    tests_dir = task_dir / "tests"
    try:
        problems = grade.check_tests_dir(tests_dir)
    except grade.TestsDirError as exc:
        problems = [str(exc)]
    if problems:
        raise PackageError(f"{tests_dir} fails grade.py --check: " + "; ".join(problems))
    if split not in SPLITS:
        raise PackageError(f"split {split!r} is not one of {SPLITS}")
    if provenance not in PROVENANCES:
        raise PackageError(f"provenance {provenance!r} is not one of {PROVENANCES}")

    rubric = json.loads((tests_dir / "rubric.json").read_text(encoding="utf-8"))
    gold = json.loads((tests_dir / "gold.json").read_text(encoding="utf-8"))
    task_id = task_dir.name
    if rubric["task_id"] != task_id or gold.get("task_id") != task_id:
        raise PackageError(
            f"{task_dir}: folder name, rubric task_id ({rubric['task_id']!r}) and gold task_id "
            f"({gold.get('task_id')!r}) must agree"
        )
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    if (tests_dir / "question.md").read_text(encoding="utf-8") != strip_canary(instruction):
        raise PackageError(
            f"{tests_dir / 'question.md'} must be instruction.md without its canary line, exactly what "
            "the solver is sent, so the judge grades against the same question"
        )

    metadata = {
        "family": rubric["family"],
        "direction": rubric["direction"],  # null for F5/F7, so the key is left out (TOML has no null)
        "paper_id": paper_id_of(task_id, rubric["family"], rubric["direction"]),
        "split": split,
        "provenance": provenance,
    }
    task_toml = render_task_toml(task_id, metadata, environment, image)
    refuse_agent_tools(task_toml, environment)

    envelope = json.dumps({"answer": gold["answer"], "rationale": gold["rationale"]}, ensure_ascii=False)
    solve_sh = _template("solve.sh").substitute(gold_envelope=envelope)
    files = {
        "task.toml": task_toml.encode("utf-8"),
        "tests/test.sh": (TEMPLATES / "test.sh").read_bytes(),
        "tests/grade.py": GRADE_PY.read_bytes(),
        "tests/judge_prompt.md": JUDGE_PROMPT.read_bytes(),
        "solution/solve.sh": solve_sh.encode("utf-8"),
    }
    if environment == "local":
        files[LOCAL_PLACEHOLDER] = (TEMPLATES / "environment_README.md").read_bytes()
    return files


def render_task_toml(
    task_id: str, metadata: dict[str, str | None], environment: str, image: str | None
) -> str:
    """task.toml for one task under one environment profile (docs/task_format.md §8)."""
    environment_lines: list[str] = []
    verifier_lines: list[str] = []
    if environment == "local":
        if image is not None:
            raise PackageError("the local profile runs without a container, so it takes no image")
    elif environment == "singularity":
        # Harbor runs only a `*.sif` path as a prebuilt image; any other value it would try to pull.
        if not image or not Path(image).is_absolute() or not image.endswith(".sif"):
            raise PackageError(f"the singularity profile needs an absolute .sif path, got {image!r}")
        environment_lines.append(f"docker_image = {_toml_string(image)}")
    elif environment == "cloud":
        if not image or "@sha256:" not in image:
            raise PackageError(
                f"the cloud profile needs an OCI image pinned by digest (…@sha256:…), got {image!r}"
            )
        environment_lines += [f"docker_image = {_toml_string(image)}", 'network_mode = "no-network"']
        hosts = ", ".join(_toml_string(host) for host in JUDGE_HOSTS)
        verifier_lines += ['network_mode = "allowlist"', f"allowed_hosts = [{hosts}]"]
    else:
        raise PackageError(f"environment profile {environment!r} is not one of {PROFILES}")

    return _template("task.toml").substitute(
        task_id=task_id,
        metadata="\n".join(
            f"{key} = {_toml_string(value)}" for key, value in metadata.items() if value is not None
        ),
        environment="".join(line + "\n" for line in environment_lines),
        verifier_network="".join(line + "\n" for line in verifier_lines),
    )


def refuse_agent_tools(task_toml: str, environment: str) -> None:
    """PLAN §9's guard: no tools for the agent under a profile without network isolation.

    Checks the task.toml about to be written, so a template that ever grows MCP servers or skills cannot
    ship them under `local` or `singularity`, where whatever the agent runs would run unisolated.
    """
    table = tomllib.loads(task_toml).get("environment", {})
    if environment in UNISOLATED_PROFILES and (table.get("mcp_servers") or table.get("skills_dir")):
        raise PackageError(
            f"the {environment} profile has no network isolation, so a task there may not give the "
            "agent tools"
        )


def package_task(
    task_dir: Path | str,
    *,
    split: str,
    provenance: str,
    environment: str = "local",
    image: str | None = None,
) -> list[Path]:
    """Write the generated files into `task_dir`; returns the files whose content changed or that it
    removed (the `local` placeholder, when packaging for a container profile)."""
    task_dir = Path(task_dir)
    files = render(task_dir, split=split, provenance=provenance, environment=environment, image=image)
    changed = []
    for relative, content in files.items():
        path = task_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
            changed.append(path)
        if path.suffix == ".sh":
            path.chmod(EXECUTABLE)
    placeholder = task_dir / LOCAL_PLACEHOLDER
    if LOCAL_PLACEHOLDER not in files and placeholder.exists():
        placeholder.unlink()
        changed.append(placeholder)
        if not any(placeholder.parent.iterdir()):
            placeholder.parent.rmdir()
    return changed


def sync(
    root: Path | str,
    *,
    split: str,
    provenance: str,
    environment: str = "local",
    image: str | None = None,
) -> list[Path]:
    """Package every task folder under `root`; returns the files whose content changed."""
    tasks = find_tasks(root)
    if not tasks:
        raise PackageError(f"no task folders (*/tests/rubric.json) under {root}")
    changed = []
    for task_dir in tasks:
        changed += package_task(
            task_dir, split=split, provenance=provenance, environment=environment, image=image
        )
    return changed


def _template(name: str) -> Template:
    return Template((TEMPLATES / name).read_text(encoding="utf-8"))


def _toml_string(value: str) -> str:
    """A TOML basic string. JSON's string escapes are a subset of TOML's, so json.dumps writes one."""
    return json.dumps(value, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Harbor files of every task folder under DIR.")
    parser.add_argument("--sync", type=Path, required=True, metavar="DIR", help="a folder of task folders")
    parser.add_argument("--split", required=True, choices=SPLITS, help="the papers' frozen-manifest split")
    parser.add_argument("--provenance", required=True, choices=PROVENANCES)
    parser.add_argument(
        "--environment", default="local", choices=PROFILES, help="the profile (default: local)"
    )
    parser.add_argument("--image", help="the .sif path (singularity) or OCI ref @sha256:… (cloud)")
    args = parser.parse_args(argv)
    try:
        changed = sync(
            args.sync,
            split=args.split,
            provenance=args.provenance,
            environment=args.environment,
            image=args.image,
        )
    except PackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in changed:
        print(f"{'wrote' if path.exists() else 'removed'} {path}")
    print(f"{len(find_tasks(args.sync))} task folders in sync, {len(changed)} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
