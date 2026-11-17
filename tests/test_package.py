"""Tests for the packager (S10 first piece): docs/task_format.md §1, §7, §8 and PLAN §9."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from p2c_task_re.evaluate import JudgeConfig
from p2c_task_re.grading import grade as g
from p2c_task_re.stages import package

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "tasks"
EXAMPLE_IDS = [path.name for path in package.find_tasks(EXAMPLES)]
# What the examples were packaged with (examples/tasks/README.md: both papers are train, built by hand).
EXAMPLE_ARGS = {"split": "train", "provenance": "handbuilt"}
SYNC_COMMAND = (
    "uv run python -m p2c_task_re.stages.package --sync examples/tasks --split train --provenance handbuilt"
)
GENERATED = (
    "task.toml",
    "tests/test.sh",
    "tests/grade.py",
    "tests/judge_prompt.md",
    "solution/solve.sh",
    "environment/README.md",  # local profile only
)
HAND_WRITTEN = ("instruction.md", "tests/question.md", "tests/rubric.json", "tests/gold.json")
SIF = "/dfs/images/p2c-base.sif"
OCI = "ghcr.io/example/p2c-base@sha256:" + "0" * 64


def hand_written_copy(task_id: str, root: Path) -> Path:
    """A task folder holding only what its author wrote, as it looks before packaging."""
    task_dir = root / task_id
    for relative in HAND_WRITTEN:
        (task_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXAMPLES / task_id / relative, task_dir / relative)
    return task_dir


def files_under(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def task_toml(task_id: str, **kwargs: str) -> dict:
    files = package.render(EXAMPLES / task_id, **EXAMPLE_ARGS, **kwargs)
    return tomllib.loads(files["task.toml"].decode("utf-8"))


# ------------------------------------------------------------------------------------- the examples


def test_there_are_three_examples() -> None:
    assert len(EXAMPLE_IDS) == 3


@pytest.mark.parametrize("task_id", EXAMPLE_IDS)
def test_the_examples_are_exactly_what_the_packager_writes(task_id: str, tmp_path: Path) -> None:
    """Re-package a hand-written copy and compare every file byte for byte, so the examples never drift."""
    fresh = hand_written_copy(task_id, tmp_path)
    package.package_task(fresh, **EXAMPLE_ARGS)
    on_disk = files_under(EXAMPLES / task_id)
    assert sorted(on_disk) == sorted(files_under(fresh)), f"files differ; re-run: {SYNC_COMMAND}"
    for relative, content in files_under(fresh).items():
        assert on_disk[relative] == content, f"{task_id}/{relative} drifted; re-run: {SYNC_COMMAND}"
    for script in ("tests/test.sh", "solution/solve.sh"):
        assert os.access(EXAMPLES / task_id / script, os.X_OK), f"{script} is not executable"


@pytest.mark.parametrize("task_id", EXAMPLE_IDS)
def test_the_grader_in_each_example_is_the_source_copy(task_id: str) -> None:
    tests_dir = EXAMPLES / task_id / "tests"
    assert (tests_dir / "grade.py").read_bytes() == Path(g.__file__).read_bytes()
    assert (tests_dir / "judge_prompt.md").read_bytes() == package.JUDGE_PROMPT.read_bytes()


@pytest.mark.parametrize("task_id", EXAMPLE_IDS)
def test_solve_sh_writes_the_gold_envelope(task_id: str, tmp_path: Path) -> None:
    subprocess.run(
        ["bash", str(EXAMPLES / task_id / "solution" / "solve.sh")],
        env={**os.environ, "OUTPUT_DIR": str(tmp_path / "out")},
        check=True,
    )
    gold = json.loads((EXAMPLES / task_id / "tests" / "gold.json").read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "out" / "answer.json").read_text(encoding="utf-8"))
    assert written == {"answer": gold["answer"], "rationale": gold["rationale"]}


@pytest.mark.parametrize("task_id", EXAMPLE_IDS)
def test_harbor_loads_each_example(task_id: str) -> None:
    task_module = pytest.importorskip("harbor.models.task.task")
    # What `harbor run -p examples/tasks` discovers: a folder with no docker_image needs environment/.
    assert task_module.Task.is_valid_dir(EXAMPLES / task_id)
    task = task_module.Task(EXAMPLES / task_id)
    assert task.name == f"p2c/{task_id}"
    assert task.config.verifier.environment_mode.value == "separate"
    assert task.config.artifacts == ["/app/output"]
    assert task.config.environment.docker_image is None


def test_sync_is_idempotent_and_reports_only_changes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for task_id in EXAMPLE_IDS:
        hand_written_copy(task_id, tmp_path)
    args = ["--sync", str(tmp_path), "--split", "train", "--provenance", "handbuilt"]
    assert package.main(args) == 0
    assert "18 files changed" in capsys.readouterr().out
    assert package.main(args) == 0
    assert "0 files changed" in capsys.readouterr().out


# -------------------------------------------------------------------------------------- task.toml


def test_metadata_comes_from_the_rubric_the_id_and_the_arguments() -> None:
    f4 = task_toml("2026.04.25.720790-f4-bwd-01")
    assert f4["task"] == {"name": "p2c/2026.04.25.720790-f4-bwd-01", "version": "1.0.0"}
    assert f4["metadata"] == {
        "family": "F4",
        "direction": "backward",
        "paper_id": "2026.04.25.720790",
        "split": "train",
        "provenance": "handbuilt",
    }


def test_a_direction_neutral_family_has_no_direction_key() -> None:
    """TOML has no null, so F5's `direction: null` is written as an absent key."""
    f5 = task_toml("2026.04.25.720790-f5-01")
    assert "direction" not in f5["metadata"]
    assert f5["metadata"]["family"] == "F5"


def test_the_verifier_declares_every_judge_knob() -> None:
    """`--ve` can only override a declared variable, so every JUDGE_* variable grade.py reads is here,
    defaulting to grade.py's own default."""
    config = task_toml(EXAMPLE_IDS[0])
    assert config["verifier"]["environment_mode"] == "separate"
    knobs = JudgeConfig().env()  # the JUDGE_* variables grade.py's CLI reads, at grade.py's defaults
    assert set(config["verifier"]["env"]) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", *knobs}
    for name, default in knobs.items():
        assert config["verifier"]["env"][name] == f"${{{name}:-{default}}}"
    assert config["verifier"]["env"]["JUDGE_MODEL"] == "${JUDGE_MODEL:-claude-sonnet-5-5}"


def test_local_profile_has_an_empty_environment() -> None:
    config = task_toml(EXAMPLE_IDS[0])
    assert config["environment"] == {}
    assert "network_mode" not in config["verifier"]


def test_only_the_local_profile_gets_the_environment_placeholder() -> None:
    """Harbor discovers a task with no image only through environment/; with an image it would upload a
    non-empty environment/ into /app, so the container profiles get none."""
    local = package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS)
    assert b"LocalEnvironment" in local[package.LOCAL_PLACEHOLDER]
    for environment, image in (("singularity", SIF), ("cloud", OCI)):
        container = package.render(
            EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS, environment=environment, image=image
        )
        assert not any(name.startswith("environment/") for name in container)


def test_packaging_for_a_container_profile_removes_the_placeholder(tmp_path: Path) -> None:
    task_dir = hand_written_copy(EXAMPLE_IDS[0], tmp_path)
    package.package_task(task_dir, **EXAMPLE_ARGS)
    assert (task_dir / package.LOCAL_PLACEHOLDER).is_file()
    changed = package.package_task(task_dir, **EXAMPLE_ARGS, environment="singularity", image=SIF)
    assert task_dir / package.LOCAL_PLACEHOLDER in changed
    assert not (task_dir / "environment").exists()


def test_singularity_profile_points_at_the_sif() -> None:
    config = task_toml(EXAMPLE_IDS[0], environment="singularity", image=SIF)
    assert config["environment"] == {"docker_image": SIF}
    assert "network_mode" not in config["verifier"]


def test_cloud_profile_isolates_the_agent_and_allowlists_the_judges() -> None:
    config = task_toml(EXAMPLE_IDS[0], environment="cloud", image=OCI)
    assert config["environment"] == {"docker_image": OCI, "network_mode": "no-network"}
    assert config["verifier"]["network_mode"] == "allowlist"
    assert config["verifier"]["allowed_hosts"] == ["api.anthropic.com", "api.openai.com"]


@pytest.mark.parametrize(("environment", "image"), [("local", None), ("singularity", SIF), ("cloud", OCI)])
def test_harbor_accepts_every_profile(environment: str, image: str | None) -> None:
    config_module = pytest.importorskip("harbor.models.task.config")
    files = package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS, environment=environment, image=image)
    config = config_module.TaskConfig.model_validate_toml(files["task.toml"].decode("utf-8"))
    assert config.environment.docker_image == image


def test_only_task_toml_and_the_placeholder_depend_on_the_profile() -> None:
    local = package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS)
    cloud = package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS, environment="cloud", image=OCI)
    assert sorted(local) == sorted(GENERATED)
    assert sorted(local.keys() - cloud.keys()) == [package.LOCAL_PLACEHOLDER]
    assert [name for name in cloud if local[name] != cloud[name]] == ["task.toml"]


@pytest.mark.parametrize(
    ("environment", "image", "fragment"),
    [
        ("local", SIF, "takes no image"),
        ("singularity", None, "absolute .sif"),
        ("singularity", "images/p2c-base.sif", "absolute .sif"),
        ("singularity", "/dfs/images/p2c-base-2026-11-16.img", "absolute .sif"),  # Harbor would pull it
        ("cloud", "ghcr.io/example/p2c-base:latest", "pinned by digest"),
        ("docker", None, "not one of"),
    ],
)
def test_a_profile_rejects_the_wrong_image(environment: str, image: str | None, fragment: str) -> None:
    with pytest.raises(package.PackageError, match=fragment):
        package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS, environment=environment, image=image)


# ------------------------------------------------------------------------------------------ guards


TOOLS_TOML = '[environment]\n[[environment.mcp_servers]]\nname = "shell"\ncommand = "bash"\n'


@pytest.mark.parametrize("environment", package.UNISOLATED_PROFILES)
def test_no_tools_for_the_agent_without_network_isolation(environment: str) -> None:
    with pytest.raises(package.PackageError, match="no network isolation"):
        package.refuse_agent_tools(TOOLS_TOML, environment)
    with pytest.raises(package.PackageError, match="no network isolation"):
        package.refuse_agent_tools('[environment]\nskills_dir = "/skills"\n', environment)


def test_tools_are_allowed_only_under_the_isolated_profile() -> None:
    package.refuse_agent_tools(TOOLS_TOML, "cloud")


def test_every_rendered_task_toml_passes_the_guard() -> None:
    files = package.render(EXAMPLES / EXAMPLE_IDS[0], **EXAMPLE_ARGS)
    package.refuse_agent_tools(files["task.toml"].decode("utf-8"), "local")


def test_an_invalid_tests_dir_is_not_packaged(tmp_path: Path) -> None:
    task_dir = hand_written_copy(EXAMPLE_IDS[0], tmp_path)
    rubric = json.loads((task_dir / "tests" / "rubric.json").read_text(encoding="utf-8"))
    rubric["w_core"] = 0.9
    (task_dir / "tests" / "rubric.json").write_text(json.dumps(rubric), encoding="utf-8")
    with pytest.raises(package.PackageError, match="grade.py --check"):
        package.package_task(task_dir, **EXAMPLE_ARGS)
    assert not (task_dir / "task.toml").exists()


def test_question_md_must_be_the_instruction_the_solver_sees(tmp_path: Path) -> None:
    """The judge grades against question.md, so it may not drift from what the solver was asked."""
    task_dir = hand_written_copy(EXAMPLE_IDS[1], tmp_path)
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    question = (task_dir / "tests" / "question.md").read_text(encoding="utf-8")
    assert question == package.strip_canary(instruction)
    (task_dir / "instruction.md").write_text(instruction.replace("PRMT7", "PRMT5"), encoding="utf-8")
    with pytest.raises(package.PackageError, match="question.md must be instruction.md"):
        package.render(task_dir, **EXAMPLE_ARGS)


def test_the_folder_name_must_be_the_task_id(tmp_path: Path) -> None:
    task_dir = hand_written_copy(EXAMPLE_IDS[0], tmp_path)
    renamed = task_dir.rename(tmp_path / "2026.02.17.706358-f2-fwd-02")
    with pytest.raises(package.PackageError, match="must agree"):
        package.render(renamed, **EXAMPLE_ARGS)


@pytest.mark.parametrize(
    ("task_id", "family", "direction", "paper_id"),
    [
        ("2026.04.25.720790-f4-bwd-01", "F4", "backward", "2026.04.25.720790"),
        ("2026.02.17.706358-f2-fwd-01", "F2", "forward", "2026.02.17.706358"),
        ("2026.04.25.720790-f5-01", "F5", None, "2026.04.25.720790"),
    ],
)
def test_paper_id_of(task_id: str, family: str, direction: str | None, paper_id: str) -> None:
    assert package.paper_id_of(task_id, family, direction) == paper_id


@pytest.mark.parametrize(
    ("task_id", "family", "direction"),
    [
        ("2026.04.25.720790-f4-fwd-01", "F4", "backward"),  # direction tag disagrees with the rubric
        ("2026.04.25.720790-f5-01", "F4", "backward"),  # family disagrees
        ("2026.04.25.720790-f5-1", "F5", None),  # counter is two digits
    ],
)
def test_paper_id_of_rejects_an_id_that_disagrees_with_the_rubric(
    task_id: str, family: str, direction: str | None
) -> None:
    with pytest.raises(package.PackageError, match="naming"):
        package.paper_id_of(task_id, family, direction)


@pytest.mark.parametrize(("field", "value"), [("split", "eval"), ("provenance", "scraped")])
def test_split_and_provenance_are_checked(field: str, value: str) -> None:
    with pytest.raises(package.PackageError, match=field):
        package.render(EXAMPLES / EXAMPLE_IDS[0], **{**EXAMPLE_ARGS, field: value})
