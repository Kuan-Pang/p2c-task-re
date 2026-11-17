"""Tests for Harbor without containers: LocalEnvironment and SingleShotAgent (PLAN §9, task_format.md §8).

The environment is exercised on a tmp trial folder, then end to end through the real `harbor` CLI on
examples/tasks itself, with a stand-in `anthropic` package on PYTHONPATH so the verifier's grade.py gets
scripted "yes" verdicts and makes no network call. The agent runs against a fake environment and the real
one, with `llm.complete` monkeypatched.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory  # noqa: E402
from harbor.agents.installed.base import ApiError  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy, TaskConfig  # noqa: E402
from harbor.models.trial.config import AgentConfig  # noqa: E402
from harbor.models.trial.error_policy import policy_for  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from p2c_task_re import llm  # noqa: E402
from p2c_task_re.harbor_agent import ANSWER_PATH, SingleShotAgent, solver_spec  # noqa: E402
from p2c_task_re.harbor_env import SAFE_AGENTS, VENV_BIN, LocalEnvironment  # noqa: E402
from p2c_task_re.llm import Completion  # noqa: E402
from p2c_task_re.stages import package  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "tasks"
TASKS = package.find_tasks(EXAMPLES)
F4 = next(task for task in TASKS if "-f4-" in task.name)
HARBOR = VENV_BIN / "harbor"
LOCAL = "p2c_task_re.harbor_env:LocalEnvironment"

FAKE_SDK = REPO / "tests" / "fake_sdk"  # an anthropic stand-in for the verifier's grade.py


def environment_config(task_dir: Path = F4) -> EnvironmentConfig:
    return TaskConfig.model_validate_toml((task_dir / "task.toml").read_text(encoding="utf-8")).environment


def make_env(
    tmp_path: Path, config: EnvironmentConfig | None = None, task_dir: Path = F4, **options: Any
) -> LocalEnvironment:
    """A LocalEnvironment for a trial folder under tmp_path, built the way Harbor's factory builds it."""
    trial = TrialPaths(tmp_path / "trial")
    trial.mkdir()
    return LocalEnvironment(
        environment_dir=task_dir / "environment",
        environment_name=task_dir.name,
        session_id="t__env",
        trial_paths=trial,
        task_env_config=config or environment_config(task_dir),
        **options,
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def gold_envelope(task_dir: Path) -> dict[str, Any]:
    gold = json.loads((task_dir / "tests" / "gold.json").read_text(encoding="utf-8"))
    return {"answer": gold["answer"], "rationale": gold["rationale"]}


# ---------------------------------------------------------------------------------- path mapping


def test_container_paths_map_onto_the_environment_folder_and_the_trial_logs(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    trial = (tmp_path / "trial").resolve()
    assert env.root == trial / "t__env"
    assert env.host_path("/app/output/answer.json") == env.root / "app" / "output" / "answer.json"
    assert env.host_path("/tests/test.sh") == env.root / "tests" / "test.sh"
    assert env.host_path("/solution") == env.root / "solution"
    assert env.host_path("/logs/verifier/reward.json") == trial / "verifier" / "reward.json"
    assert env.host_path("/logs/agent") == trial / "agent"
    assert env.host_path("/logs/artifacts/x.txt") == trial / "artifacts" / "logs" / "artifacts" / "x.txt"
    assert env.host_path("/logs/other") == env.root / "logs" / "other"
    assert env.host_path("output") == env.root / "app" / "output"  # relative to the workdir
    assert env.host_path("/usr/bin/env") == Path("/usr/bin/env")
    assert env.host_path("/tmp/.hb-transfer.tar.gz") == Path("/tmp/.hb-transfer.tar.gz")


def test_commands_are_rewritten_at_path_boundaries_only(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    verifier = (tmp_path / "trial" / "verifier").resolve()
    assert env.host_command("(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1") == (
        f"({env.root}/tests/test.sh) > {verifier}/test-stdout.txt 2>&1"
    )
    assert env.host_command("test -d '/app/output'") == f"test -d '{env.root}/app/output'"
    assert env.host_command("cd /logs/agent && find . -type f") == (
        f"cd {(tmp_path / 'trial' / 'agent').resolve()} && find . -type f"
    )
    for untouched in ("ls /home/u/app/x", "ls /apps", "ls ./app", "curl http://h/app", "ls ~/app"):
        assert env.host_command(untouched) == untouched
    root = env.root
    assert env.host_command("PATH=/app/bin:/app:$PATH") == f"PATH={root}/app/bin:{root}/app:$PATH"
    assert env.host_command("ls {/app,/tests} /tests*") == f"ls {{{root}/app,{root}/tests}} {root}/tests*"


def test_without_auto_mounting_the_logs_stay_in_the_environment_folder(tmp_path: Path) -> None:
    env = make_env(tmp_path, auto_mounting=False)
    assert env.host_path("/logs/verifier") == env.root / "logs" / "verifier"
    assert not env.is_path_mounted("/logs/verifier", tmp_path / "trial" / "verifier")


def test_the_logs_are_mounted_so_harbor_copies_nothing_back(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert env.is_path_mounted("/logs/verifier", tmp_path / "trial" / "verifier")
    assert env.is_path_mounted("/logs/agent", tmp_path / "trial" / "agent")


def test_the_root_option_moves_the_environment_folder(tmp_path: Path) -> None:
    env = make_env(tmp_path, root=str(tmp_path / "elsewhere"))
    assert env.root == (tmp_path / "elsewhere" / "t__env").resolve()


def test_unknown_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        LocalEnvironment.preflight({"bogus": 1})
    LocalEnvironment.preflight({"root": "/x", "path": "/a:/b", "auto_mounting": False})


# ------------------------------------------------------------------------- what it refuses to run


def rendered_environment(tmp_path: Path, environment: str, image: str) -> EnvironmentConfig:
    files = package.render(F4, split="train", provenance="handbuilt", environment=environment, image=image)
    path = tmp_path / f"{environment}.toml"
    path.write_bytes(files["task.toml"])
    return TaskConfig.model_validate_toml(path.read_text(encoding="utf-8")).environment


@pytest.mark.parametrize(
    ("environment", "image"),
    [("singularity", "/images/p2c-base.sif"), ("cloud", "ghcr.io/x/p2c-base@sha256:" + "0" * 64)],
)
def test_a_task_packaged_for_a_container_profile_is_refused(
    tmp_path: Path, environment: str, image: str
) -> None:
    with pytest.raises(ValueError, match="without a container"):
        make_env(tmp_path, rendered_environment(tmp_path, environment, image))


@pytest.mark.parametrize("mode", [NetworkMode.NO_NETWORK, NetworkMode.ALLOWLIST])
def test_a_restricted_network_is_refused(tmp_path: Path, mode: NetworkMode) -> None:
    # The trial derives the policy from task.toml and hands it over; this environment cannot enforce one.
    policy = NetworkPolicy(
        network_mode=mode, allowed_hosts=["api.anthropic.com"] if mode.value == "allowlist" else []
    )
    with pytest.raises(ValueError, match="not supported"):
        make_env(tmp_path, network_policy=policy)


def test_a_task_that_gives_the_agent_tools_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tools"):
        make_env(tmp_path, environment_config().model_copy(update={"skills_dir": "/skills"}))


def test_a_task_that_builds_an_image_is_refused(tmp_path: Path) -> None:
    task_dir = tmp_path / F4.name
    shutil.copytree(F4, task_dir)
    (task_dir / "environment").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM debian\n", encoding="utf-8")
    with pytest.raises(ValueError, match="environment/Dockerfile"):
        make_env(tmp_path, task_dir=task_dir)


# ------------------------------------------------------------------------------- Harbor task.toml


@pytest.mark.parametrize("task_dir", TASKS, ids=[task.name for task in TASKS])
def test_every_example_parses_as_a_local_profile_task(task_dir: Path, tmp_path: Path) -> None:
    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert config.environment.docker_image is None
    assert config.environment.network_mode.value == "public"
    assert config.verifier.environment_mode.value == "separate"
    assert [artifact if isinstance(artifact, str) else artifact.source for artifact in config.artifacts] == [
        "/app/output"
    ]
    make_env(tmp_path, config.environment, task_dir=task_dir)


@pytest.mark.parametrize(
    ("environment", "image"),
    [("singularity", "/images/p2c-base.sif"), ("cloud", "ghcr.io/x/p2c-base@sha256:" + "0" * 64)],
)
def test_the_container_profiles_parse_as_harbor_tasks(tmp_path: Path, environment: str, image: str) -> None:
    task_dir = tmp_path / F4.name
    shutil.copytree(F4, task_dir)
    package.package_task(
        task_dir, split="train", provenance="handbuilt", environment=environment, image=image
    )
    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert config.environment.docker_image == image
    assert not (task_dir / "environment").exists()  # Harbor would upload a non-empty one into /app
    if environment == "cloud":
        assert config.environment.network_mode.value == "no-network"
        assert config.verifier.network_mode.value == "allowlist"


# ---------------------------------------------------------------------------- file ops and exec


def test_the_oracle_flow_on_a_tmp_trial(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    trial = TrialPaths(tmp_path / "trial")

    async def flow() -> None:
        await env.start(force_build=False)
        await env.upload_dir(F4 / "solution", "/solution")
        assert (await env.exec("chmod +x /solution/solve.sh")).return_code == 0
        result = await env.exec("(/solution/solve.sh) > /logs/agent/oracle.txt 2>&1")
        assert result.return_code == 0
        assert await env.is_dir("/app/output")
        assert await env.is_file("/app/output/answer.json")
        await env.download_dir("/app/output", tmp_path / "collected")
        await env.download_file("/app/output/answer.json", tmp_path / "answer.json")
        await env.stop(delete=True)

    run(flow())
    answer = json.loads((tmp_path / "collected" / "answer.json").read_text(encoding="utf-8"))
    assert answer == gold_envelope(F4)
    assert (tmp_path / "answer.json").read_bytes() == (tmp_path / "collected" / "answer.json").read_bytes()
    assert (trial.agent_dir / "oracle.txt").exists()  # the mounted log folder survives the stop
    assert not env.root.exists()


def write_trial_config(env: LocalEnvironment, agent_name: str, **fields: Any) -> None:
    """The config.json Harbor writes into the trial folder before it starts an environment."""
    config = {"agent": {"name": agent_name, "import_path": None}, **fields}
    env.trial_paths.config_path.write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("agent_name", ["claude-code", "terminus-2", "probe_installed:ProbeInstalled"])
def test_start_refuses_an_agent_that_could_run_model_written_code(tmp_path: Path, agent_name: str) -> None:
    env = make_env(tmp_path)
    write_trial_config(env, agent_name)
    with pytest.raises(ValueError, match="only agents that execute nothing a model wrote"):
        run(env.start(force_build=False))
    assert not env.root.exists()


@pytest.mark.parametrize(
    ("agent_name", "fields"),
    [*((name, {}) for name in SAFE_AGENTS), ("claude-code", {"source_trial": {"action": "regrade"}})],
)
def test_start_allows_the_safe_agents_and_any_regrade(tmp_path: Path, agent_name: str, fields: dict) -> None:
    env = make_env(tmp_path)
    write_trial_config(env, agent_name, **fields)
    run(env.start(force_build=False))
    assert (env.root / "app" / "output").is_dir()


def test_start_takes_back_the_write_permission_harbor_gives_everyone(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    env.trial_paths.chmod_dir()  # what Harbor does to a trial once it has built its environment
    run(env.start(force_build=False))
    run(env.empty_dirs(["/logs/verifier"]))
    run(env.ensure_dirs(["/app/output/sub"]))
    for folder in (env.trial_paths.trial_dir, env.trial_paths.verifier_dir, env.host_path("/app/output/sub")):
        assert folder.stat().st_mode & 0o022 == 0, folder


def test_start_gives_a_fresh_environment_folder(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    env.host_path("/app/output/answer.json").write_text("from an earlier attempt", encoding="utf-8")
    run(env.start(force_build=False))
    assert not env.host_path("/app/output/answer.json").exists()


def test_a_download_copies_links_as_links(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    host_file = tmp_path / "host.txt"
    host_file.write_text("host data", encoding="utf-8")
    (env.host_path("/app/output") / "link").symlink_to(host_file)
    (env.host_path("/app/output") / "dangling").symlink_to(tmp_path / "nowhere")
    run(env.download_dir("/app/output", tmp_path / "out"))
    assert (tmp_path / "out" / "link").is_symlink() and (tmp_path / "out" / "dangling").is_symlink()


def test_start_lays_out_what_the_image_bakes_in(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    for folder in ("app/output", "tests", "solution", "logs"):
        assert (env.root / folder).is_dir()
    run(env.stop(delete=False))
    assert env.root.exists()


def test_exec_puts_the_venv_first_and_points_the_scripts_at_host_paths(tmp_path: Path) -> None:
    env = make_env(tmp_path, path="/opt/extra")
    run(env.start(force_build=False))
    script = 'echo "$PATH"; command -v python3; echo "$TESTS_DIR $ANSWER_PATH $VERIFIER_DIR $OUTPUT_DIR"; pwd'
    result = run(env.exec(script, env={"FOO": "bar"}))
    path, python3, variables, cwd = result.stdout.splitlines()
    assert path.startswith(f"/opt/extra{os.pathsep}{VENV_BIN}{os.pathsep}")
    assert python3 == str(VENV_BIN / "python3")
    assert variables.split() == [
        str(env.root / "tests"),
        str(env.root / "app" / "output" / "answer.json"),
        str((tmp_path / "trial" / "verifier").resolve()),
        str(env.root / "app" / "output"),
    ]
    assert Path(cwd).resolve() == env.root / "app"
    assert run(env.exec('echo "$FOO"', env={"FOO": "bar"})).stdout == "bar\n"


def test_exec_reports_the_exit_code_and_both_streams(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    result = run(env.exec("echo out; echo err >&2; exit 3"))
    assert (result.return_code, result.stdout, result.stderr) == (3, "out\n", "err\n")


def test_exec_kills_a_command_that_times_out(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        run(env.exec("sleep 30", timeout_sec=1))
    assert time.monotonic() - start < 10


def test_the_verifier_runs_the_shipped_test_sh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_anthropic(monkeypatch)
    env = make_env(tmp_path)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(gold_envelope(F4)), encoding="utf-8")

    async def verify() -> int:
        await env.start(force_build=False)
        await env.upload_dir(F4 / "tests", "/tests")
        await env.upload_file(answer, "/app/output/answer.json")
        judge = {"JUDGE_MODEL": "claude-sonnet-5-5", "JUDGE_REGRADE_MODEL": "", "JUDGE_SAMPLES": "3"}
        return (
            await env.exec("(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1", env=judge)
        ).return_code

    assert run(verify()) == 0
    reward = json.loads((tmp_path / "trial" / "verifier" / "reward.json").read_text(encoding="utf-8"))
    assert reward == {"reward": 1.0, "core": 1.0, "judged": 1.0}


# --------------------------------------------------------------------------------- SingleShotAgent


class FakeEnvironment:
    """Records what the agent uploads."""

    def __init__(self) -> None:
        self.uploads: dict[str, str] = {}

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = Path(source_path).read_text(encoding="utf-8")


def completion(text: str, *, refusal: bool = False) -> Completion:
    return Completion(
        text=text,
        parsed=None,
        refusal=refusal,
        refusal_category="bio" if refusal else None,
        model="claude-sonnet-5-5",
        fallback_used=False,
        stop_reason="refusal" if refusal else "end_turn",
        input_tokens=1200,
        output_tokens=300,
        cost_usd=0.0054,
        latency_s=1.5,
    )


class FakeModel:
    """Stands in for llm.complete: records each call and answers with the next queued reply."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[Completion | Exception] = []

    def complete(self, spec: llm.RoleSpec, messages: str, **kwargs: Any) -> Completion:
        self.calls.append({"spec": spec, "messages": messages, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    fake = FakeModel()
    monkeypatch.setattr(llm, "complete", fake.complete)
    return fake


def agent(tmp_path: Path, **kwargs: Any) -> SingleShotAgent:
    logs = tmp_path / "agent"
    logs.mkdir(exist_ok=True)
    return SingleShotAgent(logs_dir=logs, model_name="claude-sonnet-5-5", **kwargs)


def test_the_agent_sends_the_instruction_once_and_uploads_the_reply(tmp_path: Path, model: FakeModel) -> None:
    model.replies.append(completion('{"answer": {"choice": "prmt7_cko"}, "rationale": "r"}'))
    environment, context = FakeEnvironment(), AgentContext()
    run(agent(tmp_path).run("the instruction", environment, context))

    [call] = model.calls
    assert (call["messages"], call["role"], call["spec"].model) == (
        "the instruction",
        "solver",
        "claude-sonnet-5-5",
    )
    assert "system" not in call and "schema" not in call
    assert environment.uploads == {ANSWER_PATH: '{"answer": {"choice": "prmt7_cko"}, "rationale": "r"}'}
    assert (context.n_input_tokens, context.n_output_tokens, context.cost_usd) == (1200, 300, 0.0054)
    assert context.model_usage["claude-sonnet-5-5"].n_output_tokens == 300
    assert context.metadata["model"] == "claude-sonnet-5-5"
    assert context.metadata["refusal"] is False
    assert "text" not in context.metadata


@pytest.mark.parametrize("text", ["I can't help with that.", ""])
def test_a_refusal_is_uploaded_too(tmp_path: Path, model: FakeModel, text: str) -> None:
    model.replies.append(completion(text, refusal=True))
    environment, context = FakeEnvironment(), AgentContext()
    run(agent(tmp_path).run("the instruction", environment, context))
    assert environment.uploads == {ANSWER_PATH: text}
    assert context.metadata["refusal"] is True
    assert context.metadata["refusal_category"] == "bio"


def test_a_failed_call_raises_harbors_api_error_and_uploads_nothing(tmp_path: Path, model: FakeModel) -> None:
    model.replies.append(llm.LLMError("anthropic claude-sonnet-5-5 call failed: overloaded"))
    environment = FakeEnvironment()
    with pytest.raises(ApiError, match="overloaded"):
        run(agent(tmp_path).run("the instruction", environment, AgentContext()))
    assert environment.uploads == {}
    policy = policy_for("ApiError")  # Harbor retries it under -r, and otherwise the reward is not valid
    assert (policy.should_retry, policy.has_valid_reward) == (True, False)


def test_the_upload_lands_in_the_local_environment(tmp_path: Path, model: FakeModel) -> None:
    model.replies.append(completion("not even json"))
    env = make_env(tmp_path)
    run(env.start(force_build=False))
    run(agent(tmp_path).run("the instruction", env, AgentContext()))
    assert (env.root / "app" / "output" / "answer.json").read_text(encoding="utf-8") == "not even json"


def test_harbor_builds_the_agent_from_its_import_path_with_the_effort(tmp_path: Path) -> None:
    config = AgentConfig(
        name="p2c_task_re.harbor_agent:SingleShotAgent",
        model_name="claude-sonnet-5-5",
        kwargs={"reasoning_effort": "medium"},
    )
    built = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(built, SingleShotAgent)
    assert built.spec.effort == "medium"
    with pytest.raises(ValueError, match="reasoning_effort"):
        SingleShotAgent(logs_dir=tmp_path, model_name="claude-sonnet-5-5", reasoning_effort="extreme")


def test_the_model_maps_to_a_role_spec() -> None:
    sonnet = solver_spec("claude-sonnet-5-5")
    assert (sonnet.provider, sonnet.effort, sonnet.price_per_mtok.input) == ("anthropic", "high", 2.0)
    assert solver_spec("anthropic/claude-sonnet-5-5", "low").effort == "low"
    assert solver_spec("gpt-6.1-sol").provider == "openai"
    unlisted = solver_spec("claude-haiku-5")
    assert (unlisted.provider, unlisted.price_per_mtok.input) == ("anthropic", None)
    for bad in (None, "", "gemini-3-pro"):
        with pytest.raises(ValueError):
            solver_spec(bad)


# ------------------------------------------------------------------------------- container recipe


def test_the_image_pins_the_judge_sdks_and_server_deps_from_uv_lock() -> None:
    locked = {
        package_["name"]: package_["version"]
        for package_ in tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))["package"]
    }
    recipe = (REPO / "containers" / "p2c-base.def").read_text(encoding="utf-8")
    for name in ("anthropic", "openai", "uvicorn", "fastapi"):
        assert f"{name}=={locked[name]}" in recipe


# ----------------------------------------------------------------------------- end to end (offline)


def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow the anthropic SDK for every python started from here on (grade.py under test.sh)."""
    monkeypatch.setenv("PYTHONPATH", f"{FAKE_SDK}{os.pathsep}{REPO / 'tests'}")


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under `root`, by relative path."""
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def harbor(*args: str | Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "ANTHROPIC_API_KEY": "offline", "OPENAI_API_KEY": "offline"}
    command = [str(HARBOR), *map(str, args)]
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=600, check=True)


def rewards_of(job_dir: Path) -> dict[str, dict[str, float]]:
    rewards = {}
    for result_path in job_dir.glob("*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["exception_info"] is None, result["exception_info"]
        rewards[result["task_name"].removeprefix("p2c/")] = result["verifier_result"]["rewards"]
    return rewards


@pytest.mark.skipif(not HARBOR.exists(), reason="the harbor CLI is not installed")
def test_harbor_runs_oracle_and_nop_and_regrades_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_anthropic(monkeypatch)
    dataset, jobs = EXAMPLES, tmp_path / "jobs"
    before = snapshot(dataset)
    one = {"reward": 1.0, "core": 1.0, "judged": 1.0}
    zero = {"reward": 0.0, "core": 0.0, "judged": 0.0}
    for agent_name in ("oracle", "nop"):
        harbor(
            "run", "-p", dataset, "-e", LOCAL, "-a", agent_name, "-o", jobs, "--job-name", agent_name, "-y"
        )
    harbor("job", "regrade", jobs / "nop", "-p", dataset, "-e", LOCAL, "-o", jobs, "--job-name", "regrade")

    task_ids = {task.name for task in TASKS}
    assert rewards_of(jobs / "oracle") == dict.fromkeys(task_ids, one)
    assert rewards_of(jobs / "nop") == dict.fromkeys(task_ids, zero)
    assert rewards_of(jobs / "regrade") == dict.fromkeys(task_ids, zero)
    for trial in (jobs / "oracle").glob("*__*/"):
        assert sorted(path.name for path in trial.iterdir() if path.is_dir()) == [
            "agent",
            "artifacts",
            "verifier",
        ]
        for folder in (trial, trial / "agent", trial / "verifier", trial / "artifacts"):
            assert folder.stat().st_mode & 0o022 == 0, f"{folder} is writable by others"
        grading = json.loads((trial / "verifier" / "grading.json").read_text(encoding="utf-8"))
        assert (grading["status"], grading["answer_format"]) == ("ok", "bare")
        # The verifier's grade.py records its own judge spend: 3 samples per criterion + 3 consistency.
        calls = 3 * (len(grading["criteria"]) + 1)
        assert grading["judge_usage"] == {
            "claude-sonnet-5-5": {
                "calls": calls,
                "input_tokens": 1000 * calls,
                "output_tokens": 40 * calls,
                "refusals": 0,
            }
        }
    assert snapshot(dataset) == before  # Harbor ran on the examples and wrote nothing into them
