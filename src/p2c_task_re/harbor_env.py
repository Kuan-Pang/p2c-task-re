"""LocalEnvironment: a Harbor environment with no container, the `local` profile's runner (PLAN §9,
docs/task_format.md §8).

The ILC cluster has no docker, podman or apptainer and blocks unprivileged user namespaces, so for now a
trial runs on the host: a Harbor `BaseEnvironment` that gives each environment its own host folder standing
in for the container filesystem. The `singularity` and `cloud` profiles stay selectable for later, through
the packager's `--environment` and `harbor run -e`.

**No isolation.** Commands run on the host as the invoking user, with the host's environment variables,
network and filesystem; `user=` is ignored. That is acceptable only for no-tools tasks, where nothing the
model writes is executed: the oracle runs the task's own solve.sh, nop does nothing, and SingleShotAgent
calls the API from the host and uploads the reply as data. So `start` refuses every other agent
(`SAFE_AGENTS`): an installed agent (claude-code, codex, ...) would install software on the host, and an
agent like terminus would run model-written commands unsandboxed. A regrade runs no agent and is allowed.
The environment also refuses a task that names an image or a Dockerfile, gives the agent MCP servers or
skills, or asks for a restricted network (Harbor's own capability check: this environment cannot enforce
one). A `local` task folder carries environment/README.md only so that Harbor discovers it (a folder with
no docker_image needs an environment/ folder); nothing here reads it.

**Permissions.** Harbor opens a container's mounted folders to every user (mode 777) so that a container
user can write them. Here every command runs as the invoking user, so `start` takes group and other write
permission back off the trial folder, and folders Harbor asks this environment to create stay private.

**Path mapping.** Container paths under /app, /tests, /solution and /logs live in the environment's folder
`<root>/<session_id>/`. Harbor's log mounts put /logs/agent, /logs/verifier and /logs/artifacts straight
onto the trial's own folders (`auto_mounting`, on by default here), so nothing is copied back. Any other
absolute path is the host path itself (bash, python3, /tmp). The mapping is applied

    in file operations    upload_file/upload_dir/download_file/download_dir, is_dir/is_file
    in exec               the command string (container paths at a path boundary) and cwd (default /app)
    through variables     TESTS_DIR, ANSWER_PATH, VERIFIER_DIR, OUTPUT_DIR, which test.sh and solve.sh read
                          in place of their fixed paths (docs/task_format.md §7)

exec also puts the project venv's bin folder first on PATH, so the `python3` in test.sh is the venv's and
tests/grade.py finds the judge SDKs. One trial, with the defaults:

    <trial_dir>/agent/                     = /logs/agent      (Harbor's, mounted)
               verifier/                   = /logs/verifier   (Harbor's, mounted)
               artifacts/logs/artifacts/   = /logs/artifacts  (Harbor's, mounted)
               <trial>__env/               the agent environment: app/output/, solution/, tests/, logs/
               <trial>__verifier__trial/   the separate verifier environment (task.toml environment_mode)

stop(delete=True), Harbor's default, removes the environment's folder and leaves Harbor's.

Options (`--ek key=value`; `harbor job regrade` takes none, so the defaults must do):

    root=DIR              where the environment folders go (default: the trial's own folder)
    path=DIR[:DIR]        extra PATH entries, ahead of the venv's bin folder
    auto_mounting=false   keep /logs inside the environment folder and let Harbor copy it back

    uv run harbor run -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment -a oracle -y
    uv run harbor job regrade jobs/<job> -p examples/tasks -e p2c_task_re.harbor_env:LocalEnvironment

Both need ANTHROPIC_API_KEY and OPENAI_API_KEY set, whatever the agent (task.toml declares them), and
`harbor run` needs -y to accept that the verifier reads host variables (docs/task_format.md §8). Host paths
are pasted into command strings, so the jobs folder may not contain a space or shell metacharacter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import stat
import sys
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.options import EnvironmentOptions
from harbor.models.trial.paths import EnvironmentPaths
from pydantic import Field

__all__ = ["LocalEnvironment", "LocalEnvironmentOptions"]

# The container folders a task touches; each gets a subfolder of the environment folder.
CONTAINER_ROOTS = ("/app", "/logs", "/solution", "/tests")
WORKDIR = "/app"
# What containers/p2c-base.def bakes into the image: the answer artifact directory, so a nop trial still
# records it (as an empty directory) and stays regradable.
IMAGE_DIRS = ("/app/output",)
# Variables test.sh and solve.sh read in place of their fixed container paths (docs/task_format.md §7).
SCRIPT_PATHS = {
    "TESTS_DIR": "/tests",
    "VERIFIER_DIR": "/logs/verifier",
    "OUTPUT_DIR": "/app/output",
    "ANSWER_PATH": "/app/output/answer.json",
}
BUILD_FILES = ("Dockerfile", "docker-compose.yaml")  # environment/ files that ask for an image build
# The agents that execute nothing a model wrote, as `harbor run -a` names them (PLAN §9).
SAFE_AGENTS = ("oracle", "nop", "p2c_task_re.harbor_agent:SingleShotAgent")
GROUP_OTHER_WRITE = 0o022
VENV_BIN = Path(sys.executable).parent  # not resolved: the venv's python3 is a symlink out of the venv
# Host paths are pasted into command strings unquoted, so they may hold only characters a shell leaves alone.
SHELL_SAFE = re.compile(r"[\w./+:@%,=-]+")


class LocalEnvironmentOptions(EnvironmentOptions):
    """The `--ek` options of LocalEnvironment."""

    auto_mounting: bool = Field(
        default=True,
        description="Map /logs/agent, /logs/verifier and /logs/artifacts onto the trial's own folders.",
    )
    root: Path | None = Field(
        default=None,
        description="Folder that holds the per-environment folders; default: the trial's own folder.",
    )
    path: str = Field(
        default="",
        description="Extra PATH entries (colon-separated), ahead of the project venv's bin folder.",
    )


class LocalEnvironment(BaseEnvironment):
    """A trial's environment as a host folder, with container paths mapped onto it. No isolation."""

    options_model = LocalEnvironmentOptions
    options: LocalEnvironmentOptions

    def __init__(self, *args: Any, auto_mounting: bool = True, **kwargs: Any) -> None:
        # BaseEnvironment passes `auto_mounting` to the options explicitly, so the default lives here too.
        super().__init__(*args, auto_mounting=auto_mounting, **kwargs)
        self.root = (Path(self.options.root or self.trial_paths.trial_dir) / self.session_id).resolve()
        self._path_map = self._build_path_map()
        alternatives = "|".join(re.escape(str(target)) for target, _ in self._path_map)
        self._container_path = re.compile(rf"(?<![\w.~/-])({alternatives})(?=$|[/\s'\";&|<>()`:,*}}])")

    @staticmethod
    def type() -> str:
        return "p2c-local"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # No network control (a no-network or allowlist task is refused), no GPUs; the logs are mountable.
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        config = self.task_env_config
        builds = [name for name in BUILD_FILES if (self.environment_dir / name).exists()]
        if config.docker_image or builds:
            image = config.docker_image or f"environment/{builds[0]}"
            raise ValueError(
                f"{self.environment_name}: LocalEnvironment runs without a container, but the task names "
                f"{image}; package it with the local profile, or run it with that profile's environment"
            )
        if config.mcp_servers or config.skills_dir:
            raise ValueError(
                f"{self.environment_name}: LocalEnvironment has no isolation, so it runs only tasks that "
                "give the agent no tools (no MCP servers, no skills)"
            )

    # ------------------------------------------------------------------------------------ path mapping

    def _build_path_map(self) -> list[tuple[PurePosixPath, Path]]:
        """(container path, host path) pairs, deepest first so /logs/agent wins over /logs."""
        mapping = {PurePosixPath(root): self.root / root.lstrip("/") for root in CONTAINER_ROOTS}
        for mount in self._host_mounts:  # Harbor's log mounts when auto_mounting, plus any --mounts binds
            if mount.get("type") == "bind":
                mapping[PurePosixPath(mount["target"])] = Path(mount["source"]).resolve()
        for host in mapping.values():
            if not SHELL_SAFE.fullmatch(str(host)):
                raise ValueError(f"LocalEnvironment needs host paths a shell leaves alone, got {host}")
        return sorted(mapping.items(), key=lambda pair: len(pair[0].parts), reverse=True)

    def host_path(self, path: str | PurePath) -> Path:
        """The host path standing in for a container path; one outside the mapped folders is itself."""
        container = PurePosixPath(WORKDIR) / PurePosixPath(path)  # a relative path is relative to /app
        for target, host in self._path_map:
            if container.is_relative_to(target):
                return host.joinpath(*container.relative_to(target).parts)
        return Path(container)

    def host_command(self, command: str) -> str:
        """`command` with every container path in it replaced by its host path."""
        return self._container_path.sub(lambda match: str(self.host_path(match.group(1))), command)

    def _exec_env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = {**os.environ, **(self._merge_env(env) or {})}
        merged["PATH"] = os.pathsep.join(
            part for part in (self.options.path, str(VENV_BIN), merged.get("PATH", "")) if part
        )
        merged.update({name: str(self.host_path(path)) for name, path in SCRIPT_PATHS.items()})
        return merged

    # ------------------------------------------------------------------------------------- lifecycle

    async def start(self, force_build: bool) -> None:
        """Lay out a fresh environment folder: the container roots, what the image bakes in, the log
        folders. Refuses an agent outside SAFE_AGENTS first."""
        self._refuse_unsafe_agent()
        _drop_group_other_write(self.trial_paths.trial_dir)
        shutil.rmtree(self.root, ignore_errors=True)  # a fresh container: no output of an earlier attempt
        paths = EnvironmentPaths()
        for path in (*CONTAINER_ROOTS, *IMAGE_DIRS, paths.agent_dir, paths.verifier_dir, paths.artifacts_dir):
            self.host_path(path).mkdir(parents=True, exist_ok=True)

    def _refuse_unsafe_agent(self) -> None:
        """Harbor writes the trial's config.json before it starts any environment."""
        config_path = self.trial_paths.config_path
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("source_trial"):  # a regrade: no agent runs
            return
        agent = config.get("agent") or {}
        name = agent.get("import_path") or agent.get("name")
        if name not in SAFE_AGENTS:
            raise ValueError(
                f"LocalEnvironment has no isolation, so it runs only agents that execute nothing a model "
                f"wrote ({', '.join(SAFE_AGENTS)}), not {name!r}"
            )

    async def stop(self, delete: bool) -> None:
        if delete:
            await asyncio.to_thread(shutil.rmtree, self.root, ignore_errors=True)

    # Harbor asks for mode 777 so that a container's own user can write the folders; here there is one user.
    async def ensure_dirs(self, dirs: Any, *, chmod: bool = True) -> ExecResult | None:
        return await super().ensure_dirs(dirs, chmod=False)

    async def empty_dirs(self, dirs: Any, *, chmod: bool = True) -> ExecResult | None:
        return await super().empty_dirs(dirs, chmod=False)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run `command` with bash on the host, container paths mapped. `user` is ignored."""
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            self.host_command(command),
            cwd=self.host_path(cwd or self.task_env_config.workdir or WORKDIR),
            env=self._exec_env(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own process group, so a timeout kills everything it started
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        except (TimeoutError, asyncio.CancelledError) as stopped:  # timed out, or Harbor's own timeout
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            if isinstance(stopped, TimeoutError):
                raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
            raise
        if process.returncode != 0:
            self.logger.debug(f"exit {process.returncode}: {command[:200]}")
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode if process.returncode is not None else 1,
        )

    # ------------------------------------------------------------------------------------ file transfer

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await asyncio.to_thread(_copy_file, Path(source_path), self.host_path(target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await asyncio.to_thread(_copy_tree, Path(source_dir), self.host_path(target_dir))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await asyncio.to_thread(_copy_file, self.host_path(source_path), Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await asyncio.to_thread(_copy_tree, self.host_path(source_dir), Path(target_dir))

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return self.host_path(path).is_dir()

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return self.host_path(path).is_file()


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    """Copy the contents of `source` into `target`, merging, as `docker cp source/. target` does."""
    if not source.is_dir():
        raise FileNotFoundError(f"no such directory: {source}")
    if source.resolve() != target.resolve():
        # Links are copied as links: following one would read whatever host file it points at.
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)


def _drop_group_other_write(root: Path) -> None:
    """Take back the group and other write permission Harbor gave `root`'s folders (`chmod_dir`)."""
    for path in (root, *root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & GROUP_OTHER_WRITE:
                path.chmod(mode & ~GROUP_OTHER_WRITE)
