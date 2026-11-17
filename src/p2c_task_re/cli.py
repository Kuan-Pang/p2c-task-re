"""The `p2c-pipeline` console script: Hydra composes the config, config.py validates it, run.py runs it
(PLAN §10 `cli.py`).

    uv run p2c-pipeline +experiment=eval eval.dataset=examples/tasks eval.agents=[oracle,nop]
    uv run p2c-pipeline +experiment=pilot --cfg job                # print the composed config, run nothing
    uv run p2c-pipeline +experiment=eval ... 'models.roles.judge=${models.gpt}'
    uv run p2c-pipeline +experiment=eval ... hydra.run.dir=outputs/runs/eval/<timestamp>   # resume

`config_path` is absolute (the repo's configs/): a relative one is resolved against the calling module and
breaks once the package runs as an installed console script. A bad config, a stage that is not built yet
or a spent budget ends the run with a one-paragraph message and exit code 1, logged to pipeline.log too.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from p2c_task_re.config import describe_errors, validate
from p2c_task_re.ledger import BudgetExceeded
from p2c_task_re.run import PipelineError, run_pipeline

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
# The SDKs' HTTP clients log every request at INFO; the ledger already records each call.
QUIET_LOGGERS = ("httpx", "httpx2")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="pipeline")
def main(cfg: DictConfig) -> None:
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        config = validate(cfg)
        run_pipeline(config, Path(HydraConfig.get().runtime.output_dir))
    except ValidationError as exc:
        _fail(f"invalid config ({exc.error_count()} problem(s)):\n{describe_errors(exc)}")
    except OmegaConfBaseException as exc:  # e.g. an interpolation to a key that does not exist
        _fail(f"invalid config: {exc}")
    except (PipelineError, BudgetExceeded) as exc:
        _fail(str(exc))


def _fail(message: str) -> None:
    log.error("p2c-pipeline: %s", message)
    sys.exit(1)
