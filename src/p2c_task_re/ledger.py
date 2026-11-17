"""Append-only spend ledger: one `calls.jsonl` line per model call (PLAN §3 cost controls, §10 run folder).

Each line holds call metadata only (role, models, tokens, cost, refusal, error), never prompt or response
text. The ledger is shared by concurrent callers (gates, grader, runner), so every write and read takes one
lock.
Re-opening an existing `calls.jsonl` (a resumed run) reloads it, so the budget covers the whole run.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class CallRecord(BaseModel):
    """One `calls.jsonl` line. `cost_usd` is None when the model's price is unknown (it then counts as 0)."""

    model_config = ConfigDict(extra="forbid")

    ts: str = Field(default_factory=_now)
    role: str
    tag: str | None = None
    provider: str
    model_requested: str
    model_used: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    refusal: bool = False
    refusal_category: str | None = None
    stop_reason: str | None = None
    fallback_requested: bool = False
    fallback_used: bool = False
    latency_s: float = 0.0
    error: str | None = None


class BudgetExceeded(RuntimeError):
    """Raised before a call once the run's recorded spend has reached its budget."""


class Ledger:
    """Thread-safe append-only call log with a spend cap. `path=None` keeps records in memory only."""

    def __init__(self, path: Path | None, budget_usd: float | None = None) -> None:
        self.path = path
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []
        if path is not None and path.exists():
            self._records = _read(path)

    def record(self, call: CallRecord) -> None:
        with self._lock:
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(call.model_dump_json() + "\n")
            self._records.append(call)

    def spent(self, role: str | None = None, tag: str | None = None) -> float:
        """Total known cost in USD, optionally restricted to one role and/or one tag."""
        with self._lock:
            return sum(
                r.cost_usd or 0.0
                for r in self._records
                if (role is None or r.role == role) and (tag is None or r.tag == tag)
            )

    def check_budget(self) -> None:
        """Raise `BudgetExceeded` once spend has reached the budget.

        Concurrent callers may each pass the check and overshoot by one call apiece.
        """
        if self.budget_usd is None:
            return
        spent = self.spent()
        if spent >= self.budget_usd:
            raise BudgetExceeded(f"spent ${spent:.4f} of the ${self.budget_usd:.2f} budget")


def _read(path: Path) -> list[CallRecord]:
    """Load an existing ledger, skipping a line truncated by a crash mid-write."""
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(CallRecord.model_validate(json.loads(line)))
        except ValueError:
            logger.warning("skipping unreadable line %d of %s", lineno, path)
    return records
