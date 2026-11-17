"""The frozen paper-family split manifest, read as data (PLAN §8 "Splits are deferred", §12 decision 5).

Every task records its paper's split as metadata: `train`, `dev`, `test`, or `unassigned` for
papers the manifest does not list. The manifest (from the MCQ benchmark build) lists papers by DOI,
one entry per archive version, so a DOI may appear more than once.

A DOI listed under two different splits raises instead of warning: the split guards against eval
papers leaking into RL training, so a contradictory manifest is corrupt input, and silently picking
one entry would mislabel every task built from that paper. The frozen manifest has no such entry.
"""

import json
from pathlib import Path

from p2c_task_re.data.extraction import paper_id_from_doi

UNASSIGNED = "unassigned"


def load_split_manifest(path: Path) -> dict[str, str]:
    """paper_id (DOI suffix) → "train" | "dev" | "test"."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    splits: dict[str, str] = {}
    for entry in raw["papers"]:
        paper_id = paper_id_from_doi(entry["doi"])
        split = entry["split"]
        if splits.get(paper_id, split) != split:
            raise ValueError(f"{path}: {entry['doi']} is listed as both {splits[paper_id]!r} and {split!r}")
        splits[paper_id] = split
    return splits


def split_of(paper_id: str, manifest: dict[str, str]) -> str:
    return manifest.get(paper_id, UNASSIGNED)
