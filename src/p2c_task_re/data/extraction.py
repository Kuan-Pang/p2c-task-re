"""papers2cells extraction JSONs → typed records, deduplicated by DOI (PLAN §3, §8 Corpus, §10 data/).

One `<uuid>.extraction.json` per bioRxiv archive holds `paper_metadata`, `table_1` (cell states),
`table_2` (perturbation triplets) and `table_3` (acronyms). The same paper can appear several times:
in both corpora (109 DOIs share immunology_top300 and cancer_top300) and as several archive versions
(v1, v2, … under different uuids). `load_corpus` keeps one file per DOI, keyed by `paper_id`.

Every text field is read tolerantly: missing or null → "", numbers → str, lists → "; "-joined.
"""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, computed_field

from p2c_task_re.data.meca import MecaIndex


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_as_text(v) for v in value if v is not None)
    return str(value).strip()


Text = Annotated[str, BeforeValidator(_as_text)]


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CellState(_Row):
    """A `table_1` row."""

    cell_state: Text = ""
    species: Text = ""
    context: Text = ""
    description: Text = ""
    text_location: Text = ""


class Triplet(_Row):
    """A `table_2` row (PLAN's PerturbationTriple): start state → perturbation → end state."""

    index: int  # row position in table_2, the stable id of a triplet within its paper
    starting_cell_state: Text = ""
    ending_cell_state: Text = ""
    species: Text = ""
    description_of_perturbation: Text = ""
    context: Text = ""
    perturbation_modality: Text = ""  # e.g. "chemical", "genetic + signaling"
    cell_assay: Text = ""
    evidence_source: Text = ""  # primary | reanalysis | reference
    delta: Text = ""
    mechanism: Text = ""
    mechanism_source: Text = ""  # explicitly_described | inferred_from_data | speculated | ""
    experiment_confidence: Text = ""  # validated | single_experiment | preliminary
    sample_size: Text = ""

    @computed_field
    @property
    def is_self_loop(self) -> bool:
        """Start and end state are the same label (case and whitespace ignored)."""
        return _fold(self.starting_cell_state) == _fold(self.ending_cell_state)


class Acronym(_Row):
    """A `table_3` row."""

    acronym: Text = ""
    definition: Text = ""
    category: Text = ""  # protocol | reagent | model | other
    note: Text = ""


class PaperExtraction(BaseModel):
    doi: str
    paper_id: str  # DOI suffix, e.g. 2026.05.31.727600 (PLAN §10 "Names, no hashes")
    title: str
    authors: list[str]
    journal: str
    url: str
    uuid: str  # filename stem = the .meca uuid
    corpus: str  # key of the extraction folder it came from, e.g. "immunology"
    path: Path
    cell_states: list[CellState]
    triplets: list[Triplet]
    acronyms: list[Acronym]
    extraction_warning: str = ""  # set by papers2cells when it emitted an empty fallback extraction
    version: int | None = None  # archive version from the .meca manifest, when known
    # The other files for this DOI, which load_corpus considered and dropped.
    alternatives: list[Path] = Field(default_factory=list)


def paper_id_from_doi(doi: str) -> str:
    """DOI suffix after the registrant prefix: `10.64898/2026.05.31.727600` → `2026.05.31.727600`."""
    prefix, slash, suffix = doi.strip().partition("/")
    if not (prefix.startswith("10.") and slash and suffix):
        raise ValueError(f"not a DOI: {doi!r}")
    return suffix


def load_extraction(path: Path, corpus: str) -> PaperExtraction:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("paper_metadata") or {}
    doi = _as_text(meta.get("doi"))
    authors = meta.get("authors") or []
    return PaperExtraction(
        doi=doi,
        paper_id=paper_id_from_doi(doi),
        title=_as_text(meta.get("title")),
        authors=[_as_text(a) for a in (authors if isinstance(authors, list) else [authors])],
        journal=_as_text(meta.get("journal")),
        url=_as_text(meta.get("url")),
        uuid=path.name.removesuffix(".extraction.json"),
        corpus=corpus,
        path=path,
        cell_states=[CellState.model_validate(row) for row in _rows(raw, "table_1")],
        triplets=[Triplet.model_validate({**row, "index": i}) for i, row in enumerate(_rows(raw, "table_2"))],
        acronyms=[Acronym.model_validate(row) for row in _rows(raw, "table_3")],
        extraction_warning=_as_text(raw.get("_extraction_warning")),
    )


def load_corpus(
    extraction_dirs: dict[str, Path],
    priority: list[str],
    meca_index: MecaIndex | None = None,
) -> dict[str, PaperExtraction]:
    """Load every `*.extraction.json` and keep one file per DOI, keyed by paper_id.

    Among the files for one DOI the kept one is, in order: the newest archive version (read from the
    .meca manifest; only when `meca_index` is given), then the most table_2 rows, then the corpus
    listed first in `priority`. The kept record carries its `version` and, in `alternatives`, the
    paths of the files it beat, so every choice can be audited.
    """
    missing = set(extraction_dirs) - set(priority)
    if missing:
        raise ValueError(f"corpora {sorted(missing)} are not ranked in priority={priority}")
    by_doi: dict[str, list[PaperExtraction]] = {}
    for name, folder in extraction_dirs.items():
        if not Path(folder).is_dir():  # a mistyped path would otherwise read as an empty corpus
            raise FileNotFoundError(f"extraction folder for {name!r} is not a folder: {folder}")
        for path in sorted(Path(folder).glob("*.extraction.json")):
            paper = load_extraction(path, name)
            if meca_index is not None and paper.uuid in meca_index:
                paper.version = meca_index.version(paper.uuid)
            by_doi.setdefault(paper.doi.casefold(), []).append(paper)

    corpus: dict[str, PaperExtraction] = {}
    for papers in by_doi.values():
        papers.sort(key=lambda p: _rank(p, priority))
        kept, *dropped = papers
        kept.alternatives = [p.path for p in dropped]
        if kept.paper_id in corpus:
            other = corpus[kept.paper_id].doi
            raise ValueError(f"paper_id {kept.paper_id} is shared by {other} and {kept.doi}")
        corpus[kept.paper_id] = kept
    return corpus


def _rank(paper: PaperExtraction, priority: list[str]) -> tuple[int, int, int, str]:
    """Sort key, best first: newest archive version, most table_2 rows, earliest corpus in priority.

    The path breaks any remaining tie so the choice never depends on glob order.
    """
    return (-(paper.version or 0), -len(paper.triplets), priority.index(paper.corpus), str(paper.path))


def _rows(raw: dict, table: str) -> list[dict]:
    rows = raw.get(table) or []
    return [row for row in rows if isinstance(row, dict)]


def _fold(s: str) -> str:
    return " ".join(s.casefold().split())
