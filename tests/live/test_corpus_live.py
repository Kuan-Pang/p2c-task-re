"""Live checks of the data readers against the real post_jan corpus on /lfs (PLAN §8 Corpus, §10 data/).

Reads the paths in configs/data/post_jan.yaml.
Run with `uv run pytest -m live tests/live/test_corpus_live.py`.
"""

import random
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from p2c_task_re.data.extraction import PaperExtraction, load_corpus, load_extraction
from p2c_task_re.data.jats import PaperText, parse_jats
from p2c_task_re.data.meca import MecaIndex
from p2c_task_re.data.splits import load_split_manifest, split_of

pytestmark = pytest.mark.live

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "post_jan.yaml"

# Papers whose JATS has no top-level Results section. Each is a structural defect of the archive's
# XML, not of the parser; a stage that needs Results text must skip them or read the nested headings.
KNOWN_WITHOUT_RESULTS = {
    "2026.02.03.703253": "results subsections nested inside Introduction",
    "2026.02.22.707300": "results subsections nested inside Introduction",
    "2026.02.24.707683": "results subsections nested inside Introduction",
    "2026.03.17.712280": "body is untitled paragraphs only",
    "2026.03.25.714276": "results under a top-level 'MAIN' section",
    "2026.06.11.731561": "short report under a single 'Main Text' section",
    "2026.06.20.733552": "Results and Discussion nested inside 'STAR★Methods'",
}


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG)


@pytest.fixture(scope="module")
def extraction_dirs(cfg) -> dict[str, Path]:
    return {name: Path(folder) for name, folder in cfg.extraction_dirs.items()}


@pytest.fixture(scope="module")
def all_files(extraction_dirs) -> list[PaperExtraction]:
    return [
        load_extraction(path, corpus)
        for corpus, folder in extraction_dirs.items()
        for path in sorted(folder.glob("*.extraction.json"))
    ]


@pytest.fixture(scope="module")
def meca_index(cfg) -> MecaIndex:
    return MecaIndex(Path(cfg.meca_root))


@pytest.fixture(scope="module")
def corpus(cfg, extraction_dirs, meca_index) -> dict[str, PaperExtraction]:
    return load_corpus(extraction_dirs, list(cfg.corpus_priority), meca_index)


@pytest.fixture(scope="module")
def parsed(corpus, meca_index) -> dict[str, PaperText]:
    return {pid: parse_jats(meca_index.open_jats(p.uuid).xml) for pid, p in sorted(corpus.items())}


def test_extraction_file_counts(all_files):
    assert Counter(p.corpus for p in all_files) == {"immunology": 300, "cancer": 300}


def test_doi_overlap_and_versions(all_files, corpus):
    corpora_by_doi: dict[str, set[str]] = {}
    uuids_by_doi: dict[str, set[str]] = {}
    for p in all_files:
        corpora_by_doi.setdefault(p.doi, set()).add(p.corpus)
        uuids_by_doi.setdefault(p.doi, set()).add(p.uuid)
    assert len(corpora_by_doi) == 455
    assert sum(len(c) == 2 for c in corpora_by_doi.values()) == 109
    assert sum(len(u) > 1 for u in uuids_by_doi.values()) == 28  # DOIs with several archive versions
    assert len({p.paper_id for p in all_files}) == 455  # DOI suffixes are distinct
    assert len(corpus) == 455


def test_every_paper_has_a_meca(all_files, meca_index):
    assert [p.uuid for p in all_files if p.uuid not in meca_index] == []


def test_dedupe_keeps_the_newest_version(corpus, all_files, meca_index):
    versions = {p.uuid: meca_index.version(p.uuid) for p in all_files}
    assert None not in versions.values()
    newest: dict[str, int] = {}
    for p in all_files:
        newest[p.paper_id] = max(newest.get(p.paper_id, 0), versions[p.uuid])
    assert {pid: p.version for pid, p in corpus.items()} == newest
    assert sum(len(p.alternatives) for p in corpus.values()) == 600 - 455


def test_split_counts(cfg, corpus):
    manifest = load_split_manifest(Path(cfg.split_manifest))
    assert len(manifest) == 428
    assert set(manifest) <= set(corpus)
    counts = Counter(split_of(pid, manifest) for pid in corpus)
    assert counts == {"train": 343, "dev": 35, "test": 50, "unassigned": 27}


def test_sampled_papers_have_results_and_legends(parsed):
    """30 papers sampled with a fixed seed; the known defects are pinned by the next test instead."""
    candidates = sorted(set(parsed) - set(KNOWN_WITHOUT_RESULTS))
    failures = []
    for pid in random.Random(0).sample(candidates, 30):
        kinds = {s.kind for s in parsed[pid].sections}
        if "results" not in kinds or "figure_legend" not in kinds:
            failures.append((pid, sorted(kinds)))
    assert failures == []


def test_papers_without_results_are_the_known_jats_defects(parsed):
    without_results = {pid for pid, paper in parsed.items() if not paper.results_text()}
    assert without_results == set(KNOWN_WITHOUT_RESULTS)
    assert [pid for pid, paper in parsed.items() if not paper.legends()] == []


def test_section_spans_start_with_their_headings(parsed):
    for pid, paper in parsed.items():
        for s in paper.sections:
            span = paper.text[s.start : s.end]
            assert span.strip() == span and span, (pid, s)
            if s.title:  # untitled sections (loose body text) have no heading line
                assert span.split("\n")[0] == s.title, (pid, s)
