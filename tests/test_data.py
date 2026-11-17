"""Offline tests for the data readers (PLAN §8, §10 data/) and the quote locator (PLAN §3).

Fixtures in tests/fixtures/data/:
- immunology/<uuid>.extraction.json: a real extraction (10.64898/2026.05.31.727600) cut to its
  4 triplets (index 3 is a self-loop), 2 cell states and 3 acronyms, `_usage` removed.
- cancer/<uuid>.extraction.json: synthetic, the same paper as if re-extracted in the cancer corpus
  with only the first 3 triplets, for the DOI dedupe.
- meca/manifest.xml + meca/999001.xml: a made-up article in the real bioRxiv MECA/JATS layout; the
  `make_meca` helper zips them into a .meca.
- split_manifest.json: four entries in the frozen manifest's shape, one DOI listed twice.
"""

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from p2c_task_re.data.extraction import (
    PaperExtraction,
    Triplet,
    load_corpus,
    load_extraction,
    paper_id_from_doi,
)
from p2c_task_re.data.jats import PaperText, Section, parse_jats, section_kind
from p2c_task_re.data.meca import MecaIndex
from p2c_task_re.data.splits import load_split_manifest, split_of
from p2c_task_re.text import locate_quote, normalize, section_at

UUID = "0132fc28-9526-1014-824f-cd6aa4d4562a"
OTHER_UUID = "96ea2115-6dbc-1014-a7be-e11e44a67de8"
EXTRACTION = f"{UUID}.extraction.json"


@pytest.fixture
def data_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "data"


@pytest.fixture
def jats_xml(data_dir: Path) -> bytes:
    return (data_dir / "meca" / "999001.xml").read_bytes()


@pytest.fixture
def paper(jats_xml: bytes) -> PaperText:
    return parse_jats(jats_xml)


def make_meca(
    path: Path,
    data_dir: Path,
    version: int = 2,
    manifest: str | None = None,
    members: dict[str, bytes] | None = None,
) -> Path:
    """Zip the fixture article into a .meca at `path`, its manifest article id set to v`version`.

    `manifest=""` leaves manifest.xml out; `members` replaces the JATS member(s).
    """
    if manifest is None:
        manifest = (data_dir / "meca" / "manifest.xml").read_text().replace("999001v2", f"999001v{version}")
    if members is None:
        members = {"content/999001.xml": (data_dir / "meca" / "999001.xml").read_bytes()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/meca")
        if manifest:
            archive.writestr("manifest.xml", manifest)
        archive.writestr("content/999001.pdf", b"%PDF-1.4 not really")
        for name, data in members.items():
            archive.writestr(name, data)
    return path


# --- extraction.py -------------------------------------------------------------------------------


def test_paper_id_from_doi():
    assert paper_id_from_doi("10.64898/2026.05.31.727600") == "2026.05.31.727600"
    assert paper_id_from_doi(" 10.1101/2025.12.30.700001 ") == "2025.12.30.700001"
    for bad in ("2026.05.31.727600", "10.64898/", "doi:10.1/x"):
        with pytest.raises(ValueError):
            paper_id_from_doi(bad)


def test_load_extraction(data_dir: Path):
    paper = load_extraction(data_dir / "immunology" / EXTRACTION, "immunology")
    assert paper.doi == "10.64898/2026.05.31.727600"
    assert paper.paper_id == "2026.05.31.727600"
    assert paper.uuid == UUID
    assert paper.corpus == "immunology"
    assert paper.title.startswith("Anti-Melanoma Differentiation-Associated protein 5")
    assert paper.authors[0] == "Silvia Calandra"
    assert paper.journal == "bioRxiv"
    assert [t.index for t in paper.triplets] == [0, 1, 2, 3]
    assert [t.is_self_loop for t in paper.triplets] == [False, False, False, True]
    assert paper.triplets[0].evidence_source == "primary"
    assert "μg/ml" in paper.triplets[0].description_of_perturbation
    assert [c.cell_state for c in paper.cell_states] == ["Lung fibroblast", "Pro-fibrotic lung fibroblast"]
    assert [a.acronym for a in paper.acronyms] == ["RTCA", "NCI", "CI"]
    assert paper.version is None and paper.alternatives == []
    assert "is_self_loop" in paper.triplets[3].model_dump()


def test_triplet_tolerates_missing_null_and_odd_values():
    t = Triplet.model_validate(
        {
            "index": 7,
            "starting_cell_state": "  CD8+ T cell ",
            "ending_cell_state": "cd8+  T cell",
            "delta": ["IFN-γ up", None, "TNF up"],
            "sample_size": 3,
            "mechanism_source": None,
            "unexpected_field": "ignored",
        }
    )
    assert t.starting_cell_state == "CD8+ T cell"
    assert t.delta == "IFN-γ up; TNF up"
    assert t.sample_size == "3"
    assert t.mechanism_source == ""
    assert t.description_of_perturbation == ""  # missing
    assert t.is_self_loop  # case and whitespace are ignored


def test_load_extraction_of_an_empty_fallback(tmp_path: Path):
    path = tmp_path / f"{UUID}.extraction.json"
    raw = {
        "paper_metadata": {"doi": "10.64898/2026.01.01.999001", "title": "T", "authors": None},
        "table_1": [],
        "table_2": [],
        "table_3": None,
        "_extraction_warning": "Fallback empty extraction emitted",
    }
    path.write_text(json.dumps(raw))
    paper = load_extraction(path, "immunology")
    assert paper.triplets == [] and paper.acronyms == [] and paper.authors == []
    assert paper.extraction_warning.startswith("Fallback")


def _corpus_dirs(tmp_path: Path, data_dir: Path, immunology: list[str], cancer: list[str]) -> dict[str, Path]:
    """Copy fixture files into tmp corpus folders: each entry is "<fixture corpus>:<target uuid>"."""
    dirs = {"immunology": tmp_path / "immunology", "cancer": tmp_path / "cancer"}
    for corpus, entries in (("immunology", immunology), ("cancer", cancer)):
        dirs[corpus].mkdir()
        for entry in entries:
            source, uuid = entry.split(":")
            shutil.copy(data_dir / source / EXTRACTION, dirs[corpus] / f"{uuid}.extraction.json")
    return dirs


def test_dedupe_prefers_more_rows_over_priority(data_dir: Path):
    dirs = {"immunology": data_dir / "immunology", "cancer": data_dir / "cancer"}
    corpus = load_corpus(dirs, priority=["cancer", "immunology"])
    assert list(corpus) == ["2026.05.31.727600"]
    kept = corpus["2026.05.31.727600"]
    assert (kept.corpus, len(kept.triplets)) == ("immunology", 4)
    assert kept.alternatives == [data_dir / "cancer" / EXTRACTION]


def test_dedupe_ties_go_to_the_corpus_earlier_in_priority(tmp_path: Path, data_dir: Path):
    dirs = _corpus_dirs(tmp_path, data_dir, [f"immunology:{UUID}"], [f"immunology:{UUID}"])
    assert load_corpus(dirs, ["immunology", "cancer"])["2026.05.31.727600"].corpus == "immunology"
    kept = load_corpus(dirs, ["cancer", "immunology"])["2026.05.31.727600"]
    assert kept.corpus == "cancer"
    assert kept.alternatives == [dirs["immunology"] / EXTRACTION]


def test_dedupe_prefers_the_newest_archive_version(tmp_path: Path, data_dir: Path):
    # v1 archive (UUID) has 4 rows; v2 archive (OTHER_UUID) has 3 rows, in the lower-priority corpus.
    dirs = _corpus_dirs(tmp_path, data_dir, [f"immunology:{UUID}"], [f"cancer:{OTHER_UUID}"])
    make_meca(tmp_path / "mecas" / "May_2026" / f"{UUID}.meca", data_dir, version=1)
    make_meca(tmp_path / "mecas" / "June_2026" / f"{OTHER_UUID}.meca", data_dir, version=2)
    index = MecaIndex(tmp_path / "mecas")

    kept = load_corpus(dirs, ["immunology", "cancer"], meca_index=index)["2026.05.31.727600"]
    assert (kept.uuid, kept.version, len(kept.triplets)) == (OTHER_UUID, 2, 3)
    assert kept.alternatives == [dirs["immunology"] / EXTRACTION]

    without_versions = load_corpus(dirs, ["immunology", "cancer"])["2026.05.31.727600"]
    assert (without_versions.uuid, without_versions.version) == (UUID, None)


def test_load_corpus_rejects_an_unranked_corpus(data_dir: Path):
    with pytest.raises(ValueError, match="not ranked"):
        load_corpus({"immunology": data_dir / "immunology", "cancer": data_dir / "cancer"}, ["immunology"])


def test_a_missing_data_folder_is_an_error_not_an_empty_corpus(tmp_path: Path, data_dir: Path):
    with pytest.raises(FileNotFoundError, match="cancer"):
        dirs = {"immunology": data_dir / "immunology", "cancer": tmp_path / "typo"}
        load_corpus(dirs, ["cancer", "immunology"])
    with pytest.raises(FileNotFoundError, match="meca_root"):
        MecaIndex(tmp_path / "typo")


def test_paper_extraction_round_trips_through_json(data_dir: Path):
    paper = load_extraction(data_dir / "immunology" / EXTRACTION, "immunology")
    assert PaperExtraction.model_validate_json(paper.model_dump_json()) == paper


# --- splits.py -----------------------------------------------------------------------------------


def test_split_manifest_lookup(data_dir: Path):
    manifest = load_split_manifest(data_dir / "split_manifest.json")
    assert manifest == {
        "2026.05.31.727600": "train",  # listed twice (v1, v2), same split
        "2026.01.01.999001": "dev",
        "2025.12.30.700001": "test",
    }
    assert split_of("2026.05.31.727600", manifest) == "train"
    assert split_of("2026.12.31.000000", manifest) == "unassigned"


def test_split_manifest_conflict_raises(tmp_path: Path, data_dir: Path):
    raw = json.loads((data_dir / "split_manifest.json").read_text())
    raw["papers"][1]["split"] = "test"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="both 'train' and 'test'"):
        load_split_manifest(path)


# --- meca.py -------------------------------------------------------------------------------------


def test_meca_index_lookup_and_open_jats(tmp_path: Path, data_dir: Path, jats_xml: bytes):
    path = make_meca(tmp_path / "May_2026" / f"{UUID}.meca", data_dir)
    make_meca(tmp_path / "April_2019" / "Batch_#119" / f"{OTHER_UUID}.meca", data_dir, version=1)
    index = MecaIndex(tmp_path)
    assert len(index) == 2 and UUID in index and "nope" not in index
    assert index.path(UUID) == path
    assert index.path(OTHER_UUID).parent.name == "Batch_#119"
    jats = index.open_jats(UUID)
    assert (jats.member, jats.version, jats.xml) == ("content/999001.xml", "999001v2", jats_xml)
    assert (index.version(UUID), index.version(OTHER_UUID)) == (2, 1)
    with pytest.raises(KeyError):
        index.path("nope")


def test_meca_index_prefers_the_newest_month_for_a_duplicate_uuid(tmp_path: Path, data_dir: Path):
    make_meca(tmp_path / "March_2026" / f"{UUID}.meca", data_dir, version=3)
    make_meca(tmp_path / "December_2025" / f"{UUID}.meca", data_dir, version=1)
    index = MecaIndex(tmp_path)
    assert index.path(UUID).parent.name == "March_2026"
    assert index.version(UUID) == 3


def test_open_jats_falls_back_to_the_single_content_xml(tmp_path: Path, data_dir: Path, jats_xml: bytes):
    make_meca(tmp_path / "May_2026" / f"{UUID}.meca", data_dir, manifest="")
    jats = MecaIndex(tmp_path).open_jats(UUID)
    assert (jats.member, jats.version, jats.xml) == ("content/999001.xml", None, jats_xml)


def test_open_jats_never_follows_a_path_outside_the_archive(tmp_path: Path, data_dir: Path):
    manifest = (data_dir / "meca" / "manifest.xml").read_text()
    manifest = manifest.replace("content/999001.xml", "../../etc/passwd")
    make_meca(tmp_path / "May_2026" / f"{UUID}.meca", data_dir, manifest=manifest)
    jats = MecaIndex(tmp_path).open_jats(UUID)
    assert (jats.member, jats.version) == ("content/999001.xml", "999001v2")


def test_open_jats_refuses_an_ambiguous_archive(tmp_path: Path, data_dir: Path):
    members = {"content/1.xml": b"<article/>", "content/2.xml": b"<article/>"}
    make_meca(tmp_path / "May_2026" / f"{UUID}.meca", data_dir, manifest="", members=members)
    with pytest.raises(ValueError, match="no JATS XML"):
        MecaIndex(tmp_path).open_jats(UUID)


# --- jats.py -------------------------------------------------------------------------------------


def test_top_level_section_kinds_in_document_order(paper: PaperText):
    top = [(s.kind, s.title) for s in paper.sections if s.depth == 0]
    assert top == [
        ("title", "IL-2 drives Foxp3 expression in CD4+ T cells"),
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("results", "Results"),
        ("methods", "3. Materials and methods"),
        ("discussion", "Discussion"),
        ("figure_legend", "Figure 1."),
        ("figure_legend", "Supplementary Figure S1."),
    ]


def test_nested_sections_inherit_the_top_level_kind(paper: PaperText):
    nested = [(s.kind, s.depth, s.title) for s in paper.sections if s.depth > 0]
    assert nested == [
        ("results", 1, "IL-2 induces Foxp3 in a dose-dependent manner"),
        ("results", 1, "STAT5 is required"),
        ("methods", 1, "Cell culture"),
    ]


def test_every_section_span_starts_with_its_heading_line(paper: PaperText):
    for s in paper.sections:
        assert paper.text[s.start : s.end].split("\n")[0] == s.title
    results = next(s for s in paper.sections if s.kind == "results" and s.depth == 0)
    stat5 = next(s for s in paper.sections if s.title == "STAT5 is required")
    assert results.start < stat5.start < stat5.end == results.end  # a nested span lies inside its parent
    assert paper.text[results.end : results.end + 2] == "\n\n"  # blank line between top-level sections


def test_figure_legends_come_last_and_include_back_matter_figures(paper: PaperText):
    assert paper.legends() == [
        "Figure 1.\nIL-2 induces Foxp3.\n"
        "(A) Frequency of Foxp3+ cells after 72 h with IL-2 (mean ± SD, n = 3). CO2 was 5%.",
        "Supplementary Figure S1.\nGating strategy for CD4+CD25+ cells.",
    ]
    assert paper.text.endswith(paper.legends()[-1])
    assert "IL-2 induces Foxp3." not in paper.results_text()  # the legend is not left inside Results


def test_inline_markup_keeps_its_text(paper: PaperText):
    for kept in ("CD4+ T cells", "(Fig. 1A)", "TGF-β also", "IFN-γ", "naïve", "10–100 U/ml", "37°C"):
        assert kept in paper.text, kept
    assert "data are available at example.org/data." in paper.text


def test_citation_superscripts_and_back_matter_are_dropped(paper: PaperText):
    assert "depend on IL-2 for survival." in paper.text  # not "survival1,2."
    assert "remains to be tested." in paper.text  # not "tested [2–4]." nor "tested ."
    assert "(Smith et al., 2020)" in paper.text  # an author-year citation is ordinary text
    for dropped in ("Acknowledgements", "flow core", "Funding", "made-up grant", "References", "must not"):
        assert dropped not in paper.text, dropped


def test_accessors(paper: PaperText):
    assert paper.title == "IL-2 drives Foxp3 expression in CD4+ T cells"
    assert paper.abstract.startswith("Abstract\nInterleukin-2 (IL-2) signalling through STAT5")
    assert paper.results_text().startswith("Results\nWe first asked whether IL-2 alone is sufficient.")
    assert "STAT5 is required\nBlocking STAT5" in paper.results_text()
    assert "Cell culture\nCells were cultured at 37°C" in paper.methods_text()
    assert paper.discussion_text().startswith("Discussion\nIL-2 is sufficient")
    assert paper.introduction_text().startswith("Introduction\nRegulatory T cells")


def test_untitled_text_before_results_is_the_introduction():
    xml = (
        b"<article><front><article-meta><title-group><article-title>T</article-title></title-group>"
        b"</article-meta></front><body><p>Nature-style opening paragraph.</p>"
        b"<sec><title>Results</title><p>A result.</p></sec></body></article>"
    )
    paper = parse_jats(xml)
    kinds = [(s.kind, s.title) for s in paper.sections]
    assert kinds == [("title", "T"), ("introduction", ""), ("results", "Results")]
    assert paper.introduction_text() == "Nature-style opening paragraph."


def test_a_graphical_abstract_stays_in_the_abstract():
    xml = (
        b"<article><front><article-meta><abstract><p>Summary text.</p><sec><title>Graphical Abstract</title>"
        b"<fig id='ufig1'><caption><p>Cartoon of the model.</p></caption></fig></sec></abstract>"
        b"</article-meta></front><body/></article>"
    )
    paper = parse_jats(xml)
    assert paper.abstract == "Abstract\nSummary text.\nGraphical Abstract\nCartoon of the model."
    assert paper.legends() == []


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("Results", "results"),
        ("RESULTS AND DISCUSSION", "results"),
        ("Materials and methods", "methods"),
        ("2. Materials and Methods", "methods"),
        ("Methods", "methods"),
        ("STAR★Methods", "methods"),
        ("Discussion", "discussion"),
        ("Conclusions", "discussion"),
        ("Introduction", "introduction"),
        ("Background", "introduction"),
        ("Figure legends", "figure_legend"),
        ("Supplementary figure legends", "figure_legend"),
        ("Supporting information", "supplementary"),
        ("Supplementary Materials", "supplementary"),
        ("Data and materials availability", "other"),
        ("Funding", "other"),
        ("", "other"),
    ],
)
def test_section_kind(title: str, kind: str):
    assert section_kind(title) == kind


def test_malformed_xml_raises():
    with pytest.raises(ValueError, match="not well-formed"):
        parse_jats(b"<article><body>")


# --- text.py -------------------------------------------------------------------------------------


def test_normalize():
    assert normalize("  “IL\u20102”\u00a0\u2013\tdependent ‘Foxp3’ \n ") == "\"IL-2\" - dependent 'Foxp3'"
    assert normalize("10 µM, −5 °C, ﬁbroblast, CD8⁺") == "10 μM, -5 °C, fibroblast, CD8+"


def test_locate_quote_exact_and_missing():
    text = "Alpha beta. Gamma delta."
    assert locate_quote(text, "Gamma delta") == (12, 23)
    assert locate_quote(text, "  Gamma delta  ") == (12, 23)
    assert locate_quote(text, "gamma delta") is None  # case still matters
    assert locate_quote(text, "Gamma epsilon") is None
    assert locate_quote(text, "") is None
    assert locate_quote(text, " \n ") is None


def test_locate_quote_tolerates_whitespace_and_line_breaks():
    text = "The   treatment\nabolished the\teffect."
    start, end = locate_quote(text, "treatment abolished the effect")
    assert text[start:end] == "treatment\nabolished the\teffect"
    assert locate_quote(text, "treatmentabolished") is not None  # whitespace is ignored entirely


def test_locate_quote_tolerates_quotes_dashes_and_minus():
    text = "Cells were “primed” with anti–CD3 at −80 °C and the CD4⁺ ‘naïve’ pool grew."
    for quote, original in [
        ('were "primed" with anti-CD3', "were “primed” with anti–CD3"),
        ("at -80 °C", "at −80 °C"),
        ("the CD4+ 'naïve' pool", "the CD4⁺ ‘naïve’ pool"),
        ("the CD4+ 'nai\u0308ve' pool", "the CD4⁺ ‘naïve’ pool"),  # decomposed ï
    ]:
        span = locate_quote(text, quote)
        assert span is not None, quote
        assert text[span[0] : span[1]] == original


def test_locate_quote_maps_expanded_characters_back_to_the_original():
    text = "Dose 10 µM… then ﬁbroblasts grew."
    span = locate_quote(text, "10 μM... then fibroblasts")
    assert span is not None
    assert text[span[0] : span[1]] == "10 µM… then ﬁbroblasts"


def test_locate_quote_returns_the_first_occurrence():
    assert locate_quote("x-y and x–y", "x–y") == (8, 11)  # verbatim match wins
    assert locate_quote("x–y and x–y", "x-y") == (0, 3)  # else the first normalized one


def test_locate_quote_and_section_at_on_a_parsed_paper(paper: PaperText):
    span = locate_quote(paper.text, "STAT5 is required Blocking STAT5 with")  # heading + paragraph
    assert span is not None
    assert section_at(paper.sections, span[0]).title == "STAT5 is required"
    legend_at = paper.text.index("Gating strategy")
    assert section_at(paper.sections, legend_at).kind == "figure_legend"
    results = next(s for s in paper.sections if s.kind == "results" and s.depth == 0)
    assert section_at(paper.sections, results.start).title == "Results"  # parent before the first child
    assert section_at(paper.sections, results.end) is None  # the blank line between sections
    assert section_at(paper.sections, len(paper.text) + 5) is None


def test_section_at_prefers_the_innermost_section():
    sections = [
        Section(kind="results", title="Results", start=0, end=100, depth=0),
        Section(kind="results", title="A", start=10, end=50, depth=1),
        Section(kind="results", title="A.1", start=20, end=30, depth=2),
    ]
    assert [section_at(sections, i).title for i in (5, 15, 25, 60)] == ["Results", "A", "A.1", "Results"]
