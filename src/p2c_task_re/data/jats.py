"""JATS XML → paper.txt plus section offsets (PLAN §1 "reported" text, §3 S1 fact sheet, §10 data/).

`parse_jats` renders one article as plain text in this order: the title, the abstract, the body
sections in document order (plus any Introduction/Results/Methods/Discussion/legend section a paper
puts in <back>), then one block per figure legend. Every paragraph is one line; every section starts
with its heading (label + title, e.g. "2. Materials and methods") on its own line; top-level
sections are separated by a blank line. `sections` gives each section's kind, heading and character
span in `text`, nested sections included (`depth` > 0, kind inherited from the top-level section).

Kept: the text inside inline markup (italic, bold, sup, sub, xref, ext-link, MathML), and Greek
letters and symbols exactly as written. A figure inside the abstract (a graphical abstract) stays in
the abstract. Dropped: references, acknowledgements, glossary, footnotes, back-matter statements
(funding, contributions, competing interests, data availability), images, and superscript citation
markers (a <sup> holding only bibr xrefs), which would otherwise glue reference numbers onto names
("CD8" + refs 1,2 → "CD81,2").

The XML comes from an archive and is untrusted: it is parsed with the stdlib expat parser, which
never fetches external DTDs or entities and (Python ≥ 3.12) limits entity-expansion amplification.
"""

import re
import xml.etree.ElementTree as ET
from typing import Literal

from pydantic import BaseModel

SectionKind = Literal[
    "title",
    "abstract",
    "introduction",
    "results",
    "methods",
    "discussion",
    "figure_legend",
    "supplementary",
    "other",
]

# First match wins, on the case-folded top-level title. "material" alone is not a Methods cue:
# every real Methods title also says "method", while "Supplementary Materials" and "Materials
# availability" do not.
_KIND_KEYWORDS: list[tuple[SectionKind, tuple[str, ...]]] = [
    ("other", ("availability",)),
    ("figure_legend", ("legend",)),
    ("results", ("result",)),
    ("methods", ("method", "experimental", "statistic")),
    ("discussion", ("discussion", "discusion", "conclu", "limitation")),
    ("introduction", ("introduction", "background")),
    ("supplementary", ("supplement", "supporting information", "extended data")),
]

# Back-matter sections kept in the text; every other <back> section is dropped (its figures are kept).
_BACK_KINDS = {"introduction", "results", "methods", "discussion", "figure_legend"}

_CITATION_FILLER = re.compile(r"[\s,;\[\]()\-–—]*")


class Section(BaseModel):
    """One section of paper.txt: `text[start:end]` begins with `title` when the title is non-empty."""

    kind: SectionKind
    title: str
    start: int
    end: int
    depth: int = 0


class PaperText(BaseModel):
    """paper.txt and its sections (written to papers/<paper_id>/{paper.txt, sections.json})."""

    text: str
    sections: list[Section]

    def section_text(self, section: Section) -> str:
        return self.text[section.start : section.end]

    def text_of(self, kind: SectionKind) -> str:
        """All top-level sections of one kind, headings included, separated by blank lines."""
        return "\n\n".join(self.section_text(s) for s in self.sections if s.depth == 0 and s.kind == kind)

    @property
    def title(self) -> str:
        return self.text_of("title")

    @property
    def abstract(self) -> str:
        return self.text_of("abstract")

    def introduction_text(self) -> str:
        return self.text_of("introduction")

    def results_text(self) -> str:
        return self.text_of("results")

    def methods_text(self) -> str:
        return self.text_of("methods")

    def discussion_text(self) -> str:
        return self.text_of("discussion")

    def legends(self) -> list[str]:
        """One string per figure legend, each starting with its label line ("Figure 1.")."""
        return [self.section_text(s) for s in self.sections if s.kind == "figure_legend" and s.depth == 0]


def section_kind(title: str) -> SectionKind:
    """Classify a top-level section by its title ("Results and Discussion" → results)."""
    folded = title.casefold()
    for kind, words in _KIND_KEYWORDS:
        if any(word in folded for word in words):
            return kind
    return "other"


def parse_jats(xml: bytes) -> PaperText:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as err:
        raise ValueError(f"not well-formed JATS XML: {err}") from err
    builder = _Builder()
    meta = root.find("front/article-meta")
    if meta is not None:
        builder.section_from_blocks("title", [_child_text(meta, "title-group/article-title")])
        for abstract in meta.findall("abstract"):
            builder.sec(abstract, "abstract", 0, figures_in_place=True, default_heading="Abstract")
    body = root.find("body")
    if body is not None:
        builder.body(body)
    back = root.find("back")
    if back is not None:
        builder.back(back)
    for floats in root.findall("floats-group"):
        builder.figures.extend(floats.iter("fig"))
    for fig in builder.figures:
        heading, blocks = _caption(fig)
        if blocks:
            builder.section_from_blocks("figure_legend", [heading, *blocks])
    return PaperText(text=builder.text(), sections=builder.finished_sections())


class _Builder:
    """Accumulates paper.txt one block (line) at a time and records section spans."""

    def __init__(self) -> None:
        self.figures: list[ET.Element] = []
        self._chunks: list[str] = []
        self._length = 0
        self._separator = ""
        self._records: list[dict] = []  # in opening order, so parents precede their children
        self._open: list[dict] = []

    def text(self) -> str:
        return "".join(self._chunks)

    def finished_sections(self) -> list[Section]:
        return [Section(**r) for r in self._records if r["start"] is not None]

    def block(self, s: str) -> None:
        s = _clean(s)
        if not s:
            return
        if self._length:
            self._write(self._separator)
        self._separator = "\n"
        for record in self._open:
            if record["start"] is None:
                record["start"] = self._length
        self._write(s)

    def open(self, kind: SectionKind, title: str, depth: int) -> dict:
        if depth == 0:
            self._separator = "\n\n"
        record = {"kind": kind, "title": title, "start": None, "end": None, "depth": depth}
        self._records.append(record)
        self._open.append(record)
        return record

    def close(self, record: dict) -> None:
        record["end"] = self._length
        self._open.remove(record)

    def section_from_blocks(self, kind: SectionKind, blocks: list[str]) -> None:
        """A top-level section whose first block is its heading (title, figure label)."""
        blocks = [b for b in map(_clean, blocks) if b]
        if not blocks:
            return
        record = self.open(kind, blocks[0], 0)
        for b in blocks:
            self.block(b)
        self.close(record)

    def sec(
        self,
        el: ET.Element,
        kind: SectionKind | None,
        depth: int,
        figures_in_place: bool = False,
        default_heading: str = "",
    ) -> None:
        """Render a <sec> (or <abstract>): heading line, then its content and nested sections."""
        title = _child_text(el, "title")
        heading = " ".join(part for part in (_child_text(el, "label"), title) if part) or default_heading
        kind = kind or section_kind(title)
        record = self.open(kind, heading, depth)
        self.block(heading)
        for child in el:
            if child.tag == "sec":
                self.sec(child, kind, depth + 1, figures_in_place)
            elif child.tag not in ("title", "label"):
                self.content(child, figures_in_place)
        self.close(record)

    def content(self, el: ET.Element, figures_in_place: bool) -> None:
        """Render one non-section child of a section as one or more blocks."""
        if el.tag in ("fig", "fig-group"):
            figures = list(el.iter("fig"))
        elif el.tag in ("table-wrap", "supplementary-material"):
            label, blocks = _caption(el)
            for b in [label, *blocks]:
                self.block(b)
            return
        elif el.tag == "list":
            figures = []
            for item in el.findall("list-item"):
                self.block(_inline(item, figures))
        else:
            figures = []
            self.block(_inline(el, figures))
        if not figures_in_place:
            self.figures.extend(figures)
            return
        for fig in figures:
            label, blocks = _caption(fig)
            for b in [label, *blocks]:
                self.block(b)

    def body(self, body: ET.Element) -> None:
        """Top-level <sec>s in order; loose content between them becomes an untitled section.

        Loose paragraphs right before the first <sec>, when that is Results, are the untitled
        introduction of Nature-style papers; any other loose content is kind "other".
        """
        loose: list[ET.Element] = []
        seen_sec = False
        for child in body:
            if child.tag != "sec":
                loose.append(child)
                continue
            opens_results = section_kind(_child_text(child, "title")) == "results"
            self.untitled(loose, "introduction" if opens_results and not seen_sec else "other")
            loose = []
            seen_sec = True
            self.sec(child, None, 0)
        self.untitled(loose, "other")

    def untitled(self, elements: list[ET.Element], kind: SectionKind) -> None:
        if not elements:
            return
        record = self.open(kind, "", 0)
        for el in elements:
            self.content(el, figures_in_place=False)
        self.close(record)

    def back(self, back: ET.Element) -> None:
        """Keep paper-content sections that some papers put in <back>; collect every figure."""
        for child in back:
            kind = section_kind(_child_text(child, "title"))
            if child.tag == "sec" and kind in _BACK_KINDS:
                self.sec(child, kind, 0)
            else:
                self.figures.extend(child.iter("fig"))

    def _write(self, s: str) -> None:
        self._chunks.append(s)
        self._length += len(s)


def _caption(el: ET.Element) -> tuple[str, list[str]]:
    """(label, caption blocks) of a figure, table or supplementary item; either may be empty."""
    label = _child_text(el, "label")
    caption = el.find("caption")
    if caption is None:
        return label, []
    parts = [c for c in caption if c.tag in ("title", "p")] or [caption]
    return label, [b for b in (_clean(_inline(p, [])) for p in parts) if b]


def _inline(el: ET.Element, figures: list[ET.Element]) -> str:
    """Text content of `el` and its inline children; nested <fig>s go to `figures`, citations vanish."""
    parts = [el.text or ""]
    for child in el:
        if child.tag == "fig":
            figures.append(child)
        elif child.tag == "break":
            parts.append(" ")
        elif _is_citation(child):
            parts = ["".join(parts).rstrip()]  # "survival <sup>1</sup>." → "survival."
        else:
            parts.append(_inline(child, figures))
        parts.append(child.tail or "")
    return "".join(parts)


def _is_citation(el: ET.Element) -> bool:
    """A superscript holding only reference xrefs and separators, e.g. <sup>[<xref>1</xref>]</sup>."""
    if el.tag != "sup" or len(el) == 0:
        return False
    if any(child.tag != "xref" or child.get("ref-type") != "bibr" for child in el):
        return False
    filler = (el.text or "") + "".join(child.tail or "" for child in el)
    return _CITATION_FILLER.fullmatch(filler) is not None


def _child_text(el: ET.Element, path: str) -> str:
    """Cleaned text of the first element at `path` under `el`, or ""."""
    child = el.find(path)
    return _clean(_inline(child, [])) if child is not None else ""


def _clean(s: str) -> str:
    return " ".join(s.split())
