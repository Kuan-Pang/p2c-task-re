"""bioRxiv .meca archives: find one by uuid and read its JATS XML (PLAN §8, §10 data/).

A .meca is a zip holding manifest.xml, content/<id>.xml (JATS), the PDF, figure TIFFs and
sometimes content/supplements/. The manifest's article item names the JATS member and carries the
archive version in its id (`727600v2` → version 2). Archives sit under `<meca_root>/<Month_Year>/`,
older months one level deeper in `Batch_#N/` folders; the filename stem is the uuid that also names
the extraction JSON.

Archive contents are untrusted: members are read into memory by their exact name (nothing is
extracted to disk, no filesystem path is built from a member name) and reads are size-capped.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

_MANIFEST_NS = "{http://manuscriptexchange.org}"
_MAX_MEMBER_BYTES = 64 * 1024 * 1024  # the largest corpus JATS is ~0.3 MB
_VERSION = re.compile(r"v(\d+)$")
_CONTENT_XML = re.compile(r"content/[^/]+\.xml")


class JatsFile(BaseModel):
    xml: bytes
    member: str  # zip member the XML was read from, e.g. "content/727600.xml"
    version: str | None  # manifest article item id, e.g. "727600v1"; None if the manifest is unusual


class MecaIndex:
    """uuid → .meca path, built by globbing the month folders once."""

    def __init__(self, meca_root: Path) -> None:
        self.root = Path(meca_root)
        if not self.root.is_dir():  # a mistyped path would otherwise read as an empty archive
            raise FileNotFoundError(f"meca_root is not a folder: {self.root}")
        paths = [*self.root.glob("*/*.meca"), *self.root.glob("*/*/*.meca")]
        # A uuid present in several month folders resolves to the newest month.
        paths.sort(key=lambda p: (_month(p.relative_to(self.root).parts[0]), str(p)))
        self._paths = {p.stem: p for p in paths}

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, uuid: str) -> bool:
        return uuid in self._paths

    def path(self, uuid: str) -> Path:
        try:
            return self._paths[uuid]
        except KeyError:
            raise KeyError(f"no .meca for uuid {uuid} under {self.root}") from None

    def open_jats(self, uuid: str) -> JatsFile:
        with zipfile.ZipFile(self.path(uuid)) as archive:
            href, item_id = _article_item(archive)
            names = set(archive.namelist())
            if href not in names:
                candidates = sorted(n for n in names if _CONTENT_XML.fullmatch(n))
                if len(candidates) != 1:
                    raise ValueError(f"{self.path(uuid)}: no JATS XML in manifest and {candidates=}")
                href = candidates[0]
            return JatsFile(xml=_read_member(archive, href), member=href, version=item_id)

    def version(self, uuid: str) -> int | None:
        """Archive version number from the manifest (`727600v2` → 2), or None if absent."""
        with zipfile.ZipFile(self.path(uuid)) as archive:
            _, item_id = _article_item(archive)
        match = _VERSION.search(item_id or "")
        return int(match.group(1)) if match else None


def _article_item(archive: zipfile.ZipFile) -> tuple[str | None, str | None]:
    """(JATS member href, article item id) from manifest.xml; (None, None) when it is unusable."""
    try:
        manifest = ET.fromstring(_read_member(archive, "manifest.xml"))
    except (KeyError, ET.ParseError):
        return None, None
    for item in manifest.iter(f"{_MANIFEST_NS}item"):
        if item.get("type") != "article":
            continue
        for instance in item.iter(f"{_MANIFEST_NS}instance"):
            if instance.get("media-type") == "application/xml":
                return instance.get("href"), item.get("id")
        return None, item.get("id")
    return None, None


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    with archive.open(name) as member:
        data = member.read(_MAX_MEMBER_BYTES + 1)
    if len(data) > _MAX_MEMBER_BYTES:
        raise ValueError(f"{name} is larger than {_MAX_MEMBER_BYTES} bytes")
    return data


def _month(folder: str) -> datetime:
    try:
        return datetime.strptime(folder, "%B_%Y")
    except ValueError:
        return datetime.min
