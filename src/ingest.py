from __future__ import annotations

import json
import logging
import time

from dataclasses import asdict, dataclass
from pathlib import Path

from .embedding_store import EmbeddingStore
from .embedding import Embedder
from .logs import format_duration
from .pdf import page_image_path, pdf_to_image

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class IngestResult:
    manifest: Manifest
    rendered: bool      # False when the images were already on disk
    embedded: bool      # False when the store already held every page


@dataclass(frozen=True)
class Manifest:
    """What `ingest` leaves behind so `extract` can find a report again.

    Written to `<images_dir>/manifest.json`. `page_count` is the source of
    truth for how many images should exist, and `doc_id` ties the folder to
    its Qdrant tenant (the folder is named after the PDF stem, the tenant
    after the full file name).
    """

    doc_id: str           # Qdrant tenant == report_name, the gold standard's key
    report_name: str
    page_count: int
    dpi: int              # provenance: the resolution the VLM ends up seeing
    images_dir: Path

    def pages(self) -> range:
        return range(1, self.page_count + 1)

    def image_path(self, page: int) -> Path:
        return page_image_path(self.images_dir, page)

    def image_paths(self) -> list[Path]:
        return [self.image_path(p) for p in self.pages()]

    def images_complete(self) -> bool:
        """True when every page of the report is on disk (no truncated render)."""
        return self.page_count > 0 and all(p.exists() for p in self.image_paths())

    def write(self) -> None:
        payload = {**asdict(self), "images_dir": str(self.images_dir)}
        (self.images_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(images_dir: Path) -> Manifest | None:
    """The manifest in `images_dir`, or None when the folder holds no report."""
    images_dir = Path(images_dir)
    path = images_dir / MANIFEST_NAME
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))

    return Manifest(**{**data, "images_dir": images_dir})


def discover_ingested(pages_root: Path) -> list[Manifest]:
    """Every ingested report under `pages_root`, sorted by name.

    A subfolder counts as ingested once it has a manifest, which is how
    `extract` enumerates its work without touching the PDFs.
    """
    pages_root = Path(pages_root)
    if not pages_root.exists():
        return []

    manifests = []
    for folder in sorted(pages_root.iterdir()):
        if not folder.is_dir():
            continue
        manifest = read_manifest(folder)
        if manifest is not None:
            manifests.append(manifest)

    return sorted(manifests, key=lambda m: m.report_name)


def render_if_needed(pdf_path: Path, pages_root: Path, dpi: int) -> tuple[Manifest, bool]:
    """Renders the PDF to page images unless a complete render is already there.

    Returns the manifest and whether rendering actually happened.
    """
    manifest = read_manifest(Path(pages_root) / pdf_path.stem)
    if manifest is not None and manifest.images_complete():
        return manifest, False

    rendered = pdf_to_image(pdf_path=pdf_path,
                            output_dir=Path(pages_root),
                            dpi=dpi)

    manifest = Manifest(doc_id=pdf_path.name,
                        report_name=pdf_path.name,
                        page_count=rendered.page_count,
                        dpi=dpi,
                        images_dir=rendered.images_dir)

    manifest.write()
    return manifest, True


def ingest_report(pdf_path: Path,
                  pages_root: Path,
                  store: EmbeddingStore,
                  embedder: Embedder,
                  dpi: int = 150,
                  batch_size: int = 2) -> IngestResult:
    """Renders, embeds and stores one report; skips whatever is already done."""
    started = time.monotonic()
    manifest, rendered = render_if_needed(pdf_path, pages_root, dpi)

    stored = store.count(manifest.doc_id)
    # Is doc_id already complete in EmbeddingStore?
    if stored == manifest.page_count:
        logger.info("%s: %d pages already in the store",
                    manifest.report_name, stored)

        return IngestResult(manifest, rendered, embedded=False)

    image_paths = manifest.image_paths()
    metadatas = [{ "page": p, "image_path": str(path) }
                 for p, path in zip(manifest.pages(), image_paths)]

    written = embedder.embed_and_store(doc_id=manifest.doc_id,
                                       image_paths=image_paths,
                                       store=store,
                                       ids=[p.stem for p in image_paths],
                                       metadatas=metadatas,
                                       batch_size=batch_size)

    if written != manifest.page_count:
        logger.warning("%s: wrote %d of %d pages",
                       manifest.report_name, written, manifest.page_count)

    logger.info("%s: ingested %d pages in %s",
                manifest.report_name,
                manifest.page_count,
                format_duration(time.monotonic() - started))

    return IngestResult(manifest, rendered, embedded=True)
