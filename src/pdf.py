import logging
import pymupdf

from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Longest side of a rendered page, in pixels. A landscape page at 150 dpi is 22k
# patches for ColQwen, and the attention over those does not fit in any GPU here.
MAX_IMAGE_SIDE = 2048


def page_image_path(images_dir: Path, page: int) -> Path:
    """Physical 1-based page -> its PNG."""
    return Path(images_dir) / f"page_{page:03d}.png"


@dataclass(frozen=True)
class RenderedReport:
    page_count: int
    images_dir: Path


def pdf_to_image(pdf_path: Path, output_dir: Path, dpi: int = 150) -> RenderedReport:
    """Renders every page of the PDF into `output_dir/<pdf stem>/`."""
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise RuntimeError(f"not a PDF file: {pdf_path}")

    images_dir = Path(output_dir) / pdf_path.stem
    images_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    try:
        page_count = len(document)
        for index in range(page_count):
            page = document[index]
            fit = MAX_IMAGE_SIDE * 72 / max(page.rect.width, page.rect.height)
            page.get_pixmap(dpi=min(dpi, round(fit))).save(page_image_path(images_dir, index + 1))

        return RenderedReport(page_count=page_count, images_dir=images_dir)

    finally:
        document.close()


def discover_reports(dir: str | Path) -> list[Path]:
    """Every PDF in `dir`, sorted by name."""
    reports_dir = Path(dir)
    if not reports_dir.exists():
        return []

    return sorted(p for p in reports_dir.glob("*.pdf") if p.is_file())
