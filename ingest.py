import logging
import time

from src.utils import load_config, load_env, store_from_config
from src.logs import configure_logging, format_duration
from src.ingest import ingest_report
from src.pdf import discover_reports

from src.embedding import ColQwenEmbedder

logger = logging.getLogger("experiment")


def main() -> int:
    configure_logging()
    load_env()

    cfg = load_config()

    # 1º Discorver reports (*.pdf) on default reports folder: /data/reports
    reports = discover_reports(cfg.extraction.reports_dir)

    if not reports:
        logger.error("no PDF to ingest in %s", cfg.extraction.reports_dir)
        return 1

    # 2º Converts to jpeg + embedding and stores in qdrant
    embedder = ColQwenEmbedder(model_name=cfg.extraction.col_extraction_model,
                               device=cfg.extraction.device)

    started = time.monotonic()

    with store_from_config(cfg.qdrant) as store:
        for n, pdf in enumerate(reports, 1):
            logger.info("[%d/%d] %s", n, len(reports), pdf.name)

            result = ingest_report(pdf, cfg.extraction.pages_dir, store, embedder,
                                   dpi=cfg.extraction.pdf_image_dpi,
                                   batch_size=cfg.extraction.embed_batch_size)

            logger.info("  %d pages — rendered=%s embedded=%s — %s",
                        result.manifest.page_count,
                        result.rendered,
                        result.embedded,
                        format_duration(time.monotonic() - started))

    logger.info("ingest done: %d report(s) in %s",
                len(reports),
                format_duration(time.monotonic() - started))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
