"""
uv run main.py extract
uv run main.py extract --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time

from src.graph import build_graph, Context
from src.ingest import discover_ingested
from src.logs import configure_logging, format_duration
from src.output import OutputPaths, append_report, append_retrievals, new_run_id, write_run_config
from src.utils import AppConfig, load_config, load_env, store_from_config, VLMConfig
from src.vlm import VLMClient

logger = logging.getLogger("experiment")


def make_embedder(cfg: AppConfig):
    from src.embedding import ColQwenEmbedder  # torch: imported only when a model is needed
    return ColQwenEmbedder(model_name=cfg.extraction.col_extraction_model, device=cfg.extraction.device)


def make_vlm(v: VLMConfig) -> VLMClient:
    return VLMClient(model=v.model,
                     api_key=v.api_key,
                     max_tokens=v.max_tokens,
                     timeout=v.timeout,
                     max_retries=v.max_retries,
                     image_format=v.image_format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="retrieval only, no VLM calls")
    args = parser.parse_args()

    configure_logging()
    load_env()
    cfg = load_config()

    paths = OutputPaths(cfg.output.output_dir)
    if args.dry_run:
        paths = paths.for_dry_run()

    # 1º Get ingested reports data.
    #
    # {
    #   "doc_id": "Allianz_2022_report.pdf",
    #   "report_name": "Allianz_2022_report.pdf",
    #   "page_count": 164,
    #   "dpi": 150,
    #   "images_dir": "data/pages/Allianz_2022_report"
    # }
    reports_to_extract = discover_ingested(cfg.extraction.pages_dir)
    if not reports_to_extract:
        logger.error("nothing ingested under %s — run `ingest.py` first",
                     cfg.extraction.pages_dir)
        return 1

    logger.info("%d report(s) to extract%s",
                len(reports_to_extract), " — DRY RUN" if args.dry_run else "")

    # 2º Config VLM
    vlm = None
    if not args.dry_run:
        vlm = make_vlm(cfg.vlm)
        logger.info("VLM: model=%s concurrency=%d", vlm.model, cfg.vlm.concurrency)

    run_id = new_run_id()
    processed: list[str] = []
    started = time.monotonic()

    with store_from_config(cfg.qdrant) as store:
        graph = build_graph()
        embedder = make_embedder(cfg)
        context = Context(embedder=embedder,
                          embedding_store=store,
                          ai_client=vlm,
                          top_k=cfg.extraction.top_k,
                          is_dry_run=args.dry_run)

        for n, m in enumerate(reports_to_extract, 1):
            stored = store.count(m.doc_id)
            if stored == 0:
                logger.warning("[%d/%d] %s: tenant is empty — run `main.py ingest`; skipping",
                               n,
                               len(reports_to_extract),
                               m.report_name)
                continue

            if stored != m.page_count:
                logger.warning("%s: store has %d pages, manifest says %d",
                               m.report_name,
                               stored,
                               m.page_count)

            logger.info("[%d/%d] %s (%d pages)",
                        n,
                        len(reports_to_extract),
                        m.report_name,
                        m.page_count)

            result = graph.invoke({
                "doc_id": m.doc_id,
                "report_name": m.report_name,
                "images_dir": m.images_dir,
                "page_count": m.page_count,
                "query_embeddings": {},
                "retrievals": [],
                "calls": [],
                "records": [],
                "rows": []
            }, context=context, config={"max_concurrency": cfg.vlm.concurrency})

            written = append_report(paths, result["rows"])
            if args.dry_run:
                append_retrievals(paths, m.report_name, result["retrievals"])
            processed.append(m.report_name)

            positives = sum(1 for r in result["rows"] if r.llm_value is not None)
            elapsed = time.monotonic() - started
            logger.info("  %d cell(s), %d positive — %s elapsed, eta %s",
                        written,
                        positives,
                        format_duration(elapsed),
                        format_duration(elapsed / n * (len(reports_to_extract) - n)))

    config_path = write_run_config(paths,
                                   run_id,
                                   cfg.to_dict(),
                                   { "dry_run": args.dry_run, "reports": processed })

    logger.info("done: %d report(s), %s — %s — %s",
                len(processed),
                format_duration(time.monotonic() - started),
                paths.extractions_csv,
                config_path.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
