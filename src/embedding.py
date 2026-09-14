"""ColQwen2.5: page images and text queries -> multi-vector embeddings, streamed batch by batch."""

from __future__ import annotations

import logging
import time

from pathlib import Path
from typing import Any, Iterator, Sequence, Protocol

import numpy as np
import torch

from PIL import Image

from .embedding_store import EmbeddingRecord, EmbeddingStore
from .logs import format_duration

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"

# Pages per upsert while streaming into the store (~6MB of Python lists per page).
STORE_BATCH_SIZE = 2


class Embedder(Protocol):
    def embed_and_store(self,
                        doc_id: str,
                        image_paths: Sequence[Path],
                        store: EmbeddingStore,
                        ids: Sequence[str],
                        metadatas: Sequence[dict[str, Any]] | None = None,
                        batch_size: int = 2) -> int: ...

    def embed_queries(self, queries: Sequence[str], batch_size: int = 8) -> list[np.ndarray]: ...


class ColQwenEmbedder(Embedder):

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16) -> None:

        # Imported here so colpali-engine is only required when embedding.
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.device = device
        self.model = ColQwen2_5.from_pretrained(model_name, torch_dtype=dtype, device_map=self.device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    def _forward_images(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        # Converts Image to torch tensors (feature extraction)
        processed = self.processor.process_images(list(images)).to(self.model.device)
        with torch.no_grad():
            # forward pass
            out = self.model(**processed)

        embeddings = [e.to(torch.float32).cpu().numpy() for e in out]
        del processed, out

        # clear cache
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return embeddings

    def iter_embed(self, image_paths: Sequence[Path], batch_size: int = 2) -> Iterator[tuple[int, np.ndarray]]:
        """Yields (index, embedding of shape (num_patches, dim)) in input order, one
        batch decoded at a time."""

        paths = [Path(p) for p in image_paths]
        total = len(paths)

        started = time.monotonic()
        logger.info("embedding %d images on %s (batch_size=%d)", total, self.device, batch_size)

        done = 0
        for start in range(0, total, batch_size):
            images = [Image.open(p) for p in paths[start:start + batch_size]]
            try:
                embeddings = self._forward_images(images)
            finally:
                for im in images:
                    im.close()

            for offset, embedding in enumerate(embeddings):
                yield start + offset, embedding

            done += len(embeddings)
            elapsed = time.monotonic() - started
            logger.info("  %d/%d images — %.1fs/image — elapsed %s — eta %s",
                        done,
                        total,
                        elapsed / done,
                        format_duration(elapsed),
                        format_duration(elapsed / done * (total - done)))

        logger.info("embedded %d images in %s", total, format_duration(time.monotonic() - started))

    def embed_queries(self, queries: Sequence[str], batch_size: int = 8) -> list[np.ndarray]:
        """Text queries -> (num_query_tokens, dim) each, in the same space as the page embeddings."""
        queries = list(queries)
        if not queries:
            return []

        embeddings: list[np.ndarray] = []

        for start in range(0, len(queries), batch_size):
            processed = self.processor.process_queries(queries[start:start + batch_size]).to(self.model.device)
            with torch.no_grad():
                out = self.model(**processed)

            embeddings.extend(e.to(torch.float32).cpu().numpy() for e in out)

            del processed, out

        return embeddings

    def embed_and_store(
        self,
        doc_id: str,
        image_paths: Sequence[Path],
        store: EmbeddingStore,
        ids: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        batch_size: int = 2,
        store_batch_size: int = STORE_BATCH_SIZE,
    ) -> int:
        """Embeds a document's pages and writes them to the store every
        `store_batch_size` pages. Returns the number of pages written."""

        # len(image_paths) == len(ids) == len(metadatas)

        image_paths = list(image_paths)
        if len(ids) != len(image_paths):
            raise ValueError(f"{len(ids)} ids for {len(image_paths)} images")

        if metadatas is not None and len(metadatas) != len(image_paths):
            raise ValueError(f"{len(metadatas)} metadatas for {len(image_paths)} images")

        written = 0
        pending: list[EmbeddingRecord] = []

        def flush() -> None:
            nonlocal written
            if pending:
                store.save_many(pending)
                written += len(pending)
                pending.clear()

        for i, embedding in self.iter_embed(image_paths, batch_size=batch_size):
            pending.append(EmbeddingRecord(doc_id=doc_id,
                                           id=ids[i],
                                           embedding=embedding,
                                           metadata=dict(metadatas[i]) if metadatas is not None else {}))

            if len(pending) >= store_batch_size:
                flush()

        flush()

        logger.info("wrote %d pages of %r to the store", written, doc_id)
        return written
