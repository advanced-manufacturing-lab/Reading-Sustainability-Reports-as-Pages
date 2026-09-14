"""Storage contract for multi-vector (ColQwen) page embeddings, partitioned by document."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Protocol, Any, Sequence

# Payload keys the store manages itself; user metadata may not use them.
TENANT_KEY = "doc_id"
RECORD_ID_KEY = "record_id"
RESERVED_KEYS = frozenset({TENANT_KEY, RECORD_ID_KEY})


@dataclass(frozen=True)
class EmbeddingRecord:
    doc_id: str                 # the Qdrant tenant; scopes every search
    id: str                     # unique within the document, e.g. "page_012"
    embedding: np.ndarray       # (num_patches, dim): one vector per visual patch
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id or not self.id:
            raise ValueError("`doc_id` and `id` must be non-empty")
        collisions = RESERVED_KEYS & self.metadata.keys()
        if collisions:
            raise ValueError(f"metadata uses reserved key(s) {sorted(collisions)}")


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    id: str
    score: float
    metadata: dict[str, Any]


class EmbeddingStore(Protocol):
    """
    Every read takes `doc_id` as a required first argument: a forgotten filter
    would silently scan the whole corpus. Writes are idempotent per (doc_id, id).
    """

    def save_many(self, records: Sequence[EmbeddingRecord]) -> None: ...

    def search(self, doc_id: str, query_embedding: np.ndarray, limit: int = 10) -> list[SearchHit]:
        """Pages of one document most similar to the query (MaxSim), best first."""
        ...

    def count(self, doc_id: str | None = None) -> int:
        """Pages in the document, or in the whole collection when `doc_id` is None."""
        ...

    def close(self) -> None: ...


def validate_embedding(embedding: np.ndarray) -> np.ndarray:
    if embedding.ndim != 2:
        raise ValueError(f"embedding must have shape (num_patches, dim); got ndim={embedding.ndim}")
    if embedding.shape[0] == 0:
        raise ValueError("embedding has no vectors (num_patches == 0)")
    return embedding
