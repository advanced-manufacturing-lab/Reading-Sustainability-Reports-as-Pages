"""`EmbeddingStore` on Qdrant: one collection, one tenant per report, MaxSim over multi-vectors."""

from __future__ import annotations

import logging
import uuid

import numpy as np

from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Datatype,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    KeywordIndexParams,
    KeywordIndexType,
    MatchValue,
    Memory,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from .embedding_store import RECORD_ID_KEY, TENANT_KEY, EmbeddingRecord, SearchHit, validate_embedding

logger = logging.getLogger(__name__)

# Changing this after a collection exists breaks every id already stored.
_ID_NAMESPACE = uuid.UUID("6f6a1d1e-2a1e-4f2c-9a4e-2c0f3b9a7d10")

# 20 ColQwen pages serialize to ~38MB of JSON and exceed the server's 32MB
# limit over HTTP. 8 fits comfortably; over gRPC it can go higher.
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class QdrantPreferences:
    """Defaults keep the vectors out of RAM: float16 on disk (ColQwen emits
    bf16, nothing is lost), mmap without preload, no HNSW graph. A search is
    then an exact scan within the tenant: hundreds of ms for ~500 pages."""

    prefer_grpc: bool = True
    distance: Distance = Distance.COSINE
    datatype: Datatype = Datatype.FLOAT16
    memory: Memory = Memory.COLD
    hnsw_m: int = 0
    on_disk_payload: bool = True
    timeout: int | None = 120


class QdrantEmbeddingStore:
    """
    Single collection partitioned by `doc_id` with an `is_tenant` index, so
    Qdrant stores each document's points contiguously and a scoped search reads
    one block instead of the whole corpus.
    """

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        qdrant_preferences: QdrantPreferences = QdrantPreferences(),
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_preferences = qdrant_preferences
        self.client = QdrantClient(url=url, api_key=api_key,
                                   prefer_grpc=qdrant_preferences.prefer_grpc,
                                   timeout=qdrant_preferences.timeout)

    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        logger.info("creating collection %r (dim=%d)", self.collection_name, dim)
        prefs = self.qdrant_preferences
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=prefs.distance,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                datatype=prefs.datatype,
                memory=prefs.memory,
                hnsw_config=HnswConfigDiff(m=prefs.hnsw_m),
            ),
            on_disk_payload=prefs.on_disk_payload,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=TENANT_KEY,
            field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True),
        )

    def save_many(self,
                  records: Sequence[EmbeddingRecord],
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  wait: bool = True) -> None:

        records = list(records)
        if not records:
            return

        points = [self._to_point(record) for record in records]
        self.ensure_collection(dim=records[0].embedding.shape[1])
        total = len(points)

        for start in range(0, total, batch_size):
            self.client.upsert(collection_name=self.collection_name,
                               points=points[start:start + batch_size],
                               wait=wait)

    def search(self, doc_id: str, query_embedding: np.ndarray, limit: int = 10) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection_name):
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=validate_embedding(query_embedding).tolist(),
            limit=limit,
            query_filter=self._scoped_filter(doc_id),
            with_payload=True,
        )
        return [self._to_hit(point) for point in response.points]

    def count(self, doc_id: str | None = None) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        # `doc_id=None` is the only unscoped read, on purpose.
        count_filter = None if doc_id is None else self._scoped_filter(doc_id)
        return self.client.count(collection_name=self.collection_name,
                                 count_filter=count_filter, exact=True).count

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "QdrantEmbeddingStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @staticmethod
    def _point_id(doc_id: str, record_id: str) -> str:
        """Deterministic UUIDv5 from (doc_id, id): re-ingestion overwrites instead of duplicating."""
        return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}\x00{record_id}"))

    def _to_point(self, record: EmbeddingRecord) -> PointStruct:
        payload = {**record.metadata, TENANT_KEY: record.doc_id, RECORD_ID_KEY: record.id}
        return PointStruct(id=self._point_id(record.doc_id, record.id),
                           vector=validate_embedding(record.embedding).tolist(), payload=payload)

    @staticmethod
    def _to_hit(point: ScoredPoint) -> SearchHit:
        payload = dict(point.payload or {})
        return SearchHit(doc_id=payload.pop(TENANT_KEY, ""),
                         id=payload.pop(RECORD_ID_KEY, str(point.id)),
                         score=point.score, metadata=payload)

    @staticmethod
    def _scoped_filter(doc_id: str) -> Filter:
        return Filter(must=[FieldCondition(key=TENANT_KEY, match=MatchValue(value=doc_id))])
