"""Connector layer for interacting with multiple vector databases.

This module defines a generic ``VectorDBClient`` interface plus adapters for
Pinecone, Chroma, and Qdrant. Each adapter implements a convenience helper
``retrieve_with_background`` that fetches top search results as well as a
random-ish background sample to visualize cluster context.
"""
from __future__ import annotations

import abc
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


Vector = List[float]


@dataclass
class VectorRecord:
    """Unified record returned by all adapters."""

    id: str
    vector: Vector
    metadata: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class QueryWithContext:
    """Container for query results and background sample."""

    query: str
    query_vector: Vector
    results: List[VectorRecord]
    background: List[VectorRecord]


class VectorDBClient(abc.ABC):
    """Abstract base class for vector DB clients."""

    def __init__(self, embedder: Optional[Callable[[str], Vector]] = None) -> None:
        self.embedder = embedder

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 10) -> Tuple[Vector, List[VectorRecord]]:
        """Return the query embedding and top_k matching vectors."""

    @abc.abstractmethod
    def random_sample(self, k: int = 500, exclude_ids: Optional[Set[str]] = None) -> List[VectorRecord]:
        """Return approximately random records for background context."""

    def retrieve_with_background(
        self, query: str, top_k: int = 10, background_k: int = 500
    ) -> QueryWithContext:
        query_vector, results = self.search(query=query, top_k=top_k)
        exclude_ids = {record.id for record in results}
        background = self.random_sample(k=background_k, exclude_ids=exclude_ids)
        if len(background) < background_k:
            background.extend(
                _synthetic_background(
                    needed=background_k - len(background),
                    dim=len(query_vector),
                    existing_ids=exclude_ids | {rec.id for rec in background},
                )
            )
        return QueryWithContext(
            query=query,
            query_vector=query_vector,
            results=results,
            background=background,
        )

    # Utility helpers ----------------------------------------------------- #
    def _embed(self, text: str) -> Vector:
        if not self.embedder:
            raise ValueError("No embedder provided; cannot create query vector.")
        return self.embedder(text)


def _synthetic_background(needed: int, dim: int, existing_ids: Set[str]) -> List[VectorRecord]:
    """Create placeholder background vectors when real samples are too few."""
    records: List[VectorRecord] = []
    counter = 0
    while len(records) < needed:
        counter += 1
        doc_id = f"bg-synth-{counter}"
        if doc_id in existing_ids:
            continue
        vector = [random.gauss(0, 1) for _ in range(dim)]
        records.append(
            VectorRecord(
                id=doc_id,
                vector=vector,
                metadata={"source": "synthetic_background"},
                score=None,
            )
        )
    return records


class PineconeAdapter(VectorDBClient):
    """Adapter for Pinecone. Requires a pre-initialized ``pinecone.Index``."""

    def __init__(
        self,
        index: Any,
        namespace: Optional[str] = None,
        embedder: Optional[Callable[[str], Vector]] = None,
    ) -> None:
        super().__init__(embedder=embedder)
        self.index = index
        self.namespace = namespace

    def search(self, query: str, top_k: int = 10) -> Tuple[Vector, List[VectorRecord]]:
        query_vector = self._embed(query)
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_values=True,
            include_metadata=True,
        )
        records = [
            VectorRecord(
                id=str(match["id"]),
                vector=match["values"],
                metadata=match.get("metadata", {}) or {},
                score=match.get("score"),
            )
            for match in response["matches"]
        ]
        return query_vector, records

    def random_sample(self, k: int = 500, exclude_ids: Optional[Set[str]] = None) -> List[VectorRecord]:
        """Pinecone does not expose random scan; approximate via noisy queries."""
        if not hasattr(self.index, "describe_index_stats"):
            raise ValueError("Index missing describe_index_stats; cannot sample.")

        stats = self.index.describe_index_stats()
        dimension = stats.get("dimension")
        if not dimension:
            # Best-effort fallback to typical 768-dim text embeddings.
            dimension = 768

        def _noise_vector() -> Vector:
            return [random.uniform(-1, 1) for _ in range(dimension)]

        batch_size = min(k, 50)
        collected: List[VectorRecord] = []
        while len(collected) < k:
            response = self.index.query(
                vector=_noise_vector(),
                top_k=batch_size,
                namespace=self.namespace,
                include_values=True,
                include_metadata=True,
            )
            for match in response["matches"]:
                match_id = str(match["id"])
                if exclude_ids and match_id in exclude_ids:
                    continue
                collected.append(
                    VectorRecord(
                        id=match_id,
                        vector=match["values"],
                        metadata=match.get("metadata", {}) or {},
                        score=match.get("score"),
                    )
                )
                if len(collected) >= k:
                    break
        return collected


class ChromaAdapter(VectorDBClient):
    """Adapter for Chroma. Expects a ``chromadb.Collection`` instance."""

    def __init__(
        self,
        collection: Any,
        embedder: Optional[Callable[[str], Vector]] = None,
    ) -> None:
        super().__init__(embedder=embedder)
        self.collection = collection

    def add_text(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed and store a single text snippet in the collection."""
        meta = dict(metadata or {})
        meta.setdefault("content", text)
        embedding = self._embed(text)
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[text],
        )

    def search(self, query: str, top_k: int = 10) -> Tuple[Vector, List[VectorRecord]]:
        query_vector = self._embed(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            # Chroma always returns IDs; only request supported fields.
            include=["embeddings", "metadatas", "distances"],
        )
        records = []
        for idx, vector in enumerate(results["embeddings"][0]):
            records.append(
                VectorRecord(
                    id=str(results["ids"][0][idx]),
                    vector=vector,
                    metadata=results["metadatas"][0][idx] or {},
                    score=results["distances"][0][idx],
                )
            )
        return query_vector, records

    def random_sample(self, k: int = 500, exclude_ids: Optional[Set[str]] = None) -> List[VectorRecord]:
        count = self.collection.count()
        if count == 0:
            return []
        offset = random.randint(0, max(0, count - 1))
        limit = min(k, count - offset)
        results = self.collection.get(
            ids=None,
            where=None,
            limit=limit,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        records: List[VectorRecord] = []
        for idx, vector in enumerate(results["embeddings"]):
            doc_id = str(results["ids"][idx])
            if exclude_ids and doc_id in exclude_ids:
                continue
            records.append(
                VectorRecord(
                    id=doc_id,
                    vector=vector,
                    metadata=results["metadatas"][idx] or {},
                    score=None,
                )
            )
        return records


class QdrantAdapter(VectorDBClient):
    """Adapter for Qdrant using the Python client."""

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedder: Optional[Callable[[str], Vector]] = None,
    ) -> None:
        super().__init__(embedder=embedder)
        self.client = client
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 10) -> Tuple[Vector, List[VectorRecord]]:
        query_vector = self._embed(query)
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        records = [
            VectorRecord(
                id=str(point.id),
                vector=point.vector,
                metadata=point.payload or {},
                score=getattr(point, "score", None),
            )
            for point in response
        ]
        return query_vector, records

    def random_sample(self, k: int = 500, exclude_ids: Optional[Set[str]] = None) -> List[VectorRecord]:
        records: List[VectorRecord] = []
        page_size = min(k, 128)
        offset: Optional[int] = None

        while len(records) < k:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                with_vectors=True,
                with_payload=True,
                limit=page_size,
                offset=offset,
            )
            points, next_offset = scroll_result
            for point in points:
                point_id = str(point.id)
                if exclude_ids and point_id in exclude_ids:
                    continue
                records.append(
                    VectorRecord(
                        id=point_id,
                        vector=point.vector,
                        metadata=point.payload or {},
                        score=None,
                    )
                )
                if len(records) >= k:
                    break
            if next_offset is None:
                # Exhausted the collection.
                break
            offset = next_offset
        return records


def to_plot_points(records: Iterable[VectorRecord]) -> Tuple[List[str], List[Vector], List[Dict[str, Any]], List[Optional[float]]]:
    """Utility to unpack records into simple arrays for downstream math/plotting."""
    ids, vectors, metadata, scores = [], [], [], []
    for record in records:
        ids.append(record.id)
        vectors.append(record.vector)
        metadata.append(record.metadata)
        scores.append(record.score)
    return ids, vectors, metadata, scores
