"""Dimensionality reduction pipeline for the Vector Debugger."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

from core.connectors import QueryWithContext, VectorRecord


@dataclass
class ReducedEmbedding:
    id: str
    label: str
    x: float
    y: float
    z: float
    metadata: Dict
    score: Optional[float] = None
    cosine_sim_to_query: Optional[float] = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _prepare_matrix(records: Iterable[VectorRecord]) -> np.ndarray:
    vectors = [rec.vector for rec in records]
    return np.array(vectors, dtype=float)


def reduce_query_context(
    ctx: QueryWithContext,
    pca_components: int = 50,
    umap_components: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Reduce query, results, and background vectors down to 3D space."""
    # Order matters: first entry is always the query vector.
    labels: List[str] = ["query"]
    base_records: List[VectorRecord] = [
        VectorRecord(id="__query__", vector=ctx.query_vector, metadata={"query": ctx.query})
    ]
    base_records.extend(ctx.results)
    labels.extend(["result"] * len(ctx.results))
    base_records.extend(ctx.background)
    labels.extend(["background"] * len(ctx.background))

    matrix = _prepare_matrix(base_records)
    if matrix.ndim != 2:
        raise ValueError("Input vectors must form a 2D matrix.")

    # Adjust PCA dimensionality based on data volume.
    pca_n = min(pca_components, matrix.shape[1], max(2, matrix.shape[0] - 1))
    pca = PCA(n_components=pca_n, random_state=random_state)
    pca_result = pca.fit_transform(matrix)

    umap_components = min(umap_components, pca_result.shape[1])
    umap_model = UMAP(n_components=umap_components, random_state=random_state)
    reduced = umap_model.fit_transform(pca_result)

    # Pad to 3 dimensions if UMAP returned fewer columns.
    if reduced.shape[1] == 2:
        reduced = np.concatenate([reduced, np.zeros((reduced.shape[0], 1))], axis=1)

    query_vector = matrix[0]
    rows: List[ReducedEmbedding] = []
    for idx, (coords, label, record) in enumerate(zip(reduced, labels, base_records)):
        cosine_sim = None if idx == 0 else _cosine_similarity(query_vector, np.array(record.vector))
        rows.append(
            ReducedEmbedding(
                id=record.id,
                label=label,
                x=float(coords[0]),
                y=float(coords[1]),
                z=float(coords[2]),
                metadata=record.metadata,
                score=record.score,
                cosine_sim_to_query=cosine_sim,
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def detect_void_warning(df: pd.DataFrame, similarity_floor: float = 0.15) -> Optional[str]:
    """Heuristic to flag when query is isolated from search results."""
    result_sims = df[df["label"] == "result"]["cosine_sim_to_query"].dropna()
    if result_sims.empty:
        return "No results returned; query may be an outlier."
    max_sim = result_sims.max()
    if max_sim < similarity_floor:
        return (
            f"Query appears isolated (max cosine similarity {max_sim:.2f}). "
            "Consider adjusting chunking or the embedding model."
        )
    return None
