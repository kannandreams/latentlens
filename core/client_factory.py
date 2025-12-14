"""Helpers to build vector DB clients and demo data."""
from __future__ import annotations

import os
import random
from typing import Callable, Tuple

import numpy as np
import streamlit as st

from core.connectors import ChromaAdapter, QueryWithContext, VectorDBClient, VectorRecord


def demo_embedder(text: str) -> list[float]:
    """Stable pseudo-embedding for demo purposes."""
    rng = random.Random(hash(text) % (2**32))
    return [rng.random() for _ in range(64)]


def openai_embedder(model: str = "text-embedding-3-small") -> Callable[[str], list[float]]:
    from openai import OpenAI

    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or Streamlit secrets.")
    client = OpenAI(api_key=api_key)

    def _embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


def get_embedder(choice: str) -> Callable[[str], list[float]]:
    return demo_embedder if choice == "Demo" else openai_embedder()


def build_chroma_client(
    embedder: Callable[[str], list[float]], collection_name: str | None = None
) -> VectorDBClient:
    import chromadb

    collection_name = collection_name or st.session_state.get("chroma_collection_name")
    if not collection_name:
        collection_name = st.text_input(
            "Chroma Collection Name",
            value="default",
            help="Name of the Chroma collection to read vectors from.",
            key="chroma_collection_fallback",
        )
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    return ChromaAdapter(
        collection=collection,
        embedder=embedder,
    )


def build_client(
    connector: str, embedder: Callable[[str], list[float]], chroma_collection_name: str | None = None
) -> Tuple[VectorDBClient, str]:
    if connector == "Chroma":
        return build_chroma_client(embedder, collection_name=chroma_collection_name), "Chroma"
    raise ValueError(f"Unsupported connector: {connector}")


def generate_demo_context(query: str, top_k: int, background_k: int) -> QueryWithContext:
    """Create synthetic vectors for local demo."""
    dim = 64
    query_vector = np.random.normal(size=dim).tolist()
    results = []
    for i in range(top_k):
        vec = (np.array(query_vector) + np.random.normal(scale=0.2, size=dim)).tolist()
        results.append(
            VectorRecord(
                id=f"demo_result_{i}",
                vector=vec,
                metadata={"content": f"Synthetic result {i}", "source": "demo"},
                score=float(np.random.random()),
            )
        )

    background = []
    for i in range(background_k):
        background.append(
            VectorRecord(
                id=f"bg_{i}",
                vector=np.random.normal(size=dim).tolist(),
                metadata={"content": f"Background vector {i}"},
                score=None,
            )
        )
    return QueryWithContext(
        query=query,
        query_vector=query_vector,
        results=results,
        background=background,
    )
